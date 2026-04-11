import os
from datetime import datetime
from datetime import timedelta
from math import sqrt

from django.conf import settings
from django.core.management import call_command
from django.db.models import CharField, Count, Prefetch, Q
from django.db.models import ExpressionWrapper, DurationField, F, Sum
from django.db.models.functions import Coalesce
from django.db.models.functions import Now
from django.http import HttpResponse, HttpResponseNotFound
from django.http import JsonResponse
from django.shortcuts import render
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import csrf_exempt

from incidents.models import Incident, Report
from incidents.utils import get_last_updated
from incidents.ml import (
    PREDICTION_BOUNDARY_GRACE,
    format_block_slot,
    predict_duration,
    prediction_is_close_enough,
    prediction_outcome,
    time_block_slot,
)
from incidents.text_features import normalise_incident_text
from stations.models import Station

PREDICTION_POLICY_WINDOW_DAYS = 30
PREDICTION_POLICY_FALLBACK_WINDOW_DAYS = (90, 180)
PREDICTION_BUCKET_MIN_SAMPLES = 15
PREDICTION_BUCKET_MIN_LOWER_BOUND = 0.55
BETA_LONG_RANGE_DAYS = 7


def _format_duration(duration):
    if duration is None:
        return "0:0:0:0"

    return (
        f"{duration.days}:{duration.seconds // 3600}:"
        f"{(duration.seconds % 3600) // 60}:{duration.seconds % 60}"
    )


def _percentage(part, whole):
    if not part or not whole:
        return 0

    return round((part / whole) * 100, 2)


def _duration_percentage(part, whole):
    if not part or not whole:
        return 0

    return round((part.total_seconds() / whole.total_seconds()) * 100, 2)


def _incident_queryset():
    report_queryset = Report.objects.only("id", "source", "end_time").order_by("id")

    return Incident.objects.select_related(
        "station", "station__parent_station"
    ).prefetch_related(
        Prefetch("reports", queryset=report_queryset, to_attr="prefetched_reports")
    )


def _prediction_confidence_bucket(confidence):
    if confidence is None:
        return None
    return min(9, int(confidence * 10)) * 10


def _prediction_confidence_label(accuracy):
    if accuracy is None:
        return None
    if accuracy < 0.7:
        return "maybe"
    if accuracy < 0.85:
        return "likely"
    return "very likely"


def _beta_current_status_text(label, expected_end_time, expected_block, now):
    if expected_end_time is None or expected_block is None:
        return None
    if expected_end_time <= now:
        return None
    if expected_end_time > now + timedelta(days=BETA_LONG_RANGE_DAYS):
        return "More than a week"
    if not label:
        return None
    return f"{label.capitalize()} {expected_block}"


def _beta_resolved_status_text(issue):
    if not issue.show_prediction:
        return None
    if issue.prediction_outcome == "exact":
        return "Right"
    if issue.prediction_outcome == "near":
        return "Nearly right"
    return "Wrong"


def _beta_meter_help_text():
    return "The meter shows how certain we feel based on recent similar predictions."


def _beta_current_status_title(status_text):
    if not status_text:
        return None
    return f"AI prediction: {status_text}. {_beta_meter_help_text()}"


def _beta_resolved_status_title(issue):
    if not issue.beta_resolved_status:
        return None
    if issue.prediction_outcome == "exact":
        return (
            f"AI prediction: right. We expected {issue.expected_block}, and it was fixed then. "
            f"{_beta_meter_help_text()}"
        )
    if issue.prediction_outcome == "near":
        return (
            f"AI prediction: nearly right. We expected {issue.expected_block}, but it was fixed "
            f"{issue.actual_block}, just outside that window. {_beta_meter_help_text()}"
        )
    return (
        f"AI prediction: wrong. We expected {issue.expected_block}, but it was fixed "
        f"{issue.actual_block}. {_beta_meter_help_text()}"
    )


def _wilson_lower_bound(successes, total, z=1.96):
    if total == 0:
        return 0.0

    phat = successes / total
    denom = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
    return max(0.0, (centre - margin) / denom)


def _prediction_bucket_metrics(since):
    incidents = Incident.objects.filter(
        resolved=True,
        end_time__gt=since,
        estimated_duration__isnull=False,
        start_time__isnull=False,
        end_time__isnull=False,
    ).only("start_time", "end_time", "estimated_duration", "prediction_confidence")

    count = 0
    correct = 0
    buckets = {}

    for inc in incidents:
        predicted_end = inc.start_time + inc.estimated_duration
        is_correct = prediction_is_close_enough(predicted_end, inc.end_time)
        count += 1
        if is_correct:
            correct += 1

        bucket = _prediction_confidence_bucket(inc.prediction_confidence)
        if bucket is None:
            continue

        entry = buckets.setdefault(bucket, {"count": 0, "correct": 0})
        entry["count"] += 1
        if is_correct:
            entry["correct"] += 1

    for bucket in range(0, 100, 10):
        entry = buckets.setdefault(bucket, {"count": 0, "correct": 0})
        if entry["count"] > 0:
            accuracy = entry["correct"] / entry["count"]
            lower_bound = _wilson_lower_bound(entry["correct"], entry["count"])
        else:
            accuracy = None
            lower_bound = 0.0

        show_prediction = (
            entry["count"] >= PREDICTION_BUCKET_MIN_SAMPLES
            and lower_bound >= PREDICTION_BUCKET_MIN_LOWER_BOUND
        )
        entry.update(
            {
                "bucket": bucket,
                "accuracy": accuracy,
                "accuracy_pct": round(accuracy * 100) if accuracy is not None else None,
                "lower_bound": lower_bound,
                "lower_bound_pct": round(lower_bound * 100),
                "show_prediction": show_prediction,
                "label": (
                    _prediction_confidence_label(accuracy) if show_prediction else None
                ),
            }
        )

    return {
        "prediction_count": count,
        "prediction_correct_count": correct,
        "prediction_accuracy_pct": round(correct / count * 100) if count else 0,
        "buckets": buckets,
    }


def _prediction_window_metrics(now=None, windows=None):
    if now is None:
        now = timezone.now()
    if windows is None:
        windows = (PREDICTION_POLICY_WINDOW_DAYS,)

    windows = tuple(sorted(set(windows)))
    oldest_since = now - timedelta(days=windows[-1])
    incidents = Incident.objects.filter(
        resolved=True,
        end_time__gt=oldest_since,
        estimated_duration__isnull=False,
        start_time__isnull=False,
        end_time__isnull=False,
    ).only("start_time", "end_time", "estimated_duration", "prediction_confidence")

    raw_metrics = {
        days: {"prediction_count": 0, "prediction_correct_count": 0, "buckets": {}}
        for days in windows
    }
    window_starts = {days: now - timedelta(days=days) for days in windows}

    for inc in incidents:
        predicted_end = inc.start_time + inc.estimated_duration
        is_correct = prediction_is_close_enough(predicted_end, inc.end_time)
        bucket = _prediction_confidence_bucket(inc.prediction_confidence)

        for days in windows:
            if inc.end_time <= window_starts[days]:
                continue

            metrics = raw_metrics[days]
            metrics["prediction_count"] += 1
            if is_correct:
                metrics["prediction_correct_count"] += 1

            if bucket is None:
                continue

            entry = metrics["buckets"].setdefault(bucket, {"count": 0, "correct": 0})
            entry["count"] += 1
            if is_correct:
                entry["correct"] += 1

    finalised = {}
    for days in windows:
        metrics = raw_metrics[days]
        buckets = metrics["buckets"]
        for bucket in range(0, 100, 10):
            entry = buckets.setdefault(bucket, {"count": 0, "correct": 0})
            if entry["count"] > 0:
                accuracy = entry["correct"] / entry["count"]
                lower_bound = _wilson_lower_bound(entry["correct"], entry["count"])
            else:
                accuracy = None
                lower_bound = 0.0

            show_prediction = (
                entry["count"] >= PREDICTION_BUCKET_MIN_SAMPLES
                and lower_bound >= PREDICTION_BUCKET_MIN_LOWER_BOUND
            )
            entry.update(
                {
                    "bucket": bucket,
                    "accuracy": accuracy,
                    "accuracy_pct": (
                        round(accuracy * 100) if accuracy is not None else None
                    ),
                    "lower_bound": lower_bound,
                    "lower_bound_pct": round(lower_bound * 100),
                    "show_prediction": show_prediction,
                    "label": (
                        _prediction_confidence_label(accuracy)
                        if show_prediction
                        else None
                    ),
                }
            )

        count = metrics["prediction_count"]
        correct = metrics["prediction_correct_count"]
        finalised[days] = {
            "prediction_count": count,
            "prediction_correct_count": correct,
            "prediction_accuracy_pct": round(correct / count * 100) if count else 0,
            "buckets": buckets,
        }

    return finalised


def _prediction_display_policy_from_window_metrics(metrics_by_window):
    windows = (PREDICTION_POLICY_WINDOW_DAYS,) + PREDICTION_POLICY_FALLBACK_WINDOW_DAYS
    policy = {}

    for bucket in range(0, 100, 10):
        selected_window_days = windows[-1]
        selected_entry = metrics_by_window[selected_window_days]["buckets"][bucket]
        primary_entry = metrics_by_window[PREDICTION_POLICY_WINDOW_DAYS]["buckets"][bucket]

        if primary_entry["count"] >= PREDICTION_BUCKET_MIN_SAMPLES:
            selected_window_days = PREDICTION_POLICY_WINDOW_DAYS
            selected_entry = primary_entry
        else:
            for days in PREDICTION_POLICY_FALLBACK_WINDOW_DAYS:
                entry = metrics_by_window[days]["buckets"][bucket]
                if entry["count"] >= PREDICTION_BUCKET_MIN_SAMPLES:
                    selected_window_days = days
                    selected_entry = entry
                    break

        show_prediction = (
            selected_entry["count"] >= PREDICTION_BUCKET_MIN_SAMPLES
            and selected_entry["lower_bound"] >= PREDICTION_BUCKET_MIN_LOWER_BOUND
        )
        policy[bucket] = {
            **selected_entry,
            "bucket": bucket,
            "window_days": selected_window_days,
            "used_fallback_window": selected_window_days != PREDICTION_POLICY_WINDOW_DAYS,
            "show_prediction": show_prediction,
            "label": (
                _prediction_confidence_label(selected_entry["accuracy"])
                if show_prediction
                else None
            ),
        }

    return policy


def _prediction_display_policy(now=None):
    if now is None:
        now = timezone.now()

    windows = (PREDICTION_POLICY_WINDOW_DAYS,) + PREDICTION_POLICY_FALLBACK_WINDOW_DAYS
    metrics_by_window = _prediction_window_metrics(now, windows)
    return _prediction_display_policy_from_window_metrics(metrics_by_window)


def _prepare_incidents(queryset, prediction_policy=None):
    incidents = list(queryset)
    now = timezone.now()
    if prediction_policy is None:
        prediction_policy = _prediction_display_policy(now)

    for incident in incidents:
        reports = getattr(incident, "prefetched_reports", [])
        incident.reports_count = len(reports)
        incident.is_single_user_report = (
            incident.reports_count == 1 and reports[0].source == Report.SOURCE_USER
        )
        incident.single_user_report_end_time = (
            reports[0].end_time if incident.is_single_user_report else None
        )

        if incident.start_time and incident.estimated_duration:
            incident.expected_end_time = incident.start_time + incident.estimated_duration
            incident.expected_block = format_block_slot(
                time_block_slot(incident.expected_end_time), now
            )
        else:
            incident.expected_end_time = None
            incident.expected_block = None

        if incident.prediction_confidence is not None:
            incident.confidence_pct = int(incident.prediction_confidence * 100)
        else:
            incident.confidence_pct = None
        bucket = _prediction_confidence_bucket(incident.prediction_confidence)
        bucket_policy = prediction_policy.get(bucket, {}) if bucket is not None else {}
        incident.prediction_label = bucket_policy.get("label")
        incident.prediction_accuracy_pct = bucket_policy.get("accuracy_pct")
        incident.show_prediction = bool(
            incident.expected_block is not None
            and bucket_policy.get("show_prediction")
            and (
                incident.resolved
                or (
                    incident.expected_end_time is not None
                    and incident.expected_end_time > now
                )
            )
        )
        incident.beta_current_status = _beta_current_status_text(
            incident.prediction_label,
            incident.expected_end_time,
            incident.expected_block,
            now,
        )
        incident.beta_current_status_title = _beta_current_status_title(
            incident.beta_current_status
        )
        incident.beta_meter_help_text = _beta_meter_help_text()

        if (
            incident.resolved
            and incident.end_time
            and incident.start_time
            and incident.estimated_duration
        ):
            predicted_end = incident.start_time + incident.estimated_duration
            actual_slot = time_block_slot(incident.end_time)
            incident.prediction_outcome = prediction_outcome(
                predicted_end, incident.end_time
            )
            incident.prediction_was_correct = incident.prediction_outcome != "miss"
            incident.prediction_was_nearly_right = incident.prediction_outcome == "near"
            incident.actual_block = format_block_slot(actual_slot, now)
            incident.beta_resolved_status = _beta_resolved_status_text(incident)
            incident.beta_resolved_status_title = _beta_resolved_status_title(incident)
        else:
            incident.prediction_outcome = None
            incident.prediction_was_correct = None
            incident.prediction_was_nearly_right = None
            incident.actual_block = None
            incident.beta_resolved_status = None
            incident.beta_resolved_status_title = None

    return incidents


def _beta_status_copy(issues, resolved, information):
    current_issue_count = len(issues)
    affected_station_count = len({issue.station_id for issue in issues})
    predicted_issue_count = sum(1 for issue in issues if issue.show_prediction)
    resolved_count = len(resolved)
    information_count = len(information)

    if current_issue_count == 0:
        heading = "No reported step-free access issues"
        summary = (
            "We currently have no reports of step-free access problems on the "
            "tracked TfL networks."
        )
        if resolved_count:
            summary += (
                f" {resolved_count} issue{' has' if resolved_count == 1 else 's have'} "
                "been resolved in the last 12 hours."
            )
        if information_count:
            summary += (
                f" There {'is' if information_count == 1 else 'are'} "
                f"{information_count} information notice"
                f"{'' if information_count == 1 else 's'} to check."
            )
        tone = "clear"
    else:
        heading = (
            f"{current_issue_count} current step-free access issue"
            f"{'' if current_issue_count == 1 else 's'}"
        )
        summary = (
            f"Affecting {affected_station_count} station"
            f"{'' if affected_station_count == 1 else 's'} right now."
        )
        if predicted_issue_count:
            summary += (
                f" Estimated fix windows are shown for {predicted_issue_count} "
                f"issue{'' if predicted_issue_count == 1 else 's'}."
            )
        if information_count:
            summary += (
                f" There {'is' if information_count == 1 else 'are'} "
                f"{information_count} information notice"
                f"{'' if information_count == 1 else 's'} as well."
            )
        tone = "alert"

    return {
        "current_issue_count": current_issue_count,
        "affected_station_count": affected_station_count,
        "predicted_issue_count": predicted_issue_count,
        "resolved_recent_count": resolved_count,
        "information_count": information_count,
        "beta_status_heading": heading,
        "beta_status_summary": summary,
        "beta_status_tone": tone,
    }


def _status_page_context():
    prediction_policy = _prediction_display_policy(timezone.now())
    issues = _prepare_incidents(
        _incident_queryset()
        .filter(resolved=False, information=False)
        .order_by("-start_time", "station__parent_station"),
        prediction_policy=prediction_policy,
    )
    resolved = _prepare_incidents(
        _incident_queryset()
        .filter(resolved=True, end_time__gte=timezone.now() - timedelta(hours=12))
        .order_by("-start_time", "station__parent_station"),
        prediction_policy=prediction_policy,
    )
    information = _prepare_incidents(
        _incident_queryset()
        .filter(resolved=False, information=True)
        .order_by("-start_time", "station__parent_station"),
        prediction_policy=prediction_policy,
    )

    context = {
        "issues": issues,
        "resolved": resolved,
        "information": information,
        "last_updated": get_last_updated(),
    }
    context.update(_beta_status_copy(issues, resolved, information))
    return context


@never_cache
def detail(request):
    if request.get_host().endswith("isstpthameslinkliftbroken.com"):
        return stp(request)

    return render(request, "home.html", _status_page_context())


@never_cache
def beta_detail(request):
    return render(request, "beta.html", _status_page_context())


@never_cache
def api_incidents(request):
    # Fetch querysets
    issues_qs = (
        Incident.objects.filter(resolved=False, information=False)
        .order_by("-start_time", "station__parent_station")
        .annotate(
            station_name=Coalesce(
                "station__parent_station__name",
                "station__name",
                output_field=CharField(),
            ),
            station_naptan=Coalesce(
                "station__parent_station__naptan_id",
                "station__parent_station__hub_naptan_id",
                "station__naptan_id",
                "station__hub_naptan_id",
                output_field=CharField(),
            ),
        )
    )

    resolved_qs = (
        Incident.objects.filter(
            resolved=True, end_time__gte=timezone.now() - timedelta(hours=12)
        )
        .order_by("-start_time", "station__parent_station")
        .annotate(
            station_name=Coalesce(
                "station__parent_station__name",
                "station__name",
                output_field=CharField(),
            ),
            station_naptan=Coalesce(
                "station__parent_station__naptan_id",
                "station__parent_station__hub_naptan_id",
                "station__naptan_id",
                "station__hub_naptan_id",
                output_field=CharField(),
            ),
        )
    )

    information_qs = (
        Incident.objects.filter(resolved=False, information=True)
        .order_by("-start_time", "station__parent_station")
        .annotate(
            station_name=Coalesce(
                "station__parent_station__name",
                "station__name",
                output_field=CharField(),
            ),
            station_naptan=Coalesce(
                "station__parent_station__naptan_id",
                "station__parent_station__hub_naptan_id",
                "station__naptan_id",
                "station__hub_naptan_id",
                output_field=CharField(),
            ),
        )
    )

    # Convert them into something JSON-friendly.
    # You can specify exactly which fields you want via .values().
    issues_data = list(
        issues_qs.values(
            "id",
            "text",
            "start_time",
            "end_time",
            "resolved",
            "information",
            "station_name",
            "station_naptan",
        )
    )
    resolved_data = list(
        resolved_qs.values(
            "id",
            "text",
            "start_time",
            "end_time",
            "resolved",
            "information",
            "station_name",
            "station_naptan",
        )
    )
    information_data = list(
        information_qs.values(
            "id",
            "text",
            "start_time",
            "end_time",
            "resolved",
            "information",
            "station_name",
            "station_naptan",
        )
    )

    data = {
        "issues": issues_data,
        "resolved": resolved_data,
        "information": information_data,
        "last_updated": datetime.now().isoformat(),
    }

    return JsonResponse(data)


def api_stations(request):
    stations_qs = (
        Station.objects.filter(
            Q(tube=True) | Q(dlr=True) | Q(overground=True) | Q(crossrail=True)
        )
        .filter(Q(parent_station=F("pk")) | Q(parent_station__isnull=True))
        .annotate(
            station_name=Coalesce(
                "parent_station__name",
                "name",
                output_field=CharField(),
            ),
            station_naptan=Coalesce(
                "parent_station__naptan_id",
                "parent_station__hub_naptan_id",
                "naptan_id",
                "hub_naptan_id",
                output_field=CharField(),
            ),
        )
        .exclude(Q(station_name__isnull=True) | Q(station_naptan__isnull=True))
        .order_by("station_name")
    )

    stations = list(
        stations_qs.values(
            "station_name",
            "station_naptan",
        )
    )

    data = {
        "stations": stations,
    }

    return JsonResponse(data)


def stp(request):
    station = (
        Station.objects.select_related("parent_station")
        .only("id", "parent_station_id")
        .filter(hub_naptan_id="HUBKGX")
        .first()
    )
    stp = station.parent_station if station else None
    prediction_policy = _prediction_display_policy(timezone.now())
    issues = _prepare_incidents(
        _incident_queryset().filter(resolved=False, information=False, station=stp),
        prediction_policy=prediction_policy,
    )

    yes_or_no = False

    for issue in issues:
        if (
            "to the thameslink" in issue.text.lower()
            and "faulty lift" in issue.text.lower()
        ):
            yes_or_no = True

    return render(
        request,
        "stp.html",
        {
            "issues": issues,
            "yes_or_no": yes_or_no,
            "last_updated": get_last_updated(),
        },
    )


def stats(request):
    thirty = timezone.now() - timedelta(days=30)
    station_staff_filter = Q(text__regex=r"^.*unavailability of (?:station )?staff.*$")
    faulty_lift_filter = Q(text__regex="faulty lift")
    planned_maintenance_filter = Q(text__regex="planned maintenance")

    stats_data = (
        Incident.objects.filter(end_time__gt=thirty)
        .annotate(
            active_duration=ExpressionWrapper(
                Coalesce("end_time", Now()) - F("start_time"),
                output_field=DurationField(),
            )
        )
        .aggregate(
            total_count=Count("id"),
            total_delays=Sum("active_duration"),
            station_staff_count=Count("id", filter=station_staff_filter),
            station_staff_delays=Sum(
                "active_duration",
                filter=station_staff_filter,
            ),
            faulty_lift_count=Count("id", filter=faulty_lift_filter),
            faulty_lift_delays=Sum(
                "active_duration",
                filter=faulty_lift_filter,
            ),
            planned_maintenance_count=Count(
                "id",
                filter=planned_maintenance_filter,
            ),
            planned_maintenance_delays=Sum(
                "active_duration",
                filter=planned_maintenance_filter,
            ),
        )
    )

    total_count = stats_data["total_count"]
    total_delays = stats_data["total_delays"]
    station_staff_count = stats_data["station_staff_count"]
    station_staff_delays = stats_data["station_staff_delays"]
    faulty_lift_count = stats_data["faulty_lift_count"]
    faulty_lift_delays = stats_data["faulty_lift_delays"]
    planned_maintenance_count = stats_data["planned_maintenance_count"]
    planned_maintenance_delays = stats_data["planned_maintenance_delays"]

    return render(
        request,
        "stats.html",
        {
            "total_count": total_count,
            "total_delays": _format_duration(total_delays),
            "station_staff_count": station_staff_count,
            "station_staff_count_percentage": _percentage(
                station_staff_count, total_count
            ),
            "station_staff_delays": _format_duration(station_staff_delays),
            "station_staff_delays_percentage": _duration_percentage(
                station_staff_delays, total_delays
            ),
            "faulty_lift_count": faulty_lift_count,
            "faulty_lift_count_percentage": _percentage(faulty_lift_count, total_count),
            "faulty_lift_delays": _format_duration(faulty_lift_delays),
            "faulty_lift_delays_percentage": _duration_percentage(
                faulty_lift_delays, total_delays
            ),
            "planned_maintenance_count": planned_maintenance_count,
            "planned_maintenance_count_percentage": _percentage(
                planned_maintenance_count, total_count
            ),
            "planned_maintenance_delays": _format_duration(planned_maintenance_delays),
            "planned_maintenance_delays_percentage": _duration_percentage(
                planned_maintenance_delays, total_delays
            ),
            "last_updated": get_last_updated(),
            **_prediction_stats(thirty),
        },
    )


def _prediction_stats(since):
    now = timezone.now()
    windows = (PREDICTION_POLICY_WINDOW_DAYS,) + PREDICTION_POLICY_FALLBACK_WINDOW_DAYS
    window_metrics = _prediction_window_metrics(now, windows)
    metrics = window_metrics[PREDICTION_POLICY_WINDOW_DAYS]
    display_policy = _prediction_display_policy_from_window_metrics(window_metrics)
    count = metrics["prediction_count"]
    correct = metrics["prediction_correct_count"]
    buckets = metrics["buckets"]

    if count == 0:
        return {"prediction_count": 0}

    # Build chart data: % correct per confidence bucket, 0–100 on Y axis.
    chart_height = 160
    chart_data = []
    for i, bucket in enumerate(range(0, 100, 10)):
        x = 60 + i * 43
        entry = buckets.get(bucket)
        display_entry = display_policy.get(bucket, {})
        if entry and entry["count"] > 0:
            pct = entry["accuracy"] * 100
            bar_h = round(pct / 100 * chart_height)
            chart_data.append({
                "label": f"{bucket}-{bucket + 10}%",
                "accuracy_pct": round(pct),
                "count": entry["count"],
                "show_prediction": display_entry.get("show_prediction", False),
                "prediction_label": display_entry.get("label"),
                "policy_window_days": display_entry.get("window_days"),
                "used_fallback_window": display_entry.get(
                    "used_fallback_window", False
                ),
                "x": x,
                "y": 20 + chart_height - bar_h,
                "h": bar_h,
                "text_x": x + 17,
            })
        else:
            chart_data.append({
                "label": f"{bucket}-{bucket + 10}%",
                "accuracy_pct": None,
                "count": 0,
                "x": x,
                "y": 170,
                "h": 10,
                "text_x": x + 17,
            })

    return {
        "prediction_count": count,
        "prediction_correct_count": correct,
        "prediction_accuracy_pct": metrics["prediction_accuracy_pct"],
        "confidence_chart": chart_data,
        "prediction_boundary_grace_minutes": int(
            PREDICTION_BOUNDARY_GRACE.total_seconds() / 60
        ),
        "prediction_display_min_samples": PREDICTION_BUCKET_MIN_SAMPLES,
        "prediction_display_min_accuracy_pct": round(
            PREDICTION_BUCKET_MIN_LOWER_BOUND * 100
        ),
        "prediction_display_primary_window_days": PREDICTION_POLICY_WINDOW_DAYS,
        "prediction_display_fallback_window_days": ", ".join(
            str(days) for days in PREDICTION_POLICY_FALLBACK_WINDOW_DAYS
        ),
    }


def alexa(request):
    issues = Incident.objects.filter(resolved=False, information=False).order_by(
        "station__parent_station"
    )

    if issues.count() == 0:
        alexa_string = "There are currently no reported step free access issues on the \
            Transport for London network."
    else:
        alexa_string = "There are step free access issues at: "
        alexa_string += ", ".join(
            sorted(issues.values_list("station__parent_station__name", flat=True))[0:-1]
        )

        if issues.count() > 1:
            alexa_string += " and "

        alexa_string += sorted(
            issues.values_list("station__parent_station__name", flat=True)
        )[-1]

    alexa_string = alexa_string.replace("&", "and")

    return HttpResponse(alexa_string)


@method_decorator(csrf_exempt, name="dispatch")
class UpdateIncidentsView(View):
    def post(self, request, *args, **kwargs):
        if request.POST.get("key") == settings.FUNCTIONS_SECRET_KEY:
            call_command("update_incidents")
            return HttpResponse(status=204)
        return HttpResponseNotFound()


def api_training_data(request):
    import json

    key = request.GET.get("key")
    if key != settings.FUNCTIONS_SECRET_KEY:
        return HttpResponseNotFound()

    base_qs = Incident.objects.filter(resolved=True, end_time__isnull=False)

    # Pre-compute concurrent incidents with a sweep line using lightweight query
    events = []
    for inc_id, start, end in base_qs.values_list("id", "start_time", "end_time"):
        events.append((start, 1, inc_id))
        events.append((end, -1, inc_id))
    events.sort(key=lambda e: (e[0], e[1]))

    concurrent_at = {}
    active = 0
    for _, delta, inc_id in events:
        if delta == 1:
            concurrent_at[inc_id] = active
            active += 1
        else:
            active -= 1
    del events

    # Pre-compute report counts
    report_counts = dict(
        base_qs.annotate(n=Count("reports")).values_list("id", "n")
    )

    # Cache stations to avoid repeated lookups
    station_cache = {
        s.id: s for s in Station.objects.all()
    }

    # Resolve each station to its "effective" (parent if present, else self)
    # station. All training signals are keyed by this — a station like Bank
    # has one record per line but should be treated as one place for
    # predictions.
    def _effective(station):
        if station.parent_station_id and station.parent_station_id in station_cache:
            return station_cache[station.parent_station_id]
        return station

    last_incident_at_station = {}

    def generate():
        yield '{"incidents":['
        first = True
        for incident in (
            base_qs.order_by("start_time")
            .only(
                "id", "station_id", "information", "text",
                "start_time", "end_time", "estimated_duration",
            )
            .iterator(chunk_size=500)
        ):
            duration = (incident.end_time - incident.start_time).total_seconds()
            if duration <= 0:
                continue

            raw_station = station_cache.get(incident.station_id)
            if raw_station is None:
                continue
            station = _effective(raw_station)
            start_time_local = timezone.localtime(incident.start_time)
            text = normalise_incident_text(incident.text)
            text_lower = text.lower()

            prev_time = last_incident_at_station.get(station.id)
            if prev_time:
                days_since_last = (incident.start_time - prev_time).total_seconds() / 86400
            else:
                days_since_last = -1
            last_incident_at_station[station.id] = incident.start_time

            row = {
                "station_id": station.id,
                "station_name": station.name,
                "information": incident.information,
                "start_time": incident.start_time.isoformat(),
                "end_time": incident.end_time.isoformat(),
                "duration_minutes": duration / 60,
                "hour_of_day": start_time_local.hour,
                "day_of_week": start_time_local.weekday(),
                "month": start_time_local.month,
                "text": text,
                "has_faulty_lift": "faulty lift" in text_lower,
                "has_planned_maintenance": "planned maintenance" in text_lower,
                "has_staff_issue": "staff" in text_lower,
                "is_planned_work": (
                    "planned" in text_lower
                    or "until " in text_lower
                    or incident.information
                ),
                "tube": bool(station.tube),
                "dlr": bool(station.dlr),
                "national_rail": bool(station.national_rail),
                "crossrail": bool(station.crossrail),
                "overground": bool(station.overground),
                "access_via_lift": bool(station.access_via_lift),
                "num_reports": report_counts.get(incident.id, 0),
                "days_since_last_incident": round(days_since_last, 2),
                "concurrent_incidents": concurrent_at.get(incident.id, 0),
                "estimated_duration_minutes": (
                    incident.estimated_duration.total_seconds() / 60
                    if incident.estimated_duration
                    else None
                ),
            }

            if not first:
                yield ","
            first = False
            yield json.dumps(row)
        yield "]}"

    from django.http import StreamingHttpResponse
    response = StreamingHttpResponse(generate(), content_type="application/json")
    return response


@method_decorator(csrf_exempt, name="dispatch")
class UploadModelView(View):
    def post(self, request, *args, **kwargs):
        key = request.POST.get("key")
        if key != settings.FUNCTIONS_SECRET_KEY:
            return HttpResponseNotFound()

        model_file = request.FILES.get("model")
        if not model_file:
            return JsonResponse({"error": "No model file provided"}, status=400)

        max_size = 10 * 1024 * 1024  # 10 MB
        if model_file.size > max_size:
            return JsonResponse({"error": "Model file too large"}, status=400)

        # Read the whole upload into memory — capped at 10 MB above so this
        # is safe — and stash it in a new MLModel row. We keep historic rows
        # around rather than overwriting, so rollbacks are trivial: just
        # delete the bad row and _load_model() picks up the previous one.
        from incidents.models import MLModel

        raw = b"".join(model_file.chunks())
        MLModel.objects.create(data=raw, size_bytes=len(raw))

        # Prune: keep the last 5 uploads, drop the rest.
        old_ids = list(
            MLModel.objects.order_by("-id").values_list("id", flat=True)[5:]
        )
        if old_ids:
            MLModel.objects.filter(id__in=old_ids).delete()

        # Invalidate this process's cache so the new model is picked up.
        # Other dynos will pick it up on their next _load_model() call,
        # because that checks for the latest row id.
        predict_duration.cache_clear()

        return JsonResponse({"status": "ok", "size_bytes": len(raw)})
