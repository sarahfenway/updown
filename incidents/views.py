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
PREDICTION_STATION_DIAGNOSTIC_MIN_SAMPLES = 3
PREDICTION_STATION_OVERRIDE_BUCKET = 40
PREDICTION_STATION_OVERRIDE_WINDOW_DAYS = 180
PREDICTION_STATION_OVERRIDE_MIN_SAMPLES = 20
PREDICTION_STATION_OVERRIDE_MIN_ACCURACY = 0.65
BETA_LONG_RANGE_DAYS = 7
PREDICTION_REVIEW_LIMIT = 50
PREDICTION_NETWORK_LABELS = (
    ("tube", "Tube"),
    ("dlr", "DLR"),
    ("national_rail", "National rail"),
    ("crossrail", "Crossrail"),
    ("overground", "Overground"),
)


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


def _prediction_confidence_bucket_label(bucket):
    if bucket is None:
        return "Unknown"
    return f"{bucket}-{bucket + 10}%"


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
    return f"{label.capitalize()} by {expected_block}"


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
            f"AI prediction: right. We said it would be fixed by {issue.expected_block}, "
            f"and it was. {_beta_meter_help_text()}"
        )
    if issue.prediction_outcome == "near":
        return (
            f"AI prediction: nearly right. We said it would be fixed by "
            f"{issue.expected_block}; it was actually fixed {issue.actual_block}. "
            f"{_beta_meter_help_text()}"
        )
    return (
        f"AI prediction: wrong. We said it would be fixed by {issue.expected_block}, "
        f"but it was not fixed until {issue.actual_block}. {_beta_meter_help_text()}"
    )


def _prediction_hidden_reason(expected_end_time, expected_block, bucket_policy, now):
    if expected_end_time is None or expected_block is None:
        return "missing_prediction"
    if not bucket_policy.get("show_prediction"):
        if bucket_policy.get("count", 0) < PREDICTION_BUCKET_MIN_SAMPLES:
            return "bucket_too_sparse"
        return "bucket_accuracy_too_low"
    if expected_end_time <= now:
        return "past_due"
    return None


def _format_absolute_block_slot(slot):
    date, block = slot
    return f"{date.day} {date.strftime('%b')} {block}"


def _prediction_review_outcome_text(outcome):
    if outcome == "exact":
        return "Right"
    if outcome == "near":
        return "Nearly right"
    return "Wrong"


def _effective_station(station):
    return station.parent_station or station


def _prediction_issue_category(text):
    text_lower = (text or "").lower()
    if "faulty lift" in text_lower:
        return "Faulty lift"
    if "planned maintenance" in text_lower:
        return "Planned maintenance"
    if "staff" in text_lower:
        return "Staff issue"
    return "Other"


def _prediction_issue_networks(station):
    station = _effective_station(station)
    labels = [
        label for field, label in PREDICTION_NETWORK_LABELS if getattr(station, field, False)
    ]
    return labels or ["Unknown"]


def _count_rows_from_mapping(mapping, labels=None):
    if labels is None:
        labels = sorted(mapping)
    return [
        {"label": label, "count": mapping[label]}
        for label in labels
        if mapping.get(label, 0) > 0
    ]


def _hidden_low_accuracy_diagnostics(current_issues, resolved_incidents):
    low_accuracy_issues = [
        issue
        for issue in current_issues
        if issue.prediction_hidden_reason == "bucket_accuracy_too_low"
    ]

    bucket_counts = {}
    category_counts = {
        "Faulty lift": 0,
        "Planned maintenance": 0,
        "Staff issue": 0,
        "Other": 0,
    }
    network_counts = {label: 0 for _, label in PREDICTION_NETWORK_LABELS}
    station_history = {}
    station_current_counts = {}

    for incident in resolved_incidents:
        station = _effective_station(incident.station)
        history = station_history.setdefault(
            station.id,
            {
                "station_name": station.name,
                "prediction_count": 0,
                "correct_count": 0,
            },
        )
        history["prediction_count"] += 1
        predicted_end = incident.start_time + incident.estimated_duration
        if prediction_is_close_enough(predicted_end, incident.end_time):
            history["correct_count"] += 1

    for issue in low_accuracy_issues:
        bucket_label = _prediction_confidence_bucket_label(
            _prediction_confidence_bucket(issue.prediction_confidence)
        )
        bucket_counts[bucket_label] = bucket_counts.get(bucket_label, 0) + 1
        category_counts[_prediction_issue_category(issue.text)] += 1

        station = _effective_station(issue.station)
        station_entry = station_current_counts.setdefault(
            station.id,
            {"station_name": station.name, "current_hidden_count": 0},
        )
        station_entry["current_hidden_count"] += 1

        for network_label in _prediction_issue_networks(station):
            network_counts[network_label] = network_counts.get(network_label, 0) + 1

    station_rows = []
    for station_id, station_entry in station_current_counts.items():
        history = station_history.get(
            station_id, {"prediction_count": 0, "correct_count": 0}
        )
        prediction_count = history["prediction_count"]
        has_enough_history = (
            prediction_count >= PREDICTION_STATION_DIAGNOSTIC_MIN_SAMPLES
        )
        station_rows.append(
            {
                "station_name": station_entry["station_name"],
                "current_hidden_count": station_entry["current_hidden_count"],
                "recent_prediction_count": prediction_count,
                "recent_accuracy_pct": (
                    round(history["correct_count"] / prediction_count * 100)
                    if has_enough_history and prediction_count
                    else None
                ),
                "has_enough_history": has_enough_history,
            }
        )

    station_rows.sort(
        key=lambda row: (
            -row["current_hidden_count"],
            -(row["recent_prediction_count"] or 0),
            row["station_name"],
        )
    )

    return {
        "current_hidden_low_accuracy_bucket_rows": _count_rows_from_mapping(
            bucket_counts,
            labels=[
                _prediction_confidence_bucket_label(bucket)
                for bucket in range(0, 100, 10)
            ],
        ),
        "current_hidden_low_accuracy_category_rows": _count_rows_from_mapping(
            category_counts,
            labels=["Faulty lift", "Planned maintenance", "Staff issue", "Other"],
        ),
        "current_hidden_low_accuracy_network_rows": _count_rows_from_mapping(
            network_counts,
            labels=[label for _, label in PREDICTION_NETWORK_LABELS] + ["Unknown"],
        ),
        "current_hidden_low_accuracy_station_rows": station_rows,
    }


def _prediction_station_overrides(current_issues, prediction_policy, now=None):
    if now is None:
        now = timezone.now()

    candidate_station_ids = set()

    for issue in current_issues:
        bucket = _prediction_confidence_bucket(issue.prediction_confidence)
        if bucket != PREDICTION_STATION_OVERRIDE_BUCKET:
            continue

        bucket_policy = prediction_policy.get(bucket, {})
        if bucket_policy.get("show_prediction"):
            continue

        expected_end_time = None
        expected_block = None
        if issue.start_time and issue.estimated_duration:
            expected_end_time = issue.start_time + issue.estimated_duration
            expected_block = format_block_slot(time_block_slot(expected_end_time), now)

        hidden_reason = _prediction_hidden_reason(
            expected_end_time,
            expected_block,
            bucket_policy,
            now,
        )
        if hidden_reason != "bucket_accuracy_too_low":
            continue

        candidate_station_ids.add(_effective_station(issue.station).id)

    if not candidate_station_ids:
        return {}

    since = now - timedelta(days=PREDICTION_STATION_OVERRIDE_WINDOW_DAYS)
    station_history = {}
    history_incidents = (
        Incident.objects.filter(
            resolved=True,
            end_time__gt=since,
            estimated_duration__isnull=False,
            start_time__isnull=False,
            end_time__isnull=False,
        )
        .filter(
            Q(station_id__in=candidate_station_ids)
            | Q(station__parent_station_id__in=candidate_station_ids)
        )
        .select_related("station", "station__parent_station")
        .only(
            "station_id",
            "station__parent_station_id",
            "start_time",
            "end_time",
            "estimated_duration",
        )
    )

    for incident in history_incidents:
        station_id = _effective_station(incident.station).id
        history = station_history.setdefault(
            station_id,
            {"prediction_count": 0, "correct_count": 0},
        )
        history["prediction_count"] += 1
        predicted_end = incident.start_time + incident.estimated_duration
        if prediction_is_close_enough(predicted_end, incident.end_time):
            history["correct_count"] += 1

    overrides = {}
    for station_id, history in station_history.items():
        prediction_count = history["prediction_count"]
        if prediction_count < PREDICTION_STATION_OVERRIDE_MIN_SAMPLES:
            continue

        accuracy = history["correct_count"] / prediction_count
        if accuracy < PREDICTION_STATION_OVERRIDE_MIN_ACCURACY:
            continue

        overrides[station_id] = {
            "show_prediction": True,
            "label": "maybe",
            "accuracy": accuracy,
            "accuracy_pct": round(accuracy * 100),
            "station_override": True,
            "station_prediction_count": prediction_count,
        }

    return overrides


def _wilson_lower_bound(successes, total, z=1.96):
    if total == 0:
        return 0.0

    phat = successes / total
    denom = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
    return max(0.0, (centre - margin) / denom)


def _prediction_bucket_metrics(since):
    now = timezone.now()
    return _prediction_window_metrics(now, (max(1, (now - since).days),)).get(
        max(1, (now - since).days), {"prediction_count": 0, "buckets": {}}
    )


def _prediction_window_metrics_from_incidents(incidents, now, windows):
    windows = tuple(sorted(set(windows)))
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
    return _prediction_window_metrics_from_incidents(incidents, now, windows)


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


def _prepare_incidents(
    queryset,
    prediction_policy=None,
    station_prediction_overrides=None,
    now=None,
):
    incidents = list(queryset)
    if now is None:
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
        incident.used_station_prediction_override = False
        if (
            station_prediction_overrides
            and bucket == PREDICTION_STATION_OVERRIDE_BUCKET
        ):
            station_override = station_prediction_overrides.get(
                _effective_station(incident.station).id
            )
            if station_override:
                bucket_policy = {**bucket_policy, **station_override}
                incident.used_station_prediction_override = True
        incident.prediction_label = bucket_policy.get("label")
        incident.prediction_accuracy_pct = bucket_policy.get("accuracy_pct")
        incident.prediction_hidden_reason = (
            None
            if incident.resolved
            else _prediction_hidden_reason(
                incident.expected_end_time,
                incident.expected_block,
                bucket_policy,
                now,
            )
        )
        incident.show_prediction = bool(
            incident.resolved or incident.prediction_hidden_reason is None
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
    now = timezone.now()
    prediction_policy = _prediction_display_policy(now)
    issue_queryset = (
        _incident_queryset()
        .filter(resolved=False, information=False)
        .order_by("-start_time", "station__parent_station")
    )
    raw_issues = list(issue_queryset)
    station_prediction_overrides = _prediction_station_overrides(
        raw_issues,
        prediction_policy,
        now,
    )
    issues = _prepare_incidents(
        raw_issues,
        prediction_policy=prediction_policy,
        station_prediction_overrides=station_prediction_overrides,
        now=now,
    )
    resolved = _prepare_incidents(
        _incident_queryset()
        .filter(resolved=True, end_time__gte=now - timedelta(hours=12))
        .order_by("-start_time", "station__parent_station"),
        prediction_policy=prediction_policy,
        now=now,
    )
    information = _prepare_incidents(
        _incident_queryset()
        .filter(resolved=False, information=True)
        .order_by("-start_time", "station__parent_station"),
        prediction_policy=prediction_policy,
        now=now,
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
    API_FIELDS = (
        "id", "text", "start_time", "end_time",
        "resolved", "information", "station_name", "station_naptan",
    )

    base_qs = Incident.objects.order_by(
        "-start_time", "station__parent_station"
    ).annotate(
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

    data = {
        "issues": list(
            base_qs.filter(resolved=False, information=False).values(*API_FIELDS)
        ),
        "resolved": list(
            base_qs.filter(
                resolved=True, end_time__gte=timezone.now() - timedelta(hours=12)
            ).values(*API_FIELDS)
        ),
        "information": list(
            base_qs.filter(resolved=False, information=True).values(*API_FIELDS)
        ),
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
    oldest_since = now - timedelta(days=max(windows))
    incidents = list(
        Incident.objects.filter(
            resolved=True,
            end_time__gt=oldest_since,
            estimated_duration__isnull=False,
            start_time__isnull=False,
            end_time__isnull=False,
        )
        .select_related("station", "station__parent_station")
        .only(
            "id",
            "station__name",
            "station__parent_station__name",
            "start_time",
            "end_time",
            "estimated_duration",
            "prediction_confidence",
        )
        .order_by("-end_time", "-id")
    )
    window_metrics = _prediction_window_metrics_from_incidents(incidents, now, windows)
    metrics = window_metrics[PREDICTION_POLICY_WINDOW_DAYS]
    display_policy = _prediction_display_policy_from_window_metrics(window_metrics)
    count = metrics["prediction_count"]
    correct = metrics["prediction_correct_count"]
    buckets = metrics["buckets"]
    recent_prediction_rows = []
    current_issue_queryset = (
        Incident.objects.select_related("station", "station__parent_station")
        .filter(resolved=False, information=False)
        .only(
            "id",
            "station_id",
            "station__name",
            "station__parent_station__name",
            "station__parent_station_id",
            "information",
            "text",
            "start_time",
            "estimated_duration",
            "prediction_confidence",
            "resolved",
        )
        .order_by("-start_time", "-id")
    )
    raw_current_issues = list(current_issue_queryset)
    station_prediction_overrides = _prediction_station_overrides(
        raw_current_issues,
        display_policy,
        now,
    )
    current_issues = _prepare_incidents(
        raw_current_issues,
        prediction_policy=display_policy,
        station_prediction_overrides=station_prediction_overrides,
        now=now,
    )
    current_prediction_visible_count = sum(
        1 for issue in current_issues if issue.show_prediction
    )
    current_prediction_hidden_sparse_count = sum(
        1
        for issue in current_issues
        if issue.prediction_hidden_reason == "bucket_too_sparse"
    )
    current_prediction_hidden_low_accuracy_count = sum(
        1
        for issue in current_issues
        if issue.prediction_hidden_reason == "bucket_accuracy_too_low"
    )
    current_prediction_hidden_past_due_count = sum(
        1 for issue in current_issues if issue.prediction_hidden_reason == "past_due"
    )
    current_prediction_hidden_missing_count = sum(
        1
        for issue in current_issues
        if issue.prediction_hidden_reason == "missing_prediction"
    )
    low_accuracy_diagnostics = _hidden_low_accuracy_diagnostics(current_issues, incidents)

    for incident in incidents:
        if incident.end_time <= since or len(recent_prediction_rows) >= PREDICTION_REVIEW_LIMIT:
            continue

        predicted_end = incident.start_time + incident.estimated_duration
        outcome = prediction_outcome(predicted_end, incident.end_time)
        station = incident.station.parent_station or incident.station
        recent_prediction_rows.append(
            {
                "station_name": station.name,
                "predicted_block": _format_absolute_block_slot(
                    time_block_slot(predicted_end)
                ),
                "actual_block": _format_absolute_block_slot(
                    time_block_slot(incident.end_time)
                ),
                "actual_end_time": incident.end_time,
                "confidence_pct": round((incident.prediction_confidence or 0) * 100),
                "outcome": outcome,
                "outcome_text": _prediction_review_outcome_text(outcome),
            }
        )

    if count == 0:
        return {"prediction_count": 0, "recent_prediction_rows": []}

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
        "prediction_station_override_bucket_label": _prediction_confidence_bucket_label(
            PREDICTION_STATION_OVERRIDE_BUCKET
        ),
        "prediction_station_override_window_days": (
            PREDICTION_STATION_OVERRIDE_WINDOW_DAYS
        ),
        "prediction_station_override_min_samples": (
            PREDICTION_STATION_OVERRIDE_MIN_SAMPLES
        ),
        "prediction_station_override_min_accuracy_pct": round(
            PREDICTION_STATION_OVERRIDE_MIN_ACCURACY * 100
        ),
        "prediction_station_diagnostic_min_samples": (
            PREDICTION_STATION_DIAGNOSTIC_MIN_SAMPLES
        ),
        "prediction_station_diagnostic_window_days": max(windows),
        "current_issue_count": len(current_issues),
        "current_prediction_visible_count": current_prediction_visible_count,
        "current_prediction_hidden_sparse_count": (
            current_prediction_hidden_sparse_count
        ),
        "current_prediction_hidden_low_accuracy_count": (
            current_prediction_hidden_low_accuracy_count
        ),
        "current_prediction_hidden_past_due_count": (
            current_prediction_hidden_past_due_count
        ),
        "current_prediction_hidden_missing_count": current_prediction_hidden_missing_count,
        "recent_prediction_rows": recent_prediction_rows,
        "prediction_review_limit": PREDICTION_REVIEW_LIMIT,
        **low_accuracy_diagnostics,
    }


def alexa(request):
    station_names = sorted(
        Incident.objects.filter(
            resolved=False, information=False
        ).values_list("station__parent_station__name", flat=True)
    )

    if not station_names:
        alexa_string = "There are currently no reported step free access issues on the \
            Transport for London network."
    elif len(station_names) == 1:
        alexa_string = "There are step free access issues at: " + station_names[0]
    else:
        alexa_string = "There are step free access issues at: "
        alexa_string += ", ".join(station_names[:-1])
        alexa_string += " and " + station_names[-1]

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
    import heapq
    import json

    key = request.GET.get("key")
    if key != settings.FUNCTIONS_SECRET_KEY:
        return HttpResponseNotFound()

    base_qs = Incident.objects.filter(resolved=True, end_time__isnull=False)

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
    active_end_times = []
    incident_rows = (
        base_qs.annotate(num_reports=Count("reports"))
        .order_by("start_time", "id")
        .values_list(
            "id",
            "station_id",
            "information",
            "text",
            "start_time",
            "end_time",
            "estimated_duration",
            "num_reports",
        )
    )

    def generate():
        yield '{"incidents":['
        first = True
        for (
            _inc_id,
            station_id,
            information,
            raw_text,
            start_time,
            end_time,
            estimated_duration,
            num_reports,
        ) in incident_rows.iterator(chunk_size=500):
            duration = (end_time - start_time).total_seconds()
            if duration <= 0:
                continue

            while active_end_times and active_end_times[0] <= start_time:
                heapq.heappop(active_end_times)
            concurrent = len(active_end_times)
            heapq.heappush(active_end_times, end_time)

            raw_station = station_cache.get(station_id)
            if raw_station is None:
                continue
            station = _effective(raw_station)
            start_time_local = timezone.localtime(start_time)
            text = normalise_incident_text(raw_text)
            text_lower = text.lower()

            prev_time = last_incident_at_station.get(station.id)
            if prev_time:
                days_since_last = (start_time - prev_time).total_seconds() / 86400
            else:
                days_since_last = -1
            last_incident_at_station[station.id] = start_time

            row = {
                "station_id": station.id,
                "station_name": station.name,
                "information": information,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
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
                    or information
                ),
                "tube": bool(station.tube),
                "dlr": bool(station.dlr),
                "national_rail": bool(station.national_rail),
                "crossrail": bool(station.crossrail),
                "overground": bool(station.overground),
                "access_via_lift": bool(station.access_via_lift),
                "num_reports": num_reports,
                "days_since_last_incident": round(days_since_last, 2),
                "concurrent_incidents": concurrent,
                "estimated_duration_minutes": (
                    estimated_duration.total_seconds() / 60
                    if estimated_duration
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
