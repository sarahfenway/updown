import os
from datetime import datetime
from datetime import timedelta

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
from incidents.ml import predict_duration
from stations.models import Station


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


def _prepare_incidents(queryset):
    incidents = list(queryset)

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
        else:
            incident.expected_end_time = None

        if incident.prediction_confidence is not None:
            incident.confidence_pct = int(incident.prediction_confidence * 100)
        else:
            incident.confidence_pct = None

        if incident.resolved and incident.end_time and incident.start_time and incident.estimated_duration:
            actual = incident.end_time - incident.start_time
            diff = actual - incident.estimated_duration
            total_minutes = abs(int(diff.total_seconds())) // 60
            hours, minutes = divmod(total_minutes, 60)
            if hours:
                duration_str = f"{hours}h {minutes}m"
            else:
                duration_str = f"{minutes}m"
            if diff.total_seconds() > 0:
                incident.duration_vs_expected = f"{duration_str} longer than expected"
            else:
                incident.duration_vs_expected = f"{duration_str} shorter than expected"
        else:
            incident.duration_vs_expected = None

    return incidents


@never_cache
def detail(request):
    if request.get_host().endswith("isstpthameslinkliftbroken.com"):
        return stp(request)

    issues = _prepare_incidents(
        _incident_queryset()
        .filter(resolved=False, information=False)
        .order_by("-start_time", "station__parent_station")
    )
    resolved = _prepare_incidents(
        _incident_queryset()
        .filter(resolved=True, end_time__gte=timezone.now() - timedelta(hours=12))
        .order_by("-start_time", "station__parent_station")
    )
    information = _prepare_incidents(
        _incident_queryset()
        .filter(resolved=False, information=True)
        .order_by("-start_time", "station__parent_station")
    )

    return render(
        request,
        "home.html",
        {
            "issues": issues,
            "resolved": resolved,
            "information": information,
            "last_updated": get_last_updated(),
        },
    )


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
    issues = _prepare_incidents(
        _incident_queryset().filter(resolved=False, information=False, station=stp)
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
    incidents = Incident.objects.filter(
        resolved=True,
        end_time__gt=since,
        estimated_duration__isnull=False,
    ).annotate(
        actual_duration=ExpressionWrapper(
            F("end_time") - F("start_time"),
            output_field=DurationField(),
        )
    )

    if not incidents.exists():
        return {"prediction_count": 0}

    total_error = timedelta()
    total_abs_error = timedelta()
    abs_errors = []
    count = 0
    # Buckets for confidence vs accuracy chart
    buckets = {}  # confidence_pct -> list of absolute percentage errors

    for inc in incidents:
        error = inc.actual_duration - inc.estimated_duration
        total_error += error
        abs_error = abs(error)
        total_abs_error += abs_error
        abs_errors.append(abs_error)
        count += 1

        if inc.prediction_confidence is not None:
            actual_mins = inc.actual_duration.total_seconds() / 60
            predicted_mins = inc.estimated_duration.total_seconds() / 60
            if actual_mins > 0:
                pct_error = abs(predicted_mins - actual_mins) / actual_mins
                # Bucket by confidence in 10% bands
                bucket = min(9, int(inc.prediction_confidence * 10)) * 10
                buckets.setdefault(bucket, []).append(pct_error)

    abs_errors.sort()
    median_abs_error = abs_errors[len(abs_errors) // 2]

    mean_error = total_error / count
    mean_abs_error = total_abs_error / count

    def _fmt_signed_duration(td):
        total_minutes = int(abs(td.total_seconds())) // 60
        hours, minutes = divmod(total_minutes, 60)
        sign = "+" if td.total_seconds() >= 0 else "-"
        if hours:
            return f"{sign}{hours}h {minutes}m"
        return f"{sign}{minutes}m"

    def _fmt_duration(td):
        total_minutes = int(td.total_seconds()) // 60
        hours, minutes = divmod(total_minutes, 60)
        if hours:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"

    # Build chart data: for each confidence bucket, median percentage error
    chart_height = 160
    chart_data = []
    for i, bucket in enumerate(range(0, 100, 10)):
        errors = buckets.get(bucket, [])
        x = 60 + i * 43
        if errors:
            errors.sort()
            median_pct = errors[len(errors) // 2]
            accuracy = round(max(0, 1 - median_pct) * 100)
            bar_h = round(accuracy / 100 * chart_height)
            chart_data.append({
                "label": f"{bucket}-{bucket + 10}%",
                "accuracy": accuracy,
                "count": len(errors),
                "x": x,
                "y": 20 + chart_height - bar_h,
                "h": bar_h,
                "text_x": x + 17,
            })
        else:
            chart_data.append({
                "label": f"{bucket}-{bucket + 10}%",
                "accuracy": None,
                "count": 0,
                "x": x,
                "y": 170,
                "h": 10,
                "text_x": x + 17,
            })

    return {
        "prediction_count": count,
        "prediction_mean_error": _fmt_signed_duration(mean_error),
        "prediction_mean_abs_error": _fmt_duration(mean_abs_error),
        "prediction_median_abs_error": _fmt_duration(median_abs_error),
        "confidence_chart": chart_data,
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
    key = request.GET.get("key")
    if key != settings.FUNCTIONS_SECRET_KEY:
        return HttpResponseNotFound()

    incidents = list(
        Incident.objects.filter(resolved=True, end_time__isnull=False)
        .select_related("station")
        .prefetch_related("reports")
        .order_by("start_time")
    )

    # Pre-compute per-incident: days since last incident at same station
    last_incident_at_station = {}  # station_id -> most recent start_time

    data = []
    for incident in incidents:
        duration = (incident.end_time - incident.start_time).total_seconds()
        if duration <= 0:
            continue

        station = incident.station
        text = incident.text.lower()

        # Days since last incident at this station
        prev_time = last_incident_at_station.get(station.id)
        if prev_time:
            days_since_last = (incident.start_time - prev_time).total_seconds() / 86400
        else:
            days_since_last = -1  # no prior incident
        last_incident_at_station[station.id] = incident.start_time

        # Concurrent incidents at start time
        concurrent = sum(
            1 for other in incidents
            if other.id != incident.id
            and other.start_time <= incident.start_time
            and (other.end_time is None or other.end_time > incident.start_time)
        )

        data.append(
            {
                "station_id": station.id,
                "station_name": station.name,
                "information": incident.information,
                "start_time": incident.start_time.isoformat(),
                "end_time": incident.end_time.isoformat(),
                "duration_minutes": duration / 60,
                "hour_of_day": incident.start_time.hour,
                "day_of_week": incident.start_time.weekday(),
                "month": incident.start_time.month,
                "text": incident.text,
                "has_faulty_lift": "faulty lift" in text,
                "has_planned_maintenance": "planned maintenance" in text,
                "has_staff_issue": "staff" in text,
                "is_planned_work": (
                    "planned" in text
                    or "until " in text
                    or incident.information
                ),
                "tube": bool(station.tube),
                "dlr": bool(station.dlr),
                "national_rail": bool(station.national_rail),
                "crossrail": bool(station.crossrail),
                "overground": bool(station.overground),
                "access_via_lift": bool(station.access_via_lift),
                "num_reports": incident.reports.count(),
                "days_since_last_incident": round(days_since_last, 2),
                "concurrent_incidents": concurrent,
                "estimated_duration_minutes": (
                    incident.estimated_duration.total_seconds() / 60
                    if incident.estimated_duration
                    else None
                ),
            }
        )

    return JsonResponse({"incidents": data})


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

        model_path = os.path.join(settings.BASE_DIR, "ml_model.joblib")
        with open(model_path, "wb") as f:
            for chunk in model_file.chunks():
                f.write(chunk)

        # Clear the cached model so it's reloaded on next prediction
        predict_duration.cache_clear()

        return JsonResponse({"status": "ok"})
