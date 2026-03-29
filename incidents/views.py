from datetime import datetime
from datetime import timedelta

from django.conf import settings
from django.core.management import call_command
from django.db.models import CharField, Prefetch, Q
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
    qs = Incident.objects.filter(end_time__gt=thirty)
    total_count = qs.count()
    total_delays = qs.annotate(
        active_duration=ExpressionWrapper(
            Coalesce("end_time", Now()) - F("start_time"), output_field=DurationField()
        )
    ).aggregate(total=Sum("active_duration"))["total"]

    station_staff = qs.filter(text__regex=r"^.*unavailability of (?:station )?staff.*$")
    station_staff_count = station_staff.count()
    station_staff_delays = station_staff.annotate(
        active_duration=ExpressionWrapper(
            Coalesce("end_time", Now()) - F("start_time"), output_field=DurationField()
        )
    ).aggregate(total=Sum("active_duration"))["total"]

    faulty_lift = qs.filter(text__regex="faulty lift")
    faulty_lift_count = faulty_lift.count()
    faulty_lift_delays = faulty_lift.annotate(
        active_duration=ExpressionWrapper(
            Coalesce("end_time", Now()) - F("start_time"), output_field=DurationField()
        )
    ).aggregate(total=Sum("active_duration"))["total"]

    planned_maintenance = qs.filter(text__regex="planned maintenance")
    planned_maintenance_count = planned_maintenance.count()
    planned_maintenance_delays = planned_maintenance.annotate(
        active_duration=ExpressionWrapper(
            Coalesce("end_time", Now()) - F("start_time"), output_field=DurationField()
        )
    ).aggregate(total=Sum("active_duration"))["total"]

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
        },
    )


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
