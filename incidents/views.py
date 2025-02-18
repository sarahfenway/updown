from datetime import datetime
from datetime import timedelta

from django.conf import settings
from django.core.management import call_command
from django.db.models import ExpressionWrapper, DurationField, F, Sum
from django.db.models.functions import Coalesce, Now
from django.http import HttpResponse, HttpResponseNotFound
from django.http import JsonResponse
from django.shortcuts import render
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import csrf_exempt

from incidents.models import Incident
from incidents.utils import get_last_updated
from stations.models import Station


@never_cache
def detail(request):
    if request.headers["host"].endswith("isstpthameslinkliftbroken.com"):
        return stp(request)

    issues = Incident.objects.filter(resolved=False, information=False).order_by(
        "-start_time", "station__parent_station"
    )
    resolved = Incident.objects.filter(
        resolved=True, end_time__gte=timezone.now() - timedelta(hours=12)
    ).order_by("-start_time", "station__parent_station")
    information = Incident.objects.filter(resolved=False, information=True).order_by(
        "-start_time", "station__parent_station"
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
            # Create a new database-level alias:
            station_name=F("station__parent_station__name"),
            station_naptan=F("station__parent_station__naptan_id"),
        )
    )

    resolved_qs = (
        Incident.objects.filter(
            resolved=True, end_time__gte=timezone.now() - timedelta(hours=12)
        )
        .order_by("-start_time", "station__parent_station")
        .annotate(
            # Create a new database-level alias:
            station_name=F("station__parent_station__name"),
            station_naptan=F("station__parent_station__naptan_id"),
        )
    )

    information_qs = (
        Incident.objects.filter(resolved=False, information=True)
        .order_by("-start_time", "station__parent_station")
        .annotate(
            # Create a new database-level alias:
            station_name=F("station__parent_station__name"),
            station_naptan=F("station__parent_station__naptan_id"),
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


def stp(request):
    stp = Station.objects.filter(hub_naptan_id="HUBKGX").first().parent_station
    issues = Incident.objects.filter(resolved=False, information=False, station=stp)

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
    thirty = datetime.now() - timedelta(days=30)
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
            "total_delays": f"{total_delays.days}:{total_delays.seconds // 3600}:{(total_delays.seconds % 3600) // 60}:{(total_delays.seconds % 60)}",
            "station_staff_count": station_staff_count,
            "station_staff_count_percentage": round(
                (station_staff_count / total_count) * 100, 2
            ),
            "station_staff_delays": f"{station_staff_delays.days}:{station_staff_delays.seconds // 3600}:{(station_staff_delays.seconds % 3600) // 60}:{(station_staff_delays.seconds % 60)}",
            "station_staff_delays_percentage": round(
                (station_staff_delays.total_seconds() / total_delays.total_seconds())
                * 100,
                2,
            ),
            "faulty_lift_count": faulty_lift_count,
            "faulty_lift_count_percentage": round(
                (faulty_lift_count / total_count) * 100, 2
            ),
            "faulty_lift_delays": f"{faulty_lift_delays.days}:{faulty_lift_delays.seconds // 3600}:{(faulty_lift_delays.seconds % 3600) // 60}:{(faulty_lift_delays.seconds % 60)}",
            "faulty_lift_delays_percentage": round(
                (faulty_lift_delays.total_seconds() / total_delays.total_seconds())
                * 100,
                2,
            ),
            "planned_maintenance_count": planned_maintenance_count,
            "planned_maintenance_count_percentage": round(
                (planned_maintenance_count / total_count) * 100, 2
            ),
            "planned_maintenance_delays": f"{planned_maintenance_delays.days}:{planned_maintenance_delays.seconds // 3600}:{(planned_maintenance_delays.seconds % 3600) // 60}:{(planned_maintenance_delays.seconds % 60)}",
            "planned_maintenance_delays_percentage": round(
                (
                    planned_maintenance_delays.total_seconds()
                    / total_delays.total_seconds()
                )
                * 100,
                2,
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
