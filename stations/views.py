from datetime import timedelta

from django.db.models import Q
from django.http import Http404
from django.shortcuts import redirect, render
from django.utils import timezone
from django.utils.text import slugify

from incidents.models import Incident
from stations.models import Station
from stations.step_free.diagram import build_diagram
from stations.step_free.importer import covered_main_records
from stations.step_free.routing import BLOCKED, NO_ROUTE, OK, find_route

LIFT_STATUS_MAX_AGE = timedelta(minutes=30)


def step_free_index(request):
    chosen = request.GET.get("station")
    if chosen:
        return redirect("step_free_station", slug=slugify(chosen))
    names = covered_main_records().filter(tfl_station_id__isnull=False).values_list("name", flat=True)
    options = sorted({slugify(name): name for name in names}.items(), key=lambda item: item[1])
    return render(request, "step_free_index.html", {"options": options})


def _station_for_slug(slug):
    rows = covered_main_records().values_list("pk", "name", "tfl_station_id")
    matches = [(tfl_id is None, pk) for pk, name, tfl_id in rows if slugify(name) == slug]
    if not matches:
        raise Http404("No such station")
    _, pk = min(matches)  # prefer the record holding route data, then lower pk
    return Station.objects.get(pk=pk)


def step_free_station(request, slug):
    station = _station_for_slug(slug)
    issues = Incident.objects.filter(
        Q(station=station) | Q(station__parent_station=station),
        resolved=False,
        information=False,
    ).order_by("start_time")
    context = {"station": station, "issues": issues, "routes": station.step_free_routes}

    routes = station.step_free_routes
    if routes:
        checked = station.broken_lifts_checked_at
        known = checked is not None and timezone.now() - checked <= LIFT_STATUS_MAX_AGE
        broken = (station.broken_lift_ids or []) if known else []
        platform_id = request.GET.get("platform")
        route = find_route(routes, platform_id, broken) if platform_id in routes["platforms"] else None
        broken_names = sorted({
            link["lift_name"] for link in routes["links"]
            if link["kind"] == "lift" and link["lift_id"] in broken
        })
        context.update(
            lift_status_known=known,
            broken_names=broken_names,
            platform_options=sorted(
                ((pid, platform["label"]) for pid, platform in routes["platforms"].items()),
                key=lambda item: item[1],
            ),
            chosen=platform_id if route else None,
            chosen_label=routes["platforms"][platform_id]["label"] if route else None,
            route=route,
            OK=OK, BLOCKED=BLOCKED, NO_ROUTE=NO_ROUTE,
            diagram=build_diagram(routes, broken, route if route and route.status == OK else None),
        )
    return render(request, "step_free_station.html", context)
