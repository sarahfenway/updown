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


def _station_slugs():
    """Page addresses for every covered station.

    Returns ``(lookup, listed)``: ``lookup`` maps slug to station pk, and
    ``listed`` is ``[(slug, label)]`` for stations with route data, for the
    index. Different stations that share a name (the two Bethnal Greens,
    the two Edgware Roads) each get a slug with their first line added,
    e.g. ``bethnal-green-weaver-line``. The plain name still resolves, to
    the record with route data and then the lower pk, so the site's own
    duplicate records of one station stay on one page.
    """
    groups = {}
    for pk, name, tfl_id in covered_main_records().values_list("pk", "name", "tfl_station_id"):
        groups.setdefault(slugify(name), []).append((pk, name, tfl_id))

    lookup, listed = {}, []
    for base, group in groups.items():
        pk, name, _ = min(group, key=lambda row: (row[2] is None, row[0]))
        lookup[base] = pk
        with_data = {}
        for row_pk, row_name, tfl_id in sorted(group):
            if tfl_id and tfl_id not in with_data:
                with_data[tfl_id] = (row_pk, row_name)
        if len(with_data) < 2:
            if with_data:
                listed.append((base, name))
            continue
        platforms = dict(
            Station.objects.filter(pk__in=[row_pk for row_pk, _ in with_data.values()])
            .values_list("pk", "step_free_routes__platforms")
        )
        for tfl_id, (row_pk, row_name) in with_data.items():
            lines = sorted({line for p in (platforms.get(row_pk) or {}).values() for line in p["lines"]})
            slug = slugify(f"{row_name} {lines[0] if lines else tfl_id}")
            lookup[slug] = row_pk
            listed.append((slug, f"{row_name} ({', '.join(lines) or tfl_id})"))
    listed.sort(key=lambda item: item[1])
    return lookup, listed


def step_free_index(request):
    chosen = request.GET.get("station")
    if chosen:
        return redirect("step_free_station", slug=slugify(chosen))
    _, options = _station_slugs()
    return render(request, "step_free_index.html", {"options": options})


def _station_for_slug(slug):
    lookup, _ = _station_slugs()
    if slug not in lookup:
        raise Http404("No such station")
    return Station.objects.get(pk=lookup[slug])


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
