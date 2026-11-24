import threading
from datetime import timedelta
from io import StringIO

from django.core.management import CommandError, call_command
from django.db.models import Q
from django.http import Http404, HttpResponse, HttpResponseNotFound
from django.shortcuts import redirect, render
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.utils.text import slugify
from django.views import View
from django.views.decorators.csrf import csrf_exempt

from incidents.models import Incident
from incidents.views import _request_functions_key, _valid_functions_key
from stations.models import Station, StationRoutes
from stations.step_free.broken_lifts import lift_status_checked_at
from stations.step_free.details import (
    build_panels, facility_lines, platform_groups, station_status, station_summary,
)
from stations.step_free.importer import covered_main_records
from stations.step_free.journey import build_journey
from stations.step_free.routing import BLOCKED, NO_INFO, NO_ROUTE, OK, find_route

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
    rows = covered_main_records().values_list("pk", "name", "step_free__tfl_station_id")
    for pk, name, tfl_id in rows:
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
            StationRoutes.objects.filter(pk__in=[row_pk for row_pk, _ in with_data.values()])
            .values_list("pk", "data__platforms")
        )
        for tfl_id, (row_pk, row_name) in with_data.items():
            lines = sorted({line for p in (platforms.get(row_pk) or {}).values() for line in p["lines"]})
            slug = slugify(f"{row_name} {lines[0] if lines else tfl_id}")
            lookup[slug] = row_pk
            listed.append((slug, f"{row_name} ({', '.join(lines) or tfl_id})"))
    listed.sort(key=lambda item: item[1])
    return lookup, listed


def step_free_index(request):
    # A value with no letters or digits slugifies to "", which the
    # station URL can't take; show the index instead of a 500.
    chosen = slugify(request.GET.get("station") or "")
    if chosen:
        return redirect("step_free_station", slug=chosen)
    _, options = _station_slugs()
    return render(request, "step_free_index.html", {"options": options})


def _station_for_slug(slug):
    lookup, _ = _station_slugs()
    if slug not in lookup:
        raise Http404("No such station")
    return Station.objects.get(pk=lookup[slug])


def step_free_station(request, slug):
    station = _station_for_slug(slug)
    # The site has a few duplicate records of one station (same name); the
    # page shows one, so take issues from the others too, but not from a
    # different station that happens to share the name and has routes.
    same_station = [
        pk for pk, name, tfl_id in
        covered_main_records().values_list("pk", "name", "step_free__tfl_station_id")
        if slugify(name) == slugify(station.name) and (pk == station.pk or tfl_id is None)
    ]
    issues = Incident.objects.filter(
        Q(station__in=same_station) | Q(station__parent_station__in=same_station),
        resolved=False,
        information=False,
    ).order_by("start_time")
    stored = StationRoutes.objects.filter(station=station).first()
    routes = stored.data if stored else None
    context = {"station": station, "issues": issues, "routes": routes}

    if routes:
        checked = lift_status_checked_at()
        known = checked is not None and timezone.now() - checked <= LIFT_STATUS_MAX_AGE
        broken = (stored.broken_lift_ids or []) if known else []
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
            OK=OK, BLOCKED=BLOCKED, NO_ROUTE=NO_ROUTE, NO_INFO=NO_INFO,
        )
        journey = build_journey(routes, route if route and route.status == OK else None, known)
        summary = station_summary(routes)
        context.update(
            journey=journey,
            panels=build_panels(routes, journey, broken, known),
            summary=summary,
            status=station_status(summary),
            groups=platform_groups(routes, broken, known),
            facilities=facility_lines(routes.get("facilities")),
            platform_count=len(routes["platforms"]),
        )
    return render(request, "step_free_station.html", context)


_import_lock = threading.Lock()


@method_decorator(csrf_exempt, name="dispatch")
class ImportStationRoutesView(View):
    """Run the weekly step-free import inside the web process.

    The cron POSTs here, as it does for update_incidents, rather than
    starting a second Django process: that would add ~100MB on a 512MB
    machine and would write to the database past the maintenance freeze,
    which works by intercepting HTTP requests.
    """

    def post(self, request, *args, **kwargs):
        if not _valid_functions_key(_request_functions_key(request, request.POST)):
            return HttpResponseNotFound()
        if not _import_lock.acquire(blocking=False):
            return HttpResponse("import already running", status=409)
        try:
            out = StringIO()
            call_command("import_station_routes", stdout=out)
        except CommandError as exc:
            return HttpResponse(f"import failed: {exc}", status=500, content_type="text/plain")
        finally:
            _import_lock.release()
        # The command's report (unmatched stations, unnamed area codes) goes
        # to the process's stdout, which Fly keeps in its logs.
        print(out.getvalue(), end="", flush=True)
        return HttpResponse(status=204)
