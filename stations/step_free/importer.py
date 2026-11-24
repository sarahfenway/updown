"""Store TfL's step-free route data on our station records."""

import logging
import re
from dataclasses import dataclass, field

import requests
from django.conf import settings
from django.db import transaction
from django.db.models import F, Max, Q

from stations.models import Station
from stations.utils import cleanup_station_name

from .tfl_data import FeedError, merge_routes

logger = logging.getLogger(__name__)

FEED_URL = "https://api.tfl.gov.uk/stationdata/tfl-stationdata-detailed.zip"
MAX_ZIP_BYTES = 20 * 1024 * 1024
# The file covers about 420 of our stations. Far fewer means a broken file.
MIN_STATIONS = 200


class ImportAborted(Exception):
    """The file would leave too few stations with route data."""


@dataclass
class ImportResult:
    matched: int
    unmatched: list = field(default_factory=list)
    cleared: int = 0


def covered_main_records():
    """Main station records on a network the site covers (as /api/stations/)."""
    return (
        Station.objects.filter(Q(tube=True) | Q(dlr=True) | Q(overground=True) | Q(crossrail=True))
        .filter(Q(parent_station=F("pk")) | Q(parent_station__isnull=True))
        .order_by("pk")
    )


def last_published():
    return Station.objects.aggregate(latest=Max("step_free_routes_published"))["latest"]


def download_zip():
    params = {
        key: value
        for key, value in {"app_id": settings.TFL_API_ID, "app_key": settings.TFL_API_KEY}.items()
        if value
    }
    response = requests.get(FEED_URL, params=params, timeout=(5, 60))
    response.raise_for_status()
    if len(response.content) > MAX_ZIP_BYTES:
        raise FeedError("station data file is unexpectedly large")
    return response.content


def _normalise(name):
    name = cleanup_station_name(name).lower().replace("&", "and")
    return re.sub(r"[^a-z0-9]+", " ", name).strip()


def match_stations(feed):
    """Map our main record pks to TfL station ids.

    Codes first (TfL id, hub code, platform and stop-area NaPTANs against
    our naptan_id/hub_naptan_id on the main record or any child), then a
    unique normalised name. Where two of our main records are the same
    station, the lower pk wins.

    Returns ``({main pk: [tfl id, ...]}, [unmatched tfl id, ...])``. Several
    TfL stations map to one record when they share our codes (West
    Hampstead's Tube and Overground share hub HUBWHD); a name match never
    joins a record that already has a station, since names are ambiguous.
    """
    mains = list(covered_main_records().only("pk", "name"))
    main_pks = {station.pk for station in mains}

    by_code = {}
    rows = Station.objects.filter(Q(pk__in=main_pks) | Q(parent_station_id__in=main_pks))
    for station in rows.order_by("pk").only("pk", "parent_station_id", "naptan_id", "hub_naptan_id"):
        main_pk = station.pk if station.pk in main_pks else station.parent_station_id
        for code in (station.naptan_id, station.hub_naptan_id):
            if code:
                by_code.setdefault(code, main_pk)
                by_code[code] = min(by_code[code], main_pk)

    by_name = {}
    for station in mains:
        by_name.setdefault(_normalise(station.name), []).append(station.pk)

    matches = {}
    by_name_only = []
    unmatched = []
    for tfl_id in sorted(feed.stations):
        hits = [by_code[code] for code in feed.codes[tfl_id] if code in by_code]
        if hits:
            matches.setdefault(min(hits), []).append(tfl_id)
        else:
            by_name_only.append(tfl_id)
    for tfl_id in by_name_only:
        candidates = by_name.get(_normalise(feed.stations[tfl_id]["name"]), [])
        pk = min(candidates) if candidates else None
        if pk is None:
            unmatched.append(tfl_id)
        elif pk in matches:
            logger.warning("TfL station %s matches station %s by name only, but it already "
                           "has %s; skipping", tfl_id, pk, ", ".join(matches[pk]))
            unmatched.append(tfl_id)
        else:
            matches[pk] = [tfl_id]
    return matches, unmatched


def import_feed(feed, min_stations=None):
    if min_stations is None:
        min_stations = MIN_STATIONS
    matches, unmatched = match_stations(feed)
    if len(matches) < min_stations:
        raise ImportAborted(
            f"only {len(matches)} stations matched (need {min_stations}); keeping previous data"
        )

    unnamed = sorted({
        place["code"]
        for routes in feed.stations.values()
        for place in routes["places"].values()
        if place["kind"] == "place" and place["name"] == place["code"]
    })
    if unnamed:
        logger.info("No readable name for %d area codes: %s", len(unnamed), ", ".join(unnamed))
    if unmatched:
        logger.info("TfL stations with no matching station: %s", ", ".join(unmatched))

    with transaction.atomic():
        cleared = (
            Station.objects.exclude(pk__in=matches)
            .filter(Q(step_free_routes__isnull=False) | Q(tfl_station_id__isnull=False))
            .update(step_free_routes=None, step_free_routes_published=None, tfl_station_id=None)
        )
        for pk, tfl_ids in matches.items():
            if len(tfl_ids) == 1:
                routes = feed.stations[tfl_ids[0]]
            else:
                routes = merge_routes([feed.stations[tfl_id] for tfl_id in tfl_ids])
            Station.objects.filter(pk=pk).update(
                step_free_routes=routes,
                step_free_routes_published=feed.published,
                tfl_station_id=routes["tfl_id"],
            )
    return ImportResult(matched=len(matches), unmatched=unmatched, cleared=cleared)
