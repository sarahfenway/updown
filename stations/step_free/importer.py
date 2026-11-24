"""Store TfL's step-free route data against our station records."""

import logging
import re
from dataclasses import dataclass, field

import requests
from django.conf import settings
from django.db import transaction
from django.db.models import F, Max, Q

from stations.models import Station, StationRoutes
from stations.utils import cleanup_station_name

from .tfl_data import DATA_FORMAT, FeedError, merge_routes

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
    # Area codes with no readable name, so labels.py can be extended.
    unnamed: list = field(default_factory=list)


def covered_main_records():
    """Main station records on a network the site covers (as /api/stations/)."""
    return (
        Station.objects.filter(Q(tube=True) | Q(dlr=True) | Q(overground=True) | Q(crossrail=True))
        .filter(Q(parent_station=F("pk")) | Q(parent_station__isnull=True))
        .order_by("pk")
    )


def last_published():
    return StationRoutes.objects.aggregate(latest=Max("published"))["latest"]


def stored_format_is_current():
    """False if any station's stored data predates this code's format."""
    # exclude() alone skips rows where the key is missing, which is exactly
    # the data stored before formats existed.
    outdated = Q(data__format__isnull=True) | ~Q(data__format=DATA_FORMAT)
    return not StationRoutes.objects.filter(outdated).exists()


def download_zip():
    params = {
        key: value
        for key, value in {"app_id": settings.TFL_API_ID, "app_key": settings.TFL_API_KEY}.items()
        if value
    }
    # Stream, so an unexpectedly huge response is refused before it is all
    # in memory.
    with requests.get(FEED_URL, params=params, timeout=(5, 60), stream=True) as response:
        response.raise_for_status()
        chunks, size = [], 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            size += len(chunk)
            if size > MAX_ZIP_BYTES:
                raise FeedError("station data file is unexpectedly large")
            chunks.append(chunk)
    return b"".join(chunks)


def _normalise(name):
    name = cleanup_station_name(name).lower().replace("&", "and")
    return re.sub(r"[^a-z0-9]+", " ", name).strip()


def match_stations(feed):
    """Map our main record pks to TfL station ids.

    Codes first (TfL id, hub code, platform and stop-area NaPTANs against
    our naptan_id/hub_naptan_id on the main record or any child), then a
    unique normalised name. Where two of our main records are the same
    station, the lower pk wins and the other is logged.

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
                by_code.setdefault(code, set()).add(main_pk)
    for code, pks in sorted(by_code.items()):
        if len(pks) > 1:
            logger.warning("Stations %s share code %s; using %s",
                           ", ".join(map(str, sorted(pks))), code, min(pks))

    by_name = {}
    for station in mains:
        by_name.setdefault(_normalise(station.name), []).append(station.pk)

    matches = {}
    by_name_only = []
    unmatched = []
    for tfl_id in sorted(feed.stations):
        hits = [min(by_code[code]) for code in feed.codes[tfl_id] if code in by_code]
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

    with transaction.atomic():
        cleared, _ = StationRoutes.objects.exclude(station_id__in=matches).delete()
        for pk, tfl_ids in matches.items():
            if len(tfl_ids) == 1:
                routes = feed.stations[tfl_ids[0]]
            else:
                routes = merge_routes([feed.stations[tfl_id] for tfl_id in tfl_ids])
            # broken_lift_ids is left alone; the next lift-status run
            # recomputes it against the new data.
            StationRoutes.objects.update_or_create(
                station_id=pk,
                defaults={
                    "tfl_station_id": routes["tfl_id"],
                    "tfl_ids": routes["tfl_ids"],
                    "published": feed.published,
                    "data": routes,
                },
            )
    return ImportResult(matched=len(matches), unmatched=unmatched, cleared=cleared,
                        unnamed=unnamed)
