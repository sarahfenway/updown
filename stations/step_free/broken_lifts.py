"""Record which lifts TfL says are out of service, for the station pages.

Uses TfL's v2 lift feed because it names lifts by the same ids as the
station data file. This is separate from the incident sources, which are
not changed.

It runs as the last phase of every 5-minute update_incidents run, so it
must never raise: an optional feed must not stop the incident update.
"""

import logging

import requests
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from stations.models import LiftOutage, LiftStatusCheck, StationRoutes

logger = logging.getLogger(__name__)

LIFTS_V2_URL = "https://api.tfl.gov.uk/Disruptions/Lifts/v2/"


def update_broken_lifts():
    """Store broken lift ids per station. Returns True if stored."""
    try:
        return _update_broken_lifts()
    except Exception as exc:  # noqa: BLE001 - see module docstring
        logger.warning("Lift status update failed (%s); keeping previous data",
                       type(exc).__name__)
        return False


def _broken_by_station(payload):
    """{TfL station id: {lift id, ...}}, skipping malformed items."""
    broken = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        station_id = item.get("stationUniqueId")
        lift_ids = item.get("disruptedLiftUniqueIds")
        if not isinstance(station_id, str) or not isinstance(lift_ids, list):
            continue
        broken.setdefault(station_id.strip(), set()).update(
            lift_id for lift_id in lift_ids if isinstance(lift_id, str)
        )
    return broken


def fetch_lift_feed():
    """The lift feed's items, or None when it cannot be read or trusted.

    An empty list counts as untrustworthy: some lift in London is always out
    of service, so an empty feed is an outage. Believing it would mark
    every lift as working.
    """
    params = {
        key: value
        for key, value in {"app_id": settings.TFL_API_ID, "app_key": settings.TFL_API_KEY}.items()
        if value
    }
    try:
        response = requests.get(LIFTS_V2_URL, params=params, timeout=(5, 20))
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or not payload:
            raise ValueError("lift feed is empty or not a list")
    except (requests.RequestException, ValueError) as exc:
        # Type only: requests' messages embed the URL with app_key.
        logger.warning("TfL lift status fetch failed (%s); keeping previous data",
                       type(exc).__name__)
        return None
    return payload


def _update_broken_lifts():
    payload = fetch_lift_feed()
    if payload is None:
        return False

    broken = _broken_by_station(payload)

    # Only rows whose broken lifts changed are written, and freshness is one
    # small row: rewriting every station each run would ship hundreds of
    # pointless WAL pages to B2 every 5 minutes.
    changed = []
    station_pks = {}
    for pk, tfl_ids, old in StationRoutes.objects.values_list("pk", "tfl_ids", "broken_lift_ids"):
        for tfl_id in tfl_ids or []:
            station_pks[tfl_id] = pk
        new = sorted(set().union(*(broken.get(tfl_id, set()) for tfl_id in tfl_ids or [])))
        if new != (old or []):
            changed.append(StationRoutes(pk=pk, broken_lift_ids=new))
    now = timezone.now()
    previous_read = lift_status_checked_at()
    with transaction.atomic():
        if changed:
            StationRoutes.objects.bulk_update(changed, ["broken_lift_ids"])
        LiftStatusCheck.objects.update_or_create(pk=1, defaults={"checked_at": now})

    # After, and apart from, the station pages' data: the history is a
    # background record and must not cost the pages an update.
    try:
        _record_lift_outages(payload, broken, station_pks, previous_read, now)
    except Exception as exc:  # noqa: BLE001 - see comment above
        logger.warning("Lift outage recording failed (%s)", type(exc).__name__)
    return True


def _record_lift_outages(payload, broken, station_pks, previous_read, now):
    """Bring the per-lift outage history in line with one successful read."""
    listed = {
        (station_id, lift_id) for station_id, lift_ids in broken.items() for lift_id in lift_ids
    }
    open_outages = dict(
        ((tfl_station_id, lift_id), pk)
        for pk, tfl_station_id, lift_id in LiftOutage.objects.filter(
            cleared_at=None
        ).values_list("pk", "tfl_station_id", "lift_id")
    )
    started = listed - open_outages.keys()
    ended = [open_outages[key] for key in open_outages.keys() - listed]
    if not started and not ended:
        return

    # A station can have several items, each with its own lifts and message.
    messages = {}
    for item in payload:
        if not isinstance(item, dict) or not isinstance(item.get("message"), str):
            continue
        station_id = item.get("stationUniqueId")
        lift_ids = item.get("disruptedLiftUniqueIds")
        if isinstance(station_id, str) and isinstance(lift_ids, list):
            for lift_id in lift_ids:
                if isinstance(lift_id, str):
                    messages.setdefault((station_id.strip(), lift_id), item["message"])

    with transaction.atomic():
        LiftOutage.objects.bulk_create(
            LiftOutage(
                lift_id=lift_id,
                tfl_station_id=station_id,
                station_id=station_pks.get(station_id),
                message=messages.get((station_id, lift_id), ""),
                first_seen=now,
                last_seen=now,
            )
            for station_id, lift_id in sorted(started)
        )
        if ended:
            LiftOutage.objects.filter(pk__in=ended).update(
                last_seen=previous_read or now, cleared_at=now
            )


def lift_status_checked_at():
    """When the lift feed was last read successfully, or None."""
    return LiftStatusCheck.objects.filter(pk=1).values_list("checked_at", flat=True).first()
