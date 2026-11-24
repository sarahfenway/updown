"""Record which lifts TfL says are out of service, for the station pages.

Uses TfL's v2 lift feed because it names lifts by the same ids as the
station data file. This is separate from the incident sources, which are
not changed.
"""

import logging

import requests
from django.conf import settings
from django.utils import timezone

from stations.models import Station

logger = logging.getLogger(__name__)

LIFTS_V2_URL = "https://api.tfl.gov.uk/Disruptions/Lifts/v2/"


def update_broken_lifts():
    params = {
        key: value
        for key, value in {"app_id": settings.TFL_API_ID, "app_key": settings.TFL_API_KEY}.items()
        if value
    }
    try:
        response = requests.get(LIFTS_V2_URL, params=params, timeout=(5, 20))
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError("lift feed is not a list")
    except (requests.RequestException, ValueError) as exc:
        # Type only: requests' messages embed the URL with app_key.
        logger.warning("TfL lift status fetch failed (%s); keeping previous data",
                       type(exc).__name__)
        return False

    broken = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        station_id = (item.get("stationUniqueId") or "").strip()
        broken.setdefault(station_id, set()).update(item.get("disruptedLiftUniqueIds") or [])

    # A station's route data can cover several TfL stations (see
    # tfl_data.merge_routes). Read just that list, not the whole route JSON.
    now = timezone.now()
    stations = []
    rows = Station.objects.filter(tfl_station_id__isnull=False).values_list(
        "pk", "tfl_station_id", "step_free_routes__tfl_ids"
    )
    for pk, tfl_station_id, tfl_ids in rows:
        lift_ids = set()
        for tfl_id in tfl_ids or [tfl_station_id]:
            lift_ids |= broken.get(tfl_id, set())
        stations.append(Station(pk=pk, broken_lift_ids=sorted(lift_ids),
                                broken_lifts_checked_at=now))
    Station.objects.bulk_update(stations, ["broken_lift_ids", "broken_lifts_checked_at"])
    return True
