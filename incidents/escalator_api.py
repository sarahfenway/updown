"""/api/escalators/: escalator faults, and nothing else.

Read from the background record of TfL's StopPoint disruption feed
(incidents/sources/escalators.py). Escalator faults are not issues on the
site: they have no reports, no predictions and no social posts, so this is
the only place they are published.
"""

from datetime import timedelta

from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.cache import never_cache
from django.views.decorators.http import require_safe

from incidents.models import EscalatorFault, Report, SourceCheck
from incidents.prediction_shared import is_planned_work
from incidents.sources.helpers import SOURCE_STALE_AFTER
from stations.utils import cleanup_station_name

# Same window as "resolved" in /api/incidents/.
RESOLVED_FOR = timedelta(hours=12)


def _station(fault):
    """(grouping key, name, naptan) for the station a record belongs to.

    The feed lists one fault once per stop, and a station can have several
    stops (one per line), so records are rolled up to the parent station,
    the same way issues are.
    """
    station = fault.station
    if station is None:
        name = cleanup_station_name(fault.station_name) or fault.atco_code
        return ("stop", name), name, fault.atco_code
    parent = station.parent_station or station
    naptan = (
        parent.naptan_id
        or parent.hub_naptan_id
        or station.naptan_id
        or station.hub_naptan_id
        or fault.atco_code
    )
    return ("station", parent.pk), parent.name, naptan


def _grouped(faults):
    """One row per fault per station, keyed by (station, text)."""
    groups = {}
    for fault in sorted(faults, key=lambda fault: (fault.first_seen, fault.pk)):
        key, name, naptan = _station(fault)
        row = groups.get((key, fault.text))
        if row is None:
            # Sorted oldest first, so this record gives the id and start.
            groups[(key, fault.text)] = {
                "id": fault.pk,
                "text": fault.text,
                "start_time": fault.first_seen,
                "end_time": fault.cleared_at,
                "resolved": fault.cleared_at is not None,
                "planned": fault.planned or is_planned_work(fault.text),
                "station_name": name,
                "station_naptan": naptan,
            }
            continue
        row["planned"] = row["planned"] or fault.planned
        if fault.cleared_at is not None and row["end_time"] is not None:
            row["end_time"] = max(row["end_time"], fault.cleared_at)
    return groups


def _newest_first(rows):
    return sorted(
        rows, key=lambda row: (row["start_time"], row["id"]), reverse=True
    )


@never_cache
@require_safe
def api_escalators(request):
    now = timezone.now()
    faults = EscalatorFault.objects.select_related(
        "station", "station__parent_station"
    ).order_by()
    listed = _grouped(faults.filter(cleared_at=None))
    current = _newest_first(listed.values())
    # A fault dropped at one stop but still listed at another stop of the
    # same station has not ended.
    resolved = _newest_first(
        row
        for key, row in _grouped(
            faults.filter(cleared_at__gte=now - RESOLVED_FOR)
        ).items()
        if key not in listed
    )

    # Escalator faults come from the StopPoint feed alone. If that has not
    # been read lately, faults may have ended without our seeing it.
    last_checked = (
        SourceCheck.objects.filter(pk=Report.SOURCE_TFLAPI_V1)
        .values_list("checked_at", flat=True)
        .first()
    )
    return JsonResponse(
        {
            "issues": [row for row in current if not row["planned"]],
            "planned": [row for row in current if row["planned"]],
            "resolved": resolved,
            "last_checked": last_checked,
            "up_to_date": (
                last_checked is not None
                and last_checked >= now - SOURCE_STALE_AFTER
            ),
            "last_updated": now,
        }
    )
