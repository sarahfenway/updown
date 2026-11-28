"""Record escalator messages from TfL's StopPoint disruption feed.

The feed the site already reads for step-free faults also carries escalator
faults and planned escalator work. This keeps a history of them and nothing
more: no reports, no issues, nothing shown.
"""

import logging

from django.db import transaction
from django.utils import timezone
from django.utils.dateparse import parse_datetime

from incidents.models import EscalatorFault
from stations.utils import find_station_from_naptan

logger = logging.getLogger(__name__)


def record_escalator_faults(items, previous_read, now=None):
    """Bring the escalator records in line with one successful feed read.

    ``previous_read`` is when the feed was last read successfully before
    this (None the first time). Never raises: this is a background record
    and must not stop the issue update.
    """
    try:
        _record(items, previous_read, now or timezone.now())
    except Exception as exc:  # noqa: BLE001 - see docstring
        logger.warning("Escalator recording failed (%s)", type(exc).__name__)


def _start_time(item):
    value = item.get("fromDate")
    parsed = parse_datetime(value) if isinstance(value, str) else None
    if parsed is not None and timezone.is_naive(parsed):
        parsed = timezone.make_aware(parsed)
    return parsed


def _record(items, previous_read, now):
    listed = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        text = item.get("description")
        atco_code = item.get("atcoCode")
        if not isinstance(text, str) or not isinstance(atco_code, str):
            continue
        if "escalator" not in text.lower():
            continue
        listed.setdefault((atco_code, text.strip()), item)

    open_faults = {
        (fault.atco_code, fault.text): fault
        for fault in EscalatorFault.objects.filter(cleared_at=None).only(
            "id", "atco_code", "text"
        )
    }

    with transaction.atomic():
        for key in listed.keys() - open_faults.keys():
            atco_code, text = key
            item = listed[key]
            name = item.get("commonName")
            EscalatorFault.objects.create(
                station=find_station_from_naptan(atco_code),
                atco_code=atco_code,
                station_name=name[:200] if isinstance(name, str) else "",
                text=text,
                planned=item.get("appearance") == "PlannedWork",
                source_start=_start_time(item),
                first_seen=now,
                last_seen=now,
            )
        gone = [open_faults[key].pk for key in open_faults.keys() - listed.keys()]
        if gone:
            EscalatorFault.objects.filter(pk__in=gone).update(
                last_seen=previous_read or now, cleared_at=now
            )
