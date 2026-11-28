"""TfL's lift feed (Disruptions/Lifts/v2) as a second source of reports.

On unless settings.TFL_LIFT_FEED_AS_SOURCE is false. It mostly repeats the
StopPoint feed (tflapiv1) in the same words, so its reports fold into the
same issues; its value is that the site keeps opening and closing issues
while the StopPoint feed is down.

It is the less trusted of the two, so it has to say a thing twice: a fault
must be listed on two reads in a row before it becomes a report, and be
missing from two reads in a row before its report is closed. One odd read
(a truncated list, a fault that flickers) therefore changes nothing.

It also drops faults early: on the morning of 28 November 2026 it listed
none of the twelve live lift faults the StopPoint feed still had. That is
harmless here, because an issue stays open while any readable feed lists it
(consolidate_incidents), but never let this feed close an issue on its own
word while the StopPoint feed disagrees.
"""

from datetime import datetime

import logging

from django.db import transaction
from django.utils import timezone

from incidents.models import Report
from incidents.prediction_shared import is_planned_work
from incidents.sources.helpers import (
    existing_report_lookup,
    mark_source_read,
    pending_changes,
)
from incidents.utils import remove_tfl_specifics, strip_step_free_prefix
from stations.step_free.broken_lifts import fetch_lift_feed
from stations.utils import find_station_from_naptan

logger = logging.getLogger(__name__)


def check():
    """Never raises: an optional second feed must not stop the incident update."""
    try:
        _check()
    except Exception as exc:  # noqa: BLE001 - see docstring
        logger.warning(
            "TfL lift feed source failed (%s); skipping this run", type(exc).__name__
        )


def _report_text(item):
    message = item["message"]
    text = message[message.find(":") + 1 :].strip()
    return remove_tfl_specifics(strip_step_free_prefix(text))


def _first_seen(value, now):
    try:
        seen = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return now
    return seen if timezone.is_aware(seen) and seen <= now else now


def _listed_faults(items):
    """(station pk, text) for each usable item, in feed order, no repeats."""
    listed = {}
    for item in items:
        try:
            station = find_station_from_naptan(item["stationUniqueId"])
            text = _report_text(item)
        except (KeyError, TypeError, AttributeError):
            continue
        # This feed lists dated and planned work as plain faults. The
        # StopPoint feed handles those properly, with their dates.
        if not station or not text or is_planned_work(text):
            continue
        listed[(station.pk, text)] = True
    return list(listed)


def _check():
    items = fetch_lift_feed()
    if items is None:
        # Down, empty or malformed. Nothing is closed and the read is not
        # recorded, so after SOURCE_STALE_AFTER this feed's open reports stop
        # holding issues open. What was waiting for a second read keeps
        # waiting: a failed read is not a read.
        return

    now = timezone.now()
    source = Report.SOURCE_TFLAPI_V2
    reports_by_key, not_listed = existing_report_lookup(source)
    waiting = pending_changes(source)
    seen_once = {
        (entry[0], entry[1]): entry[2]
        for entry in waiting.get("appeared", [])
        if isinstance(entry, list) and len(entry) == 3
    }
    missed_once = set(waiting.get("missing", []))

    appeared = []
    missing = []
    with transaction.atomic():
        for station_pk, text in _listed_faults(items):
            existing = reports_by_key.get((station_pk, text, source, False))
            if existing:
                for report in existing:
                    not_listed.pop(report.pk, None)
            elif (station_pk, text) in seen_once:
                Report.objects.create(
                    station_id=station_pk,
                    text=text,
                    source=source,
                    information=False,
                    start_time=_first_seen(seen_once[(station_pk, text)], now),
                )
            else:
                appeared.append([station_pk, text, now.isoformat()])

        for report in not_listed.values():
            if report.pk in missed_once:
                report.resolved = True
                report.end_time = now
                report.save()
            else:
                missing.append(report.pk)

        mark_source_read(
            source, now, pending={"appeared": appeared, "missing": missing}
        )
