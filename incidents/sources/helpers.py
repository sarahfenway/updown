from datetime import timedelta

from django.utils import timezone

from incidents.models import Report, SourceCheck

# The TfL feeds we poll. Anything else (user reports) is not a feed.
FEED_SOURCES = (Report.SOURCE_TFLAPI_V1, Report.SOURCE_TFLAPI_V2)

# update_incidents runs every 5 minutes, so this is three missed reads: long
# enough to ride out a blip, short enough that a fixed lift is not shown as
# broken for long when one feed is down.
SOURCE_STALE_AFTER = timedelta(minutes=15)


# The lift feed is a second opinion on lift faults, so it is the only feed we
# ever give up on: after this long unread, whatever it last listed is dropped
# (see consolidate_incidents). The StopPoint feed is the main source for
# everything, so its faults wait for it however long it is away.
GIVE_UP_AFTER = {Report.SOURCE_TFLAPI_V2: timedelta(hours=6)}


def mark_source_read(source, now=None, pending=None):
    """Record a successful read of ``source``; returns the previous read time."""
    now = now or timezone.now()
    previous = (
        SourceCheck.objects.filter(pk=source)
        .values_list("checked_at", flat=True)
        .first()
    )
    defaults = {"checked_at": now}
    if pending is not None:
        defaults["pending"] = pending
    SourceCheck.objects.update_or_create(pk=source, defaults=defaults)
    return previous


def pending_changes(source):
    """What ``source`` showed on its last read that is waiting for a second."""
    pending = (
        SourceCheck.objects.filter(pk=source).values_list("pending", flat=True).first()
    )
    return pending if isinstance(pending, dict) else {}


def feeds_given_up(now=None):
    """Feeds unread for so long (or never read) that we stop waiting for them."""
    now = now or timezone.now()
    last_read = dict(
        SourceCheck.objects.filter(pk__in=GIVE_UP_AFTER).values_list(
            "source", "checked_at"
        )
    )
    return {
        source
        for source, limit in GIVE_UP_AFTER.items()
        if source not in last_read or last_read[source] < now - limit
    }


def stale_feed_sources(now=None):
    """Feeds with no successful read in the last SOURCE_STALE_AFTER.

    Covers a feed that is down, one returning rubbish, and one that is
    switched off: in every case we can no longer see faults leave it.
    """
    cutoff = (now or timezone.now()) - SOURCE_STALE_AFTER
    fresh = set(
        SourceCheck.objects.filter(
            pk__in=FEED_SOURCES, checked_at__gte=cutoff
        ).values_list("source", flat=True)
    )
    return set(FEED_SOURCES) - fresh


def usable_feed_items(payload):
    """The feed's items, or None when the response is not a usable read.

    An empty list counts as unusable: TfL always has some disruption
    somewhere, so an empty feed is an outage. Believing it would close
    every open report from that feed at once.
    """
    if not isinstance(payload, list) or not payload:
        return None
    return payload


def existing_report_lookup(source, *, resolved=False):
    reports_by_key = {}
    cleared_disruption = {}

    reports = Report.objects.filter(source=source)
    if resolved is not None:
        reports = reports.filter(resolved=resolved)
    reports = reports.order_by().only(
        "id", "station_id", "text", "source", "information", "resolved", "end_time"
    )

    for report in reports:
        key = (report.station_id, report.text, report.source, report.information)
        reports_by_key.setdefault(key, []).append(report)
        if resolved is False:
            cleared_disruption[report.pk] = report

    if resolved is False:
        return reports_by_key, cleared_disruption

    cleared_reports = (
        Report.objects.filter(resolved=False, source=source)
        .order_by()
        .only(
            "id",
            "station_id",
            "text",
            "source",
            "information",
            "resolved",
            "end_time",
        )
    )
    for report in cleared_reports:
        cleared_disruption[report.pk] = report

    return reports_by_key, cleared_disruption
