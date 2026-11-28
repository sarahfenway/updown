import logging
import time
from difflib import SequenceMatcher

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from incidents.models import Report, Incident
from incidents.sources.helpers import (
    FEED_SOURCES,
    feeds_given_up,
    stale_feed_sources,
)
from incidents.sources.tflapiv1 import check as check_tflv1
from stations.step_free.broken_lifts import update_broken_lifts
from incidents.sources.tflapiv2 import check as check_tflv2
from incidents.utils import send_tweet, update_last_updated, send_bluesky_messages
from incidents.ml import predict_duration

logger = logging.getLogger(__name__)

# The most brand-new issues the lift feed may open in one run. It normally
# repeats faults the main feed already lists, so more than this at once
# means its wording has changed, not that this many lifts broke together.
# Those reports are held back (no issue, no post) until they match an issue
# or the feed drops them.
LIFT_FEED_MAX_NEW_ISSUES = 3


def refresh_prediction(incident):
    """Store a fault's first prediction.

    Called once per fault. The first date shown is the one we are scored on,
    so it is never recomputed afterwards.
    """
    duration, confidence = predict_duration(incident)
    incident.estimated_duration = duration
    incident.prediction_confidence = confidence
    incident.save(update_fields=["estimated_duration", "prediction_confidence"])


def _attach_report(incident, report):
    through = Incident.reports.through
    if through.objects.filter(incident_id=incident.pk, report_id=report.pk).exists():
        if incident.report_count is None:
            incident.report_count = incident.reports.count()
            incident.save(update_fields=["report_count"])
        return

    count_was_unknown = incident.report_count is None
    incident.reports.add(report)
    incident.report_count = (
        incident.reports.count()
        if count_was_unknown
        else incident.report_count + 1
    )
    incident.save(update_fields=["report_count"])


def _matching_incident(report):
    """The open issue at the report's station with near-identical text."""
    incidents = Incident.objects.filter(
        station=report.station.parent_station, resolved=False
    )
    for item in incidents:
        # if the same report station and similar text, then same incident.
        # autojunk must be off: for texts >=200 chars the default
        # heuristic makes the ratio asymmetric and near-zero for
        # near-identical strings (0.08 one way, 0.95 the other),
        # which made this loop create a duplicate incident on
        # every run instead of matching the existing one.
        if (
            SequenceMatcher(None, item.text, report.text, autojunk=False).ratio()
            > 0.9
        ):
            return item
    return None


def _drop_reports_from_feeds_given_up(now):
    """Close what a long-gone feed last listed; returns the issues affected.

    Those issues then close in the usual way if nothing else lists them,
    but without a "restored" post: we do not know the fault is fixed.
    """
    links = Incident.reports.through.objects
    affected = set()
    for source in feeds_given_up(now):
        report_ids = list(
            Report.objects.filter(source=source, resolved=False)
            .order_by()
            .values_list("pk", flat=True)
        )
        if not report_ids:
            continue
        affected.update(
            links.filter(
                report_id__in=report_ids, incident__resolved=False
            ).values_list("incident_id", flat=True)
        )
        Report.objects.filter(pk__in=report_ids).update(resolved=True, end_time=now)
        logger.warning(
            "Gave up on feed %s: dropped %d open report(s)", source, len(report_ids)
        )
    return affected


def _lift_feed_reports_to_hold(unresolved_reports):
    """Lift-feed reports that would open too many new issues at once."""
    lift_feed = [
        report
        for report in unresolved_reports
        if report.source == Report.SOURCE_TFLAPI_V2
    ]
    if len(lift_feed) <= LIFT_FEED_MAX_NEW_ISSUES:
        return set()
    attached = set(
        Incident.reports.through.objects.filter(
            report_id__in=[report.pk for report in lift_feed]
        ).values_list("report_id", flat=True)
    )
    unattached = [report for report in lift_feed if report.pk not in attached]
    if len(unattached) <= LIFT_FEED_MAX_NEW_ISSUES:
        return set()
    new = {
        report.pk for report in unattached if _matching_incident(report) is None
    }
    if len(new) <= LIFT_FEED_MAX_NEW_ISSUES:
        return set()
    logger.warning(
        "Lift feed would open %d new issues at once; holding them back", len(new)
    )
    return new


def consolidate_incidents():
    """Fold unresolved reports into incidents and detect resolutions.

    Returns a list of tweet/bluesky message strings to be sent *after*
    the database work commits. SQLite has one writer, so the transaction
    below is kept to the cheap consolidation writes only. Predictions and
    social posts can involve model loading, history reads, and HTTP calls;
    they happen after the write lock has been released.
    """
    pending_posts = []
    prediction_incident_ids = set()

    with transaction.atomic():
        now = timezone.now()
        given_up_on = _drop_reports_from_feeds_given_up(now)

        # Take all the reports and consolidate them into incidents
        unresolved_reports = sorted(
            Report.objects.filter(resolved=False)
            .order_by()
            .select_related("station", "station__parent_station"),
            key=lambda report: (report.start_time, report.pk),
            reverse=True,
        )
        held_back = _lift_feed_reports_to_hold(unresolved_reports)
        for report in unresolved_reports:
            if report.pk in held_back:
                continue
            incident = _matching_incident(report)

            # A fault is predicted once, when it first appears. The first
            # date shown is the one we are scored on, so a later text edit
            # never replaces it; it only fills in a fault that has none.
            should_predict = False

            if incident is None:
                # Create a new incident
                incident = Incident(
                    information=report.information,
                    station=report.station.parent_station,
                    text=report.text,
                    start_time=report.start_time,
                    end_time=report.end_time,
                    resolved=report.resolved,
                    report_count=0,
                )
                incident.save()
                should_predict = True

                tweet = f"{incident.station.name}: {incident.text}"

                if report.source == Report.SOURCE_USER:
                    tweet += "\n\nThis is a user report"

                if len(tweet) > 280:
                    if incident.information:
                        tweet = f"New information on step free access at {incident.station.name}"
                    else:
                        tweet = (
                            f"Step free access issues reported at {incident.station.name}"
                        )

                pending_posts.append(tweet)
            else:
                # Update the existing incident, but only write the row when
                # something actually changed. In steady state the TfL text is
                # byte-identical run after run, so an unconditional save() was
                # a full-row UPDATE per unresolved report per 5-minute tick —
                # pointless WAL frames shipped to B2 and time held inside the
                # single-writer transaction that admin writes wait behind.
                text_changed = incident.text != report.text
                info_changed = incident.information != report.information
                start_earlier = report.start_time < incident.start_time
                if text_changed or info_changed or start_earlier:
                    incident.information = report.information
                    incident.text = report.text
                    if start_earlier:
                        incident.start_time = report.start_time
                    incident.save(
                        update_fields=["information", "text", "start_time"]
                    )
                if text_changed and incident.estimated_duration is None:
                    should_predict = True

            # Add the report to the incident
            _attach_report(incident, report)
            if should_predict:
                prediction_incident_ids.add(incident.pk)

        Report.objects.filter(
            source=Report.SOURCE_USER, resolved=False, end_time__lt=timezone.now()
        ).update(resolved=True)

        # An issue stays open while a feed we can read still lists it. A
        # feed we cannot read (down, or switched off) never shows a fault
        # leaving, so its open reports are set aside: see below.
        stale_feeds = stale_feed_sources(now)
        readable_feeds = set(FEED_SOURCES) - stale_feeds
        links = Incident.reports.through.objects
        listed = set()
        unreadable_reports = {}
        for incident_id, report_id, source in links.filter(
            incident__resolved=False, report__resolved=False
        ).values_list("incident_id", "report_id", "report__source"):
            if source in stale_feeds:
                unreadable_reports.setdefault(incident_id, []).append(report_id)
            else:
                listed.add(incident_id)

        unresolved_incidents = sorted(
            Incident.objects.filter(resolved=False)
            .order_by()
            .select_related("station"),
            key=lambda incident: (incident.start_time, incident.pk),
            reverse=True,
        )
        for incident in unresolved_incidents:
            # Check if the incident has been resolved
            if incident.pk in listed:
                continue
            held_by = unreadable_reports.get(incident.pk)
            if held_by:
                # Only an unreadable feed still lists this fault. Close it
                # only on evidence: a feed we can read reported it too and
                # has since dropped it. Without that we cannot know, so it
                # stays open until the feed is back.
                if not (
                    readable_feeds
                    and links.filter(
                        incident_id=incident.pk,
                        report__source__in=readable_feeds,
                    ).exists()
                ):
                    continue
                # Close the unreadable feed's reports as well, or the next
                # run would open a new issue from them.
                Report.objects.filter(pk__in=held_by).update(
                    resolved=True, end_time=now
                )
            incident.resolved = True
            incident.end_time = now
            incident.save()

            reports_count = (
                incident.report_count
                if incident.report_count is not None
                else incident.reports.count()
            )
            first_report_source = (
                incident.reports.order_by()
                .values_list("source", flat=True)
                .first()
            )
            if incident.pk in given_up_on:
                # Closed because its feed went away, not because we saw
                # the fault end: nothing to announce.
                continue
            if (
                reports_count > 1
                or first_report_source != Report.SOURCE_USER
            ):
                pending_posts.append(
                    f"Step free access has been restored at {incident.station.name}"
                )

    for incident in (
        Incident.objects.filter(pk__in=prediction_incident_ids)
        .select_related("station", "station__parent_station")
        .order_by("id")
    ):
        refresh_prediction(incident)

    return pending_posts


def _send_pending_posts(posts):
    """Fire the social posts once the DB transaction has committed.

    Kept outside ``consolidate_incidents``' transaction so a slow or
    hung Twitter/Bluesky call can never hold the SQLite write lock.
    """
    for message in posts:
        send_tweet(message)
    send_bluesky_messages(posts)


class Command(BaseCommand):
    help = "Updates the incidents list"

    def add_arguments(self, parser):
        parser.add_argument(
            "--timing",
            action="store_true",
            help="Print how long each phase takes (TfL fetch, consolidate, posts).",
        )

    def handle(self, *args, **options):
        timing = options.get("timing")

        def _phase(label, fn):
            if timing:
                self.stdout.write(f"  [{label}] starting")
            start = time.monotonic()
            result = fn()
            if timing:
                elapsed = time.monotonic() - start
                self.stdout.write(f"  [{label}] {elapsed:.2f}s")
            return result

        try:
            _phase("tfl_fetch", check_tflv1)
            if settings.TFL_LIFT_FEED_AS_SOURCE:
                _phase("tfl_lift_fetch", check_tflv2)
            posts = _phase("consolidate", consolidate_incidents)
            _phase("social_posts", lambda: _send_pending_posts(posts))
            update_last_updated()
            # Last, and it never raises: the step-free pages' lift status
            # must not delay or stop the incident update.
            _phase("lift_status", update_broken_lifts)
        except Exception as e:
            raise CommandError(f"Error updating incidents list: {e}")

        self.stdout.write(self.style.SUCCESS("Successfully updated incident list"))
