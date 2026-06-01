import time
from difflib import SequenceMatcher

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.db.models import Exists, OuterRef
from django.utils import timezone

from incidents.models import Report, Incident
from incidents.sources.tflapiv1 import check as check_tflv1
# from incidents.sources.tflapiv2 import check as check_tflv2
from incidents.utils import send_tweet, update_last_updated, send_bluesky_messages
from incidents.ml import predict_duration


def refresh_prediction(incident):
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
        # Take all the reports and consolidate them into incidents
        unresolved_reports = sorted(
            Report.objects.filter(resolved=False)
            .order_by()
            .select_related("station", "station__parent_station"),
            key=lambda report: (report.start_time, report.pk),
            reverse=True,
        )
        for report in unresolved_reports:
            # Check if there is an incident for this station
            incidents = Incident.objects.filter(
                station=report.station.parent_station, resolved=False
            )

            incident = None
            for item in incidents:
                # if the same report station and similar text, then same incident
                if SequenceMatcher(None, item.text, report.text).ratio() > 0.9:
                    incident = item
                    break

            # Only (re)compute the ML prediction when it can actually
            # differ: a brand-new incident, or one whose text changed.
            # The prediction is a function of the incident's text /
            # station / timing features, so re-running it every 5 minutes
            # for an unchanged incident was pure wasted work (a model
            # load + several queries + a write, per report, per run).
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
                # Update the existing incident
                text_changed = incident.text != report.text
                incident.information = report.information
                incident.text = report.text
                incident.start_time = (
                    report.start_time
                    if report.start_time < incident.start_time
                    else incident.start_time
                )
                incident.save()
                if text_changed:
                    should_predict = True

            # Add the report to the incident
            _attach_report(incident, report)
            if should_predict:
                prediction_incident_ids.add(incident.pk)

        Report.objects.filter(
            source=Report.SOURCE_USER, resolved=False, end_time__lt=timezone.now()
        ).update(resolved=True)

        report_links = Incident.reports.through.objects.filter(
            incident_id=OuterRef("pk")
        )
        unresolved_report_links = report_links.filter(report__resolved=False)

        unresolved_incidents = sorted(
            Incident.objects.filter(resolved=False)
            .order_by()
            .select_related("station")
            .annotate(has_unresolved_reports=Exists(unresolved_report_links)),
            key=lambda incident: (incident.start_time, incident.pk),
            reverse=True,
        )
        for incident in unresolved_incidents:
            # Check if the incident has been resolved
            if not incident.has_unresolved_reports:
                incident.resolved = True
                incident.end_time = timezone.now()
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
            # check_tflv2()
            posts = _phase("consolidate", consolidate_incidents)
            _phase("social_posts", lambda: _send_pending_posts(posts))
            update_last_updated()
        except Exception as e:
            raise CommandError(f"Error updating incidents list: {e}")

        self.stdout.write(self.style.SUCCESS("Successfully updated incident list"))
