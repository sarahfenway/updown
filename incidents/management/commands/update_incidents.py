from difflib import SequenceMatcher

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from incidents.models import Report, Incident
from incidents.sources.tflapiv1 import check as check_tflv1
# from incidents.sources.tflapiv2 import check as check_tflv2
from incidents.utils import send_tweet, update_last_updated, send_bluesky
from incidents.ml import predict_duration


def refresh_prediction(incident):
    duration, confidence = predict_duration(incident)
    incident.estimated_duration = duration
    incident.prediction_confidence = confidence
    incident.save(update_fields=["estimated_duration", "prediction_confidence"])


def consolidate_incidents():
    # Take all the reports and consolidate them into incidents
    for report in Report.objects.filter(resolved=False).select_related(
        "station", "station__parent_station"
    ):
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

        if incident is None:
            # Create a new incident
            incident = Incident(
                information=report.information,
                station=report.station.parent_station,
                text=report.text,
                start_time=report.start_time,
                end_time=report.end_time,
                resolved=report.resolved,
            )
            incident.save()

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

            send_tweet(tweet)
            send_bluesky(tweet)
        else:
            # Update the existing incident
            incident.information = report.information
            incident.text = report.text
            incident.start_time = (
                report.start_time
                if report.start_time < incident.start_time
                else incident.start_time
            )
            incident.save()

        # Add the report to the incident
        incident.reports.add(report)
        refresh_prediction(incident)

    Report.objects.filter(
        source=Report.SOURCE_USER, resolved=False, end_time__lt=timezone.now()
    ).update(resolved=True)

    for incident in Incident.objects.filter(resolved=False).select_related(
        "station"
    ).prefetch_related("reports"):
        # Check if the incident has been resolved
        unresolved_reports = [r for r in incident.reports.all() if not r.resolved]
        if not unresolved_reports:
            incident.resolved = True
            incident.end_time = timezone.now()
            incident.save()

            all_reports = list(incident.reports.all())
            if (
                len(all_reports) > 1
                or all_reports[0].source != Report.SOURCE_USER
            ):
                tweet = f"Step free access has been restored at {incident.station.name}"
                send_tweet(tweet)
                send_bluesky(tweet)


class Command(BaseCommand):
    help = "Updates the incidents list"

    def handle(self, *args, **options):
        try:
            check_tflv1()
#             check_tflv2()
            consolidate_incidents()
            update_last_updated()
        except Exception as e:
            raise CommandError(f"Error updating incidents list: {e}")

        self.stdout.write(self.style.SUCCESS("Successfully updated incident list"))
