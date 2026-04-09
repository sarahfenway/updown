from django.core.management.base import BaseCommand

from incidents.models import Incident
from incidents.ml import predict_duration


class Command(BaseCommand):
    help = "Add or update ML duration predictions for open incidents"

    def add_arguments(self, parser):
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Re-predict even if an incident already has an estimate",
        )

    def handle(self, *args, **options):
        incidents = Incident.objects.filter(resolved=False)
        if not options["overwrite"]:
            incidents = incidents.filter(estimated_duration__isnull=True)

        updated = 0
        for incident in incidents:
            try:
                duration = predict_duration(incident)
            except Exception as e:
                self.stderr.write(f"  {incident.station.name}: {e}")
                continue

            if duration is not None:
                incident.estimated_duration = duration
                incident.save(update_fields=["estimated_duration"])
                updated += 1

        self.stdout.write(self.style.SUCCESS(f"Updated {updated} incident(s)"))
