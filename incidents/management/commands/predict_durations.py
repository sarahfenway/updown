from datetime import timedelta

from django.core.management.base import BaseCommand
from django.db.models import Q
from django.utils import timezone

from incidents.models import Incident
from incidents.ml import predict_duration


class Command(BaseCommand):
    help = "Add or update ML duration predictions for open and recently resolved incidents"

    def add_arguments(self, parser):
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Re-predict even if an incident already has an estimate",
        )
        parser.add_argument(
            "--backfill",
            type=int,
            metavar="DAYS",
            help="Also predict for incidents resolved in the last N days",
        )

    def handle(self, *args, **options):
        backfill_days = options["backfill"] or 1
        incidents = Incident.objects.filter(
            Q(resolved=False)
            | Q(resolved=True, end_time__gte=timezone.now() - timedelta(days=backfill_days))
        )
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
