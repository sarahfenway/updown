from datetime import timedelta

from django.core.management.base import BaseCommand
from django.db.models import Q
from django.utils import timezone

from incidents.models import Incident
from incidents.ml import _load_model, predict_duration


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
        if _load_model() is None:
            self.stderr.write(self.style.ERROR(
                "No ml_model.joblib found — re-upload the trained model. "
                "Heroku's filesystem is ephemeral, so every deploy wipes it."
            ))
            return

        backfill_days = options["backfill"] or 1
        incidents = Incident.objects.filter(
            Q(resolved=False)
            | Q(resolved=True, end_time__gte=timezone.now() - timedelta(days=backfill_days))
        )
        if not options["overwrite"]:
            incidents = incidents.filter(estimated_duration__isnull=True)

        total = incidents.count()
        self.stdout.write(f"Predicting for {total} incident(s)...")
        self.stdout.flush()

        skipped_planned = 0
        updated = 0
        for i, incident in enumerate(incidents, 1):
            try:
                duration, confidence = predict_duration(incident)
            except Exception as e:
                self.stderr.write(f"  {incident.station.name}: {e}")
                self.stderr.flush()
                continue

            if duration is not None:
                incident.estimated_duration = duration
                incident.prediction_confidence = confidence
                incident.save(update_fields=["estimated_duration", "prediction_confidence"])
                updated += 1
            else:
                skipped_planned += 1

            # Flush progress every 20 rows so we can see where we are if
            # the process gets killed mid-run (OOM, dyno shutdown, etc.).
            if i % 20 == 0:
                self.stdout.write(
                    f"  [{i}/{total}] updated={updated} skipped={skipped_planned}"
                )
                self.stdout.flush()

        self.stdout.write(self.style.SUCCESS(
            f"Updated {updated} / {total} incident(s) "
            f"({skipped_planned} skipped as planned work)"
        ))
