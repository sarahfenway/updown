from datetime import timedelta

from django.core.management.base import BaseCommand
from django.db.models import Count, Q
from django.utils import timezone

from incidents.models import Incident
from incidents.ml import _load_model, predict_duration


PREDICTION_INCIDENT_FIELDS = (
    "id",
    "station_id",
    "information",
    "text",
    "start_time",
    "end_time",
    "resolved",
    "estimated_duration",
    "prediction_confidence",
    "station__id",
    "station__name",
    "station__tube",
    "station__dlr",
    "station__national_rail",
    "station__crossrail",
    "station__overground",
    "station__access_via_lift",
    "station__parent_station__id",
    "station__parent_station__name",
    "station__parent_station__tube",
    "station__parent_station__dlr",
    "station__parent_station__national_rail",
    "station__parent_station__crossrail",
    "station__parent_station__overground",
    "station__parent_station__access_via_lift",
)


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
        parser.add_argument(
            "--after-id",
            type=int,
            metavar="ID",
            help="Skip incidents up to and including this id",
        )
        parser.add_argument(
            "--limit",
            type=int,
            metavar="N",
            help="Process at most N incidents in this run",
        )
        parser.add_argument(
            "--chunk-size",
            type=int,
            default=100,
            metavar="N",
            help="How many incidents to stream from the DB at once",
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
        if options["after_id"]:
            incidents = incidents.filter(id__gt=options["after_id"])

        incidents = (
            incidents.select_related("station", "station__parent_station")
            .annotate(num_reports=Count("reports", distinct=True))
            .only(*PREDICTION_INCIDENT_FIELDS)
            .order_by("id")
        )
        if options["limit"]:
            total = min(options["limit"], incidents.count())
            incidents = incidents[: options["limit"]]
        else:
            total = incidents.count()

        self.stdout.write(f"Predicting for {total} incident(s)...")
        self.stdout.flush()

        skipped_planned = 0
        updated = 0
        last_processed_id = None
        for i, incident in enumerate(
            incidents.iterator(chunk_size=max(1, options["chunk_size"])),
            1,
        ):
            last_processed_id = incident.id
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
                    f"  [{i}/{total}] last_id={last_processed_id} "
                    f"updated={updated} skipped={skipped_planned}"
                )
                self.stdout.flush()

        if last_processed_id is not None:
            self.stdout.write(f"Last processed incident id: {last_processed_id}")

        self.stdout.write(self.style.SUCCESS(
            f"Updated {updated} / {total} incident(s) "
            f"({skipped_planned} skipped as planned work)"
        ))
