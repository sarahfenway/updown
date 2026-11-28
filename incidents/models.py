from django.db import models
from django.utils import timezone

from incidents.utils import split_additional_info


class MLModel(models.Model):
    """Trained ML model bytes, stored in the database.

    Heroku's filesystem is ephemeral — every deploy or dyno restart wipes
    anything written to disk, including an uploaded ``ml_model.joblib``.
    Stashing the joblib bytes in a single DB row means the model survives
    restarts and is visible to every dyno at the same time.
    """

    data = models.BinaryField()
    size_bytes = models.PositiveIntegerField()
    uploaded_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["-id"]

    def __str__(self):
        return f"MLModel #{self.pk} ({self.size_bytes} bytes @ {self.uploaded_at:%Y-%m-%d %H:%M})"


class Incident(models.Model):
    information = models.BooleanField(help_text="Is this information only?")
    station = models.ForeignKey("stations.Station", on_delete=models.DO_NOTHING)
    text = models.TextField()
    start_time = models.DateTimeField(default=timezone.now)
    end_time = models.DateTimeField(null=True, blank=True)
    resolved = models.BooleanField(default=False)
    reports = models.ManyToManyField("incidents.Report")
    report_count = models.PositiveIntegerField(null=True, blank=True)
    estimated_duration = models.DurationField(
        null=True,
        blank=True,
        help_text="ML-predicted duration for this incident",
    )
    prediction_confidence = models.FloatField(
        null=True,
        blank=True,
        help_text="ML prediction confidence (0-1)",
    )
    # Conditional re-prediction for an incident that is still open past its
    # original ``estimated_duration`` bound: P(resolved by X | still open
    # now). Display-only — the accuracy metric always scores the original
    # ``estimated_duration`` so "fixed by predicted time" stays honest.
    revised_estimated_duration = models.DurationField(
        null=True,
        blank=True,
        help_text="Re-predicted duration for an overdue open incident",
    )
    revised_prediction_confidence = models.FloatField(
        null=True,
        blank=True,
        help_text="Confidence (0-1) of the revised prediction",
    )

    @property
    def main_text(self):
        """The text without TfL's additional information."""
        return split_additional_info(self.text)[0]

    @property
    def additional_info(self):
        """TfL's additional information (travel advice), as plain text."""
        return split_additional_info(self.text)[1]

    def __str__(self):
        if hasattr(self, "reports_count"):
            reports_count = self.reports_count
        elif hasattr(self, "prefetched_reports"):
            reports_count = len(self.prefetched_reports)
        elif self.report_count is not None:
            reports_count = self.report_count
        else:
            reports_count = self.reports.count()

        return (
            f"{self.station.name} - Resolved: {self.resolved} - "
            f"{reports_count} reports"
        )

    class Meta:
        indexes = [
            models.Index(
                fields=["resolved", "information", "-start_time", "-id"],
                name="incident_status_idx",
            ),
            models.Index(fields=["end_time"], name="incident_end_time_idx"),
            models.Index(
                fields=["station_id", "resolved"],
                name="incident_station_resolved_idx",
            ),
            models.Index(
                fields=["id"],
                name="incident_unresolved_idx",
                condition=models.Q(resolved=False),
            ),
            models.Index(
                fields=["station_id", "-start_time"],
                name="incident_station_start_idx",
            ),
            models.Index(
                fields=["start_time", "id"],
                name="incident_training_idx",
                condition=models.Q(resolved=True, end_time__isnull=False),
            ),
            models.Index(
                fields=["end_time"],
                name="incident_prediction_end_idx",
                condition=models.Q(
                    resolved=True,
                    estimated_duration__isnull=False,
                ),
            ),
        ]
        ordering = ["-start_time", "-id"]


class Report(models.Model):
    SOURCE_TWITTER = "T"
    SOURCE_MASTODON = "M"
    SOURCE_TFLAPI_V1 = "1"
    SOURCE_TFLAPI_V2 = "2"
    SOURCE_TRACKERNET = "K"
    SOURCE_USER = "U"

    SOURCE_CHOICES = (
        (SOURCE_TWITTER, "Twitter"),
        (SOURCE_MASTODON, "Mastodon"),
        (SOURCE_TFLAPI_V1, "TfL API StopPoint Disruption (v1)"),
        (SOURCE_TFLAPI_V2, "TfL API Disruptions Lifts (v2)"),
        (SOURCE_TRACKERNET, "TfL Trackernet"),
        (SOURCE_USER, "User"),
    )

    information = models.BooleanField(help_text="Is this information only?")
    station = models.ForeignKey("stations.Station", on_delete=models.DO_NOTHING)
    text = models.TextField()
    start_time = models.DateTimeField(default=timezone.now)
    end_time = models.DateTimeField(null=True, blank=True)
    resolved = models.BooleanField(default=False)
    source = models.CharField(
        max_length=1,
        choices=SOURCE_CHOICES,
    )

    def __str__(self):
        return f"{self.station.name} - Resolved: {self.resolved} - {self.get_source_display()}"

    class Meta:
        indexes = [
            models.Index(fields=["resolved", "source"]),
            models.Index(fields=["-start_time", "-id"], name="idx_start_time_id"),
            # Note: a standalone station_id index is intentionally omitted —
            # the ForeignKey already creates one
            # (incidents_report_station_id_*), so an explicit idx_station_id
            # was a pure duplicate. Removed in migration 0010.
            models.Index(
                fields=["id"],
                name="report_unresolved_idx",
                condition=models.Q(resolved=False),
            ),
            models.Index(
                fields=["source", "station_id", "information", "text"],
                name="report_live_source_key_idx",
                condition=models.Q(resolved=False),
            ),
        ]
        ordering = ["-start_time", "-id"]


class SourceCheck(models.Model):
    """When each TfL feed was last read successfully. One small row per feed.

    A feed that has not been read for a while is treated as down: its open
    reports stop holding issues open (see consolidate_incidents).
    """

    source = models.CharField(
        max_length=1, primary_key=True, choices=Report.SOURCE_CHOICES
    )
    checked_at = models.DateTimeField()
    # Changes the feed has shown once and must show again before we act on
    # them (see incidents/sources/tflapiv2). Empty for the main feed.
    pending = models.JSONField(default=dict, blank=True)

    def __str__(self):
        return f"{self.get_source_display()} read at {self.checked_at}"


class EscalatorFault(models.Model):
    """One spell of an escalator message in TfL's StopPoint disruption feed.

    Recorded in the background so there is history to predict from later.
    It makes no report or issue and is not shown anywhere. Written only
    when a message enters or leaves the feed.
    """

    # Null for a station we do not hold (most without step-free access).
    station = models.ForeignKey(
        "stations.Station", null=True, blank=True, on_delete=models.SET_NULL,
        related_name="escalator_faults",
    )
    atco_code = models.CharField(max_length=32)
    station_name = models.CharField(max_length=200, blank=True)
    text = models.TextField()
    planned = models.BooleanField(default=False)
    # TfL's own start time for the message, when it gives one.
    source_start = models.DateTimeField(null=True, blank=True)
    first_seen = models.DateTimeField()
    # The last successful read that still listed the message. Only set when
    # the fault closes: it ended between ``last_seen`` and ``cleared_at``.
    last_seen = models.DateTimeField()
    cleared_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-first_seen", "-id"]
        indexes = [
            models.Index(fields=["atco_code", "first_seen"], name="escalator_stop_idx"),
            models.Index(
                fields=["atco_code"],
                name="escalator_open_idx",
                condition=models.Q(cleared_at__isnull=True),
            ),
        ]

    def __str__(self):
        return f"{self.station_name or self.atco_code} from {self.first_seen:%Y-%m-%d %H:%M}"
