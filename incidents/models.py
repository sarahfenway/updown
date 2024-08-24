from django.db import models
from django.utils import timezone


class Incident(models.Model):
    information = models.BooleanField(help_text="Is this information only?")
    station = models.ForeignKey("stations.Station", on_delete=models.DO_NOTHING)
    text = models.TextField()
    start_time = models.DateTimeField(default=timezone.now)
    end_time = models.DateTimeField(null=True, blank=True)
    resolved = models.BooleanField(default=False)
    reports = models.ManyToManyField("incidents.Report")

    def __str__(self):
        return f"{self.station.name} - Resolved: {self.resolved} - {self.reports.count()} reports"




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
            models.Index(fields=["station_id","source","information","resolved","text"]),
            models.Index(fields=["resolved", "source"])
        ]