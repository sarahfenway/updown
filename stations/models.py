from django.db import models


class Station(models.Model):
    STEP_ZERO_TO_FIFTY_MM = "G"
    STEP_FIFTY_TO_ONE_TWENTY_MM = "A"
    STEP_OVER_ONE_TWENTY_MM = "R"

    STEP_BETWEEN_PLATFORM_AND_TRAIN_CHOICES = (
        (STEP_ZERO_TO_FIFTY_MM, "0 - 50mm (0 - 2 inches)"),
        (STEP_FIFTY_TO_ONE_TWENTY_MM, "51 - 120mm (2 - 4.7 inches)"),
        (STEP_OVER_ONE_TWENTY_MM, "Over 120mm (4.7 inches)"),
    )

    GAP_ZERO_TO_EIGHTY_FIVE_MM = "A"
    GAP_EIGHTY_FIVE_TO_ONE_EIGHTY_MM = "B"
    GAP_OVER_ONE_EIGHTY_MM = "C"

    GAP_BETWEEN_PLATFORM_AND_TRAIN_CHOICES = (
        (GAP_ZERO_TO_EIGHTY_FIVE_MM, "0 - 85mm (0 - 3.3 inches)"),
        (GAP_EIGHTY_FIVE_TO_ONE_EIGHTY_MM, "86 - 180mm (3.3 - 7 inches)"),
        (GAP_OVER_ONE_EIGHTY_MM, "Over 180mm (7 inches)"),
    )

    name = models.CharField(max_length=200)
    alternate_names = models.JSONField(null=True)
    naptan_id = models.CharField(
        max_length=32, null=True, blank=True, verbose_name="NaPTAN ID"
    )
    hub_naptan_id = models.CharField(
        max_length=32, null=True, blank=True, verbose_name="Hub NaPTAN ID"
    )
    notes = models.TextField()

    step_to_train = models.CharField(
        max_length=1,
        choices=STEP_BETWEEN_PLATFORM_AND_TRAIN_CHOICES,
        blank=True,
        null=True,
    )

    gap_to_train = models.CharField(
        max_length=1,
        choices=GAP_BETWEEN_PLATFORM_AND_TRAIN_CHOICES,
        blank=True,
        null=True,
    )

    designated_boarding_point = models.BooleanField(null=True, blank=True)
    access_via_ramp = models.BooleanField(null=True, blank=True)
    access_via_lift = models.BooleanField(null=True, blank=True)
    connection_to_nr = models.BooleanField(
        null=True, blank=True, verbose_name="Connection to National Rail"
    )

    tube = models.BooleanField(null=True, blank=True)
    dlr = models.BooleanField(null=True, blank=True, verbose_name="DLR")
    national_rail = models.BooleanField(null=True, verbose_name="National Rail")
    crossrail = models.BooleanField(null=True, blank=True)
    overground = models.BooleanField(null=True, blank=True)

    parent_station = models.ForeignKey(
        "stations.Station", on_delete=models.DO_NOTHING, null=True, blank=True
    )

    def __str__(self):
        return self.name

    class Meta:
        indexes = [
            models.Index(fields=["name"], name="station_name_idx"),
            models.Index(fields=["naptan_id"], name="station_naptan_idx"),
            models.Index(fields=["hub_naptan_id"], name="station_hub_naptan_idx"),
        ]


class StationRoutes(models.Model):
    """Step-free route data for one station, from TfL's station data file.

    Kept off ``Station`` on purpose: station rows are loaded on every page
    and API request (incidents select_related them), and this JSON is
    3–22 KB per station. Written by the weekly ``import_station_routes``.
    """

    station = models.OneToOneField(
        Station, on_delete=models.CASCADE, primary_key=True, related_name="step_free"
    )
    tfl_station_id = models.CharField(max_length=32, db_index=True, verbose_name="TfL station id")
    # Every TfL station merged into this one (see tfl_data.merge_routes).
    tfl_ids = models.JSONField(default=list)
    # FeedStartDate of the TfL file this came from.
    published = models.DateTimeField()
    data = models.JSONField()
    # Lift ids TfL reports out of service here. Written only when it changes.
    broken_lift_ids = models.JSONField(default=list)

    def __str__(self):
        return f"Step-free routes for {self.station}"


class LiftStatusCheck(models.Model):
    """When TfL's lift feed was last read successfully. One row (pk 1).

    A single timestamp instead of one per station, so a 5-minute run that
    finds no change writes one small row, not ~420 large ones.
    """

    checked_at = models.DateTimeField()
