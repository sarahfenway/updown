from django.contrib.postgres.fields import ArrayField
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
    alternate_names = ArrayField(models.CharField(max_length=128), null=True)
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
