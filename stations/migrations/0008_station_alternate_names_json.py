from django.contrib.postgres.fields import ArrayField
from django.db import migrations, models


def copy_alternate_names_to_json(apps, schema_editor):
    Station = apps.get_model("stations", "Station")
    for station in Station.objects.exclude(alternate_names__isnull=True).iterator():
        station.alternate_names_json = list(station.alternate_names)
        station.save(update_fields=["alternate_names_json"])


def copy_alternate_names_from_json(apps, schema_editor):
    Station = apps.get_model("stations", "Station")
    for station in Station.objects.exclude(
        alternate_names_json__isnull=True
    ).iterator():
        station.alternate_names = list(station.alternate_names_json)
        station.save(update_fields=["alternate_names"])


class Migration(migrations.Migration):
    dependencies = [
        (
            "stations",
            "0007_station_station_name_idx_station_station_naptan_idx_and_more",
        ),
    ]

    operations = [
        migrations.AddField(
            model_name="station",
            name="alternate_names_json",
            field=models.JSONField(null=True),
        ),
        migrations.RunPython(
            copy_alternate_names_to_json,
            copy_alternate_names_from_json,
        ),
        migrations.RemoveField(
            model_name="station",
            name="alternate_names",
        ),
        migrations.RenameField(
            model_name="station",
            old_name="alternate_names_json",
            new_name="alternate_names",
        ),
    ]
