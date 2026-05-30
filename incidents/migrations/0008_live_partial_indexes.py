from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("incidents", "0007_mlmodel"),
    ]

    operations = [
        migrations.AddIndex(
            model_name="incident",
            index=models.Index(
                fields=["id"],
                name="incident_unresolved_idx",
                condition=models.Q(resolved=False),
            ),
        ),
        migrations.AddIndex(
            model_name="report",
            index=models.Index(
                fields=["id"],
                name="report_unresolved_idx",
                condition=models.Q(resolved=False),
            ),
        ),
        migrations.AddIndex(
            model_name="report",
            index=models.Index(
                fields=["source", "station_id", "information", "text"],
                name="report_live_source_key_idx",
                condition=models.Q(resolved=False),
            ),
        ),
    ]
