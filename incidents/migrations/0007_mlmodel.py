import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("incidents", "0006_incident_prediction_confidence"),
    ]

    operations = [
        migrations.CreateModel(
            name="MLModel",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("data", models.BinaryField()),
                ("size_bytes", models.PositiveIntegerField()),
                (
                    "uploaded_at",
                    models.DateTimeField(default=django.utils.timezone.now),
                ),
            ],
            options={"ordering": ["-id"]},
        ),
    ]
