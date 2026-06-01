from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("incidents", "0008_live_partial_indexes"),
    ]

    operations = [
        migrations.AddField(
            model_name="incident",
            name="report_count",
            field=models.PositiveIntegerField(blank=True, null=True),
        ),
    ]
