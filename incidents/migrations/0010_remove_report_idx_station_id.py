from django.db import migrations


class Migration(migrations.Migration):
    """Drop the redundant standalone station_id index on Report.

    The ForeignKey to Station already creates an index on station_id
    (incidents_report_station_id_*), so the explicit ``idx_station_id``
    was a duplicate — pure write overhead and ~22MB of dead weight on a
    2M-row table. The FK index continues to serve every station_id
    lookup, so nothing that used idx_station_id loses its index.
    """

    dependencies = [
        ("incidents", "0009_incident_report_count"),
    ]

    operations = [
        migrations.RemoveIndex(
            model_name="report",
            name="idx_station_id",
        ),
    ]
