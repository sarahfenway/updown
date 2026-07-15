"""Remove duplicate incidents created by the consolidation loop bug.

TfL published the same disruption twice with inconsistent punctuation
("No Step Free Access - " vs "No Step Free Access- "). Combined with
SequenceMatcher's autojunk heuristic (asymmetric ratios for texts
>= 200 chars), consolidate_incidents created a brand-new incident on
every cron run instead of matching the existing one — 44 open
duplicates for Royal Victoria in July 2026, 110 resolved ones from the
same loop in May 2026.

The dedupe is a pure function of the data and skips groups with no
duplicates, so it is safe to run repeatedly — Django applies the
migration once, but it can also be re-run by hand from a shell:

    import importlib
    from django.apps import apps
    importlib.import_module(
        "incidents.migrations.0008_dedupe_duplicate_incidents"
    ).dedupe_duplicate_incidents(apps, None)
"""

from django.db import migrations
from django.db.models import Count


def _collapse(queryset, group_fields):
    """Keep the newest row of each duplicate group, fold the rest into it.

    The newest row (highest id) is the one the consolidation loop is
    still matching and updating, so it is the keeper. The extras'
    report links are copied onto the keeper before deletion so no
    report loses its incident history.
    """
    duplicate_groups = (
        queryset.values(*group_fields).annotate(n=Count("id")).filter(n__gt=1)
    )

    for group in duplicate_groups:
        rows = list(
            queryset.filter(
                **{field: group[field] for field in group_fields}
            ).order_by("-id")
        )
        keeper, extras = rows[0], rows[1:]
        for extra in extras:
            keeper.reports.add(*extra.reports.all())
            extra.delete()


def dedupe_duplicate_incidents(apps, schema_editor):
    Incident = apps.get_model("incidents", "Incident")

    # Open incidents: consolidate_incidents can never legitimately hold
    # two unresolved incidents with identical text for one station (an
    # identical-text report always matches the existing incident), so
    # any such pair is an artifact of the bug — regardless of
    # start_time, which the loop back-dated to the report's start.
    _collapse(Incident.objects.filter(resolved=False), ("station_id", "text"))

    # Resolved history: identical text alone is legitimate there (the
    # same lift fails the same way months apart), so also require an
    # identical start_time — a real recurrence starts at a new moment,
    # only the bug stamps many incidents with one report's start_time.
    _collapse(
        Incident.objects.filter(resolved=True),
        ("station_id", "text", "start_time"),
    )


class Migration(migrations.Migration):
    dependencies = [
        ("incidents", "0010_remove_report_idx_station_id"),
    ]

    operations = [
        migrations.RunPython(
            dedupe_duplicate_incidents, migrations.RunPython.noop
        ),
    ]
