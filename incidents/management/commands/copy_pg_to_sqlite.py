"""Copy data from a Postgres source into the configured default DB.

Used as a one-shot during the DO Postgres → Fly SQLite migration. We
attach the source as a second Django DB at runtime (so the ORM handles
all the type translation: DurationField microseconds, JSONField encoding,
BinaryField bytes), then stream rows model-by-model in FK dependency
order.

The destination is whatever ``DATABASES["default"]`` resolves to — in
practice the new SQLite file, after migrations have created the schema.
We turn off SQLite FK enforcement for the duration of the copy so the
self-referential parent_station FK on Station doesn't force a
hand-written two-pass.

Usage:

    python -m django copy_pg_to_sqlite \
        --source-host=...  --source-port=25060 \
        --source-db=...    --source-user=...  \
        --source-password=... \
        --settings=updown.settings

The SQLite path comes from the SQLITE_PATH env var via settings.py.
"""

from django.conf import settings
from django.contrib.auth.models import Group, Permission, User
from django.contrib.contenttypes.models import ContentType
from django.core.management.base import BaseCommand, CommandError
from django.db import connections

from incidents.models import Incident, MLModel, Report
from stations.models import Station


# Order matters: each entry depends only on those before it. Station is
# self-referential, but we run with foreign_keys=OFF so the parent_station
# pointers don't have to be ordered.
#
# ContentType + Permission are populated by ``migrate`` already, but the
# auto-generated pks will differ from prod. We copy them so admin users'
# group/permission FKs still resolve correctly.
COPY_ORDER = [
    ContentType,
    Permission,
    Group,
    User,
    Station,
    MLModel,
    Report,
    Incident,
]

# M2M auto-through tables for the models we care about. Each entry is
# (through_class, friendly_name) — both sides of the relationship must
# have been copied above before we hit the through table.
M2M_THROUGHS = [
    (User.groups.through, "user_groups"),
    (User.user_permissions.through, "user_user_permissions"),
    (Group.permissions.through, "group_permissions"),
    (Incident.reports.through, "incident_reports"),
]


SOURCE_ALIAS = "pg_source"


class Command(BaseCommand):
    help = "Copy all app data from a Postgres source DB into the default DB."

    def add_arguments(self, parser):
        parser.add_argument("--source-host", required=True)
        parser.add_argument("--source-port", default="5432")
        parser.add_argument("--source-db", required=True)
        parser.add_argument("--source-user", required=True)
        parser.add_argument("--source-password", required=True)
        parser.add_argument(
            "--source-sslmode",
            default="require",
            help="psycopg2 sslmode (default: require, matches DO Managed PG)",
        )
        parser.add_argument(
            "--chunk-size",
            type=int,
            default=2000,
            help="Rows fetched / inserted per batch (default: 2000)",
        )
        parser.add_argument(
            "--allow-non-empty",
            action="store_true",
            help="Skip the safety check that the destination DB is empty.",
        )

    def handle(self, **opts):
        # Wire up the source DB without writing it to settings.py
        # permanently. Mutating settings.DATABASES + invalidating Django's
        # cached_property is the supported way to add a connection at
        # runtime; we don't assign to ``connections.databases`` directly
        # because it's a cached_property and assignment raises.
        settings.DATABASES[SOURCE_ALIAS] = {
            "ENGINE": "django.db.backends.postgresql_psycopg2",
            "NAME": opts["source_db"],
            "USER": opts["source_user"],
            "PASSWORD": opts["source_password"],
            "HOST": opts["source_host"],
            "PORT": opts["source_port"],
            "OPTIONS": {"sslmode": opts["source_sslmode"]},
            "TIME_ZONE": None,
            "CONN_MAX_AGE": 0,
            "CONN_HEALTH_CHECKS": False,
            "AUTOCOMMIT": True,
            "ATOMIC_REQUESTS": False,
        }
        # Clear the cached_property so the next access re-reads settings.
        connections.__dict__.pop("databases", None)
        connections.__dict__.pop("settings", None)

        chunk_size = opts["chunk_size"]

        # ContentType + Permission are auto-populated by ``migrate``. Their
        # rows are valid but their PKs probably differ from the source DB,
        # which would break FKs from User/Group/Permission. We wipe them
        # so the source PKs can be inserted clean.
        auto_populated = {ContentType, Permission}

        # Safety net: refuse to clobber a destination DB that already has
        # data we'd silently overwrite. Auto-populated tables are
        # whitelisted because we're about to wipe them deliberately.
        if not opts["allow_non_empty"]:
            for model in COPY_ORDER:
                if model in auto_populated:
                    continue
                count = model.objects.using("default").count()
                if count:
                    raise CommandError(
                        f"Destination already has {count} {model.__name__} "
                        f"rows. Re-run with --allow-non-empty if that's "
                        f"really what you want."
                    )

        # Disable FK enforcement during load. SQLite-only; psycopg2
        # accepts and ignores the pragma. We restore it at the end.
        is_sqlite = (
            settings.DATABASES["default"]["ENGINE"].endswith("sqlite3")
        )
        if is_sqlite:
            with connections["default"].cursor() as cur:
                cur.execute("PRAGMA foreign_keys=OFF;")

        try:
            # Clear the auto-populated tables in reverse FK order so
            # nothing references them when they're emptied.
            for model in [Permission, ContentType]:
                deleted, _ = model.objects.using("default").all().delete()
                if deleted:
                    self.stdout.write(
                        f"Cleared {deleted} auto-populated {model.__name__} rows"
                    )

            for model in COPY_ORDER:
                self._copy_model(model, chunk_size)

            for through, name in M2M_THROUGHS:
                self._copy_m2m_through(through, chunk_size, name)
        finally:
            if is_sqlite:
                with connections["default"].cursor() as cur:
                    cur.execute("PRAGMA foreign_keys=ON;")

        self.stdout.write(self.style.SUCCESS("\nDone."))

    # ----------------------------------------------------------------------

    def _copy_model(self, model, chunk_size):
        label = model.__name__
        src_total = model.objects.using(SOURCE_ALIAS).count()
        self.stdout.write(f"\n{label}: {src_total} rows in source")
        if src_total == 0:
            return

        # ``iterator(chunk_size=...)`` paginates server-side so we don't
        # buffer 2M rows. We sort by pk so resumes (and progress logs)
        # are deterministic.
        qs = model.objects.using(SOURCE_ALIAS).order_by("pk").iterator(
            chunk_size=chunk_size
        )

        batch = []
        copied = 0
        for obj in qs:
            # Detach from the source connection so save/bulk_create on
            # ``default`` doesn't try to issue an UPDATE.
            obj._state.adding = True
            obj._state.db = None
            batch.append(obj)
            if len(batch) >= chunk_size:
                model.objects.using("default").bulk_create(
                    batch, batch_size=chunk_size
                )
                copied += len(batch)
                self._progress(label, copied, src_total)
                batch = []

        if batch:
            model.objects.using("default").bulk_create(
                batch, batch_size=chunk_size
            )
            copied += len(batch)
            self._progress(label, copied, src_total)

    def _copy_m2m_through(self, through, chunk_size, name=None):
        # Auto-generated M2M tables don't have a Django model in our app
        # — we reach for the ORM-owned through class instead. Same
        # streaming approach. ``name`` overrides the default class name
        # for log output (e.g. "user_groups" vs "User_groups").
        label = name or through.__name__
        src_total = through.objects.using(SOURCE_ALIAS).count()
        self.stdout.write(f"\n{label}: {src_total} rows in source")
        if src_total == 0:
            return

        qs = (
            through.objects
            .using(SOURCE_ALIAS)
            .order_by("pk")
            .iterator(chunk_size=chunk_size)
        )

        batch = []
        copied = 0
        for obj in qs:
            obj._state.adding = True
            obj._state.db = None
            batch.append(obj)
            if len(batch) >= chunk_size:
                through.objects.using("default").bulk_create(
                    batch, batch_size=chunk_size
                )
                copied += len(batch)
                self._progress(label, copied, src_total)
                batch = []

        if batch:
            through.objects.using("default").bulk_create(
                batch, batch_size=chunk_size
            )
            copied += len(batch)
            self._progress(label, copied, src_total)

    def _progress(self, label, copied, total):
        pct = (copied / total * 100) if total else 100.0
        self.stdout.write(f"  {label}: {copied:>8}/{total} ({pct:.1f}%)")
