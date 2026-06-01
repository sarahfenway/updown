import csv
import gzip
import os
import time
from datetime import timedelta

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import OperationalError, connection, transaction
from django.utils import timezone
from django.utils.dateparse import parse_datetime

from incidents.models import Incident, Report


REPORT_FIELDS = (
    "id",
    "information",
    "station_id",
    "text",
    "start_time",
    "end_time",
    "resolved",
    "source",
)


def _csv_value(value):
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _default_output_dir():
    sqlite_path = os.getenv("SQLITE_PATH")
    if sqlite_path:
        return os.path.join(os.path.dirname(sqlite_path), "report_archives")
    return os.path.join(settings.BASE_DIR, "report_archives")


class Command(BaseCommand):
    help = "Archive old resolved reports to CSV before optionally deleting them"

    def add_arguments(self, parser):
        parser.add_argument(
            "--older-than-days",
            type=int,
            default=90,
            help="Archive reports resolved before this many days ago (default: 90).",
        )
        parser.add_argument(
            "--before",
            help="Archive reports resolved before this ISO datetime instead.",
        )
        parser.add_argument(
            "--output-dir",
            default=None,
            help="Directory for reports_*.csv.gz and incident_report_links_*.csv.gz.",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=1000,
            help="How many reports to export/delete per batch.",
        )
        parser.add_argument(
            "--delete",
            action="store_true",
            help="Delete archived reports after their rows and M2M links are exported.",
        )
        parser.add_argument(
            "--skip-report-count-backfill",
            action="store_true",
            help="Do not fill Incident.report_count before deleting.",
        )
        parser.add_argument(
            "--backfill-report-count",
            action="store_true",
            help="Fill Incident.report_count even when running without --delete.",
        )
        parser.add_argument(
            "--lock-timeout",
            type=int,
            default=120,
            help="Seconds to keep retrying when SQLite reports database is locked.",
        )
        parser.add_argument(
            "--retry-delay",
            type=float,
            default=2.0,
            help="Initial seconds to sleep between lock retries.",
        )

    def handle(self, *args, **options):
        batch_size = max(1, options["batch_size"])
        if connection.features.max_query_params:
            batch_size = min(batch_size, connection.features.max_query_params)
        self.lock_timeout = max(0, options["lock_timeout"])
        self.retry_delay = max(0.1, options["retry_delay"])
        self._configure_sqlite_timeout()
        cutoff = self._cutoff(options)
        output_dir = options["output_dir"] or _default_output_dir()
        os.makedirs(output_dir, exist_ok=True)

        if options["skip_report_count_backfill"] and options["backfill_report_count"]:
            raise CommandError(
                "--skip-report-count-backfill and --backfill-report-count conflict"
            )

        should_backfill = options["backfill_report_count"] or (
            options["delete"] and not options["skip_report_count_backfill"]
        )
        if should_backfill:
            backfilled = self._backfill_report_counts(batch_size)
            self.stdout.write(f"Backfilled report_count on {backfilled} incident(s)")

        timestamp = timezone.now().strftime("%Y%m%d%H%M%S")
        reports_path = os.path.join(output_dir, f"reports_{timestamp}.csv.gz")
        links_path = os.path.join(
            output_dir, f"incident_report_links_{timestamp}.csv.gz"
        )

        self.stdout.write(
            f"Archiving reports resolved before {cutoff.isoformat()}"
        )

        exported = 0
        deleted = 0
        with gzip.open(reports_path, "wt", newline="") as reports_file, gzip.open(
            links_path, "wt", newline=""
        ) as links_file:
            report_writer = csv.writer(reports_file)
            link_writer = csv.writer(links_file)
            report_writer.writerow(REPORT_FIELDS)

            through = Incident.reports.through
            link_fields = [field.attname for field in through._meta.fields]
            link_writer.writerow(link_fields)

            for report_ids in self._candidate_id_batches(cutoff, batch_size):
                self._write_reports(report_writer, report_ids)
                self._write_links(link_writer, through, link_fields, report_ids)
                exported += len(report_ids)

                if options["delete"]:
                    self._delete_reports(report_ids)
                    deleted += len(report_ids)

        self.stdout.write(f"Wrote {reports_path}")
        self.stdout.write(f"Wrote {links_path}")
        if options["delete"]:
            self.stdout.write(self.style.SUCCESS(f"Deleted {deleted} row(s)"))
        else:
            self.stdout.write(
                self.style.WARNING(
                    f"Archive complete: exported {exported} report(s); "
                    "rerun with --delete to remove them."
                )
            )

    def _cutoff(self, options):
        if options["before"]:
            cutoff = parse_datetime(options["before"])
            if cutoff is None:
                raise CommandError("--before must be an ISO datetime")
            if timezone.is_naive(cutoff):
                cutoff = timezone.make_aware(cutoff)
            return cutoff
        return timezone.now() - timedelta(days=options["older_than_days"])

    def _candidate_id_batches(self, cutoff, batch_size):
        last_id = 0
        while True:
            report_ids = self._with_lock_retry(
                "load candidate report ids",
                lambda: self._fetch_candidate_ids(cutoff, last_id, batch_size),
            )
            if not report_ids:
                break
            yield report_ids
            last_id = report_ids[-1]

    def _backfill_report_counts(self, batch_size):
        updated = 0
        while True:
            incidents = self._with_lock_retry(
                "load incident report counts",
                lambda: self._fetch_report_counts_to_backfill(batch_size),
            )
            if not incidents:
                break

            self._with_lock_retry(
                "backfill incident report counts",
                lambda: self._update_report_counts(incidents),
            )
            updated += len(incidents)

        return updated

    def _write_reports(self, writer, report_ids):
        rows = self._with_lock_retry(
            "load report archive rows",
            lambda: self._fetch_report_rows(report_ids),
        )
        for row in rows:
            writer.writerow([_csv_value(value) for value in row])

    def _write_links(self, writer, through, link_fields, report_ids):
        rows = self._with_lock_retry(
            "load report link archive rows",
            lambda: self._fetch_link_rows(through, link_fields, report_ids),
        )
        for row in rows:
            writer.writerow([_csv_value(value) for value in row])

    def _delete_reports(self, report_ids):
        def delete_batch():
            with transaction.atomic():
                through = Incident.reports.through
                with connection.cursor() as cursor:
                    placeholders = self._placeholders(report_ids)
                    cursor.execute(
                        f"DELETE FROM {self._qn(through._meta.db_table)} "
                        f"WHERE {self._qn('report_id')} IN ({placeholders})",
                        report_ids,
                    )
                    cursor.execute(
                        f"DELETE FROM {self._qn(Report._meta.db_table)} "
                        f"WHERE {self._qn('id')} IN ({placeholders})",
                        report_ids,
                    )

        self._with_lock_retry("delete archived reports", delete_batch)

    def _fetch_candidate_ids(self, cutoff, last_id, batch_size):
        through = Incident.reports.through
        sql = f"""
            SELECT r.{self._qn('id')}
            FROM {self._qn(Report._meta.db_table)} r
            WHERE r.{self._qn('resolved')} = %s
              AND r.{self._qn('end_time')} IS NOT NULL
              AND r.{self._qn('end_time')} < %s
              AND r.{self._qn('id')} > %s
              AND NOT EXISTS (
                  SELECT 1
                  FROM {self._qn(through._meta.db_table)} ir
                  JOIN {self._qn(Incident._meta.db_table)} i
                    ON i.{self._qn('id')} = ir.{self._qn('incident_id')}
                  WHERE ir.{self._qn('report_id')} = r.{self._qn('id')}
                    AND i.{self._qn('resolved')} = %s
              )
            ORDER BY r.{self._qn('id')}
            LIMIT %s
        """
        with connection.cursor() as cursor:
            cursor.execute(sql, [True, cutoff, last_id, False, batch_size])
            return [row[0] for row in cursor.fetchall()]

    def _fetch_report_counts_to_backfill(self, batch_size):
        through = Incident.reports.through
        sql = f"""
            SELECT i.{self._qn('id')}, (
                SELECT COUNT(*)
                FROM {self._qn(through._meta.db_table)} ir
                WHERE ir.{self._qn('incident_id')} = i.{self._qn('id')}
            ) AS current_report_count
            FROM {self._qn(Incident._meta.db_table)} i
            WHERE i.{self._qn('report_count')} IS NULL
            ORDER BY i.{self._qn('id')}
            LIMIT %s
        """
        with connection.cursor() as cursor:
            cursor.execute(sql, [batch_size])
            return cursor.fetchall()

    def _update_report_counts(self, incident_counts):
        sql = (
            f"UPDATE {self._qn(Incident._meta.db_table)} "
            f"SET {self._qn('report_count')} = %s "
            f"WHERE {self._qn('id')} = %s"
        )
        with connection.cursor() as cursor:
            cursor.executemany(
                sql,
                [
                    (current_report_count, incident_id)
                    for incident_id, current_report_count in incident_counts
                ],
            )

    def _fetch_report_rows(self, report_ids):
        placeholders = self._placeholders(report_ids)
        columns = ", ".join(self._qn(field) for field in REPORT_FIELDS)
        sql = (
            f"SELECT {columns} "
            f"FROM {self._qn(Report._meta.db_table)} "
            f"WHERE {self._qn('id')} IN ({placeholders}) "
            f"ORDER BY {self._qn('id')}"
        )
        with connection.cursor() as cursor:
            cursor.execute(sql, report_ids)
            return cursor.fetchall()

    def _fetch_link_rows(self, through, link_fields, report_ids):
        placeholders = self._placeholders(report_ids)
        columns = ", ".join(self._qn(field) for field in link_fields)
        sql = (
            f"SELECT {columns} "
            f"FROM {self._qn(through._meta.db_table)} "
            f"WHERE {self._qn('report_id')} IN ({placeholders}) "
            f"ORDER BY {self._qn('report_id')}, {self._qn('incident_id')}"
        )
        with connection.cursor() as cursor:
            cursor.execute(sql, report_ids)
            return cursor.fetchall()

    def _placeholders(self, values):
        return ", ".join(["%s"] * len(values))

    def _qn(self, name):
        return connection.ops.quote_name(name)

    def _configure_sqlite_timeout(self):
        if connection.vendor != "sqlite":
            return

        timeout_ms = int(self.lock_timeout * 1000)
        with connection.cursor() as cursor:
            cursor.execute(f"PRAGMA busy_timeout = {timeout_ms}")

    def _with_lock_retry(self, label, operation):
        deadline = time.monotonic() + self.lock_timeout
        delay = self.retry_delay

        while True:
            try:
                return operation()
            except OperationalError as exc:
                if not self._is_database_locked(exc) or time.monotonic() >= deadline:
                    raise

                sleep_for = min(delay, max(0.1, deadline - time.monotonic()))
                self.stderr.write(
                    self.style.WARNING(
                        f"Database locked while trying to {label}; "
                        f"retrying in {sleep_for:.1f}s"
                    )
                )
                time.sleep(sleep_for)
                delay = min(delay * 2, 10)

    def _is_database_locked(self, exc):
        message = str(exc).lower()
        return "database is locked" in message or "database table is locked" in message
