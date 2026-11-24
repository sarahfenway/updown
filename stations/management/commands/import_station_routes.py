import logging

import requests
from django.core.management.base import BaseCommand, CommandError

from stations.step_free.importer import (
    ImportAborted,
    download_zip,
    import_feed,
    last_published,
)
from stations.step_free.tfl_data import FeedError, build_feed, read_published, read_zip

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Import TfL's step-free route data when the published file has changed"

    def add_arguments(self, parser):
        parser.add_argument("--zip", help="Import this local zip instead of downloading it")
        parser.add_argument("--force", action="store_true",
                            help="Import even if the publish date has not changed")

    def handle(self, *args, **options):
        try:
            if options.get("zip"):
                with open(options["zip"], "rb") as handle:
                    data = handle.read()
            else:
                data = download_zip()
            tables = read_zip(data)
            published = read_published(tables)
        except (requests.RequestException, FeedError, OSError) as exc:
            # Type only: requests' messages embed the URL with app_key.
            logger.warning("Station data fetch failed (%s); keeping previous data",
                           type(exc).__name__)
            raise CommandError(f"Could not read station data ({type(exc).__name__})")

        if not options.get("force") and published == last_published():
            self.stdout.write(f"Station data unchanged (published {published})")
            return

        try:
            result = import_feed(build_feed(tables))
        except (FeedError, ImportAborted) as exc:
            logger.warning("Station data import aborted: %s", exc)
            raise CommandError(f"Import aborted: {exc}")

        self.stdout.write(self.style.SUCCESS(
            f"Imported step-free routes for {result.matched} stations "
            f"(published {published}; {result.cleared} cleared; "
            f"{len(result.unmatched)} TfL stations unmatched)"
        ))
