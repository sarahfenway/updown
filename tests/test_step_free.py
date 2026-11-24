import io
import zipfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase, TestCase
from django.utils import timezone

from stations.models import Station

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "stationdata"


def fixture_zip(skip=()):
    """TfL's station data zip, cut down to three stations."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path in sorted(FIXTURE_DIR.glob("*.csv")):
            if path.name not in skip:
                archive.write(path, path.name)
    return buffer.getvalue()


def make_station(name, **fields):
    fields.setdefault("notes", "")
    fields.setdefault("national_rail", False)
    return Station.objects.create(name=name, **fields)


class StationFieldTests(TestCase):
    def test_step_free_fields_round_trip(self):
        station = make_station("Bank", tube=True)
        station.step_free_routes = {"tfl_id": "HUBBAN", "places": {}}
        station.step_free_routes_published = date(2026, 10, 20)
        station.tfl_station_id = "HUBBAN"
        station.broken_lift_ids = ["HUBBAN-Lift-1"]
        station.broken_lifts_checked_at = timezone.now()
        station.save()

        # A fresh object, so plain Python attributes can't fake a pass.
        station = Station.objects.get(pk=station.pk)

        self.assertEqual(station.step_free_routes["tfl_id"], "HUBBAN")
        self.assertEqual(station.step_free_routes_published, date(2026, 10, 20))
        self.assertEqual(station.broken_lift_ids, ["HUBBAN-Lift-1"])

    def test_new_fields_default_to_empty(self):
        station = make_station("Oval", tube=True)
        self.assertIsNone(station.step_free_routes)
        self.assertIsNone(station.tfl_station_id)
        self.assertIsNone(station.broken_lifts_checked_at)
