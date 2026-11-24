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


from stations.step_free.labels import is_dropped, place_label
from stations.step_free.layout import compute_layout


class LabelTests(SimpleTestCase):
    def test_line_and_direction_codes(self):
        self.assertEqual(place_label("CenEB"), "Central line eastbound platform")
        self.assertEqual(place_label("EL WB"), "Elizabeth line westbound platform")
        self.assertEqual(place_label("NORPL"), "Northern line platforms")

    def test_common_areas(self):
        cases = [
            ("BookH", "Ticket hall"), ("DSBH", "Ticket hall"), ("BookW", "Ticket hall"),
            ("UpCon", "Upper concourse"), ("MiCon", "Middle concourse"),
            ("FBdg", "Footbridge"), ("Sub", "Subway"), ("RPL-1", "Platform level"),
            ("MET", "Platform level"), ("Entr", "Entrance"), ("IENT", "Entrance"),
            ("A-TOT", "Entrance"), ("Jubil", "Jubilee line level"),
        ]
        for code, label in cases:
            with self.subTest(code=code):
                self.assertEqual(place_label(code), label)

    def test_unknown_code_has_no_label(self):
        self.assertIsNone(place_label("BCQ"))

    def test_bus_and_taxi_points_are_dropped(self):
        self.assertTrue(is_dropped("Bus C"))
        self.assertTrue(is_dropped("BUS"))
        self.assertTrue(is_dropped("Taxi"))
        self.assertFalse(is_dropped("BookH"))


class LayoutTests(SimpleTestCase):
    def test_rows_run_from_highest_level_and_columns_west_to_east(self):
        places = {
            "street": {"level": 0, "kind": "street", "lon": None},
            "east": {"level": 0, "kind": "entrance", "lon": -0.10},
            "west": {"level": 0, "kind": "entrance", "lon": -0.20},
            "bridge": {"level": 1, "kind": "place", "lon": -0.15},
            "platform": {"level": -2, "kind": "platform", "lon": None},
        }

        layout = compute_layout(places)

        self.assertEqual(layout["levels"], [1, 0, -2])
        self.assertEqual(layout["columns"], 2)
        self.assertEqual(layout["positions"]["bridge"], [0, 0])
        self.assertEqual(layout["positions"]["west"], [0, 1])
        self.assertEqual(layout["positions"]["east"], [1, 1])
        self.assertEqual(layout["positions"]["platform"], [0, 2])
        self.assertNotIn("street", layout["positions"])
