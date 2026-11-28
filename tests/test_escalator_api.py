"""/api/escalators/: escalator faults and nothing else.

All stations and messages are made up.
"""

from datetime import timedelta

from django.test import TestCase
from django.utils import timezone

from incidents import ratelimit
from incidents.models import EscalatorFault, Incident, Report
from incidents.sources.helpers import mark_source_read
from stations.models import Station

FAULT = "Example Cross: reduced escalator service - no down escalator due to a fault."
URL = "/api/escalators/"


def make_station(name, naptan, parent=None, hub=None):
    station = Station.objects.create(
        name=name, notes="", naptan_id=naptan, hub_naptan_id=hub,
        tube=True, national_rail=False,
    )
    station.parent_station = parent or station
    station.save(update_fields=["parent_station"])
    return station


def make_fault(station=None, text=FAULT, hours_ago=3, cleared_hours_ago=None, **fields):
    now = timezone.now()
    defaults = {
        "station": station,
        "atco_code": station.naptan_id if station else "940GELSEWHERE",
        "station_name": f"{station.name} Underground Station" if station else "",
        "text": text,
        "planned": False,
        "first_seen": now - timedelta(hours=hours_ago),
        "last_seen": now - timedelta(hours=hours_ago),
    }
    if cleared_hours_ago is not None:
        defaults["cleared_at"] = now - timedelta(hours=cleared_hours_ago)
        defaults["last_seen"] = defaults["cleared_at"] - timedelta(minutes=5)
    defaults.update(fields)
    return EscalatorFault.objects.create(**defaults)


class EscalatorApiTests(TestCase):
    def setUp(self):
        self.station = make_station("Example Cross", "940GEXAMPLE")
        mark_source_read(Report.SOURCE_TFLAPI_V1)

    def get(self):
        response = self.client.get(URL)
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_a_live_fault_is_listed_with_its_station_and_times(self):
        fault = make_fault(self.station)

        data = self.get()

        self.assertEqual(data["planned"], [])
        self.assertEqual(data["resolved"], [])
        (row,) = data["issues"]
        self.assertEqual(
            set(row),
            {
                "id", "text", "start_time", "end_time", "resolved", "planned",
                "station_name", "station_naptan",
            },
        )
        self.assertEqual(row["id"], fault.pk)
        self.assertEqual(row["text"], FAULT)
        self.assertEqual(row["station_name"], "Example Cross")
        self.assertEqual(row["station_naptan"], "940GEXAMPLE")
        self.assertFalse(row["resolved"])
        self.assertFalse(row["planned"])
        self.assertIsNone(row["end_time"])
        self.assertTrue(row["start_time"].startswith(str(fault.first_seen.year)))

    def test_planned_work_is_listed_separately(self):
        make_fault(self.station, text="Until mid December, no down escalator.", planned=True)
        # TfL sometimes files planned work as a live fault; the wording
        # still says what it is.
        make_fault(
            self.station,
            text="No down escalator in service due to planned maintenance.",
        )

        data = self.get()

        self.assertEqual(data["issues"], [])
        self.assertEqual(len(data["planned"]), 2)
        self.assertTrue(all(row["planned"] for row in data["planned"]))

    def test_faults_that_ended_in_the_last_twelve_hours_are_resolved(self):
        make_fault(self.station, text="Recently fixed escalator", cleared_hours_ago=11)
        make_fault(self.station, text="Long fixed escalator", hours_ago=30,
                   cleared_hours_ago=13)

        data = self.get()

        self.assertEqual(data["issues"], [])
        (row,) = data["resolved"]
        self.assertEqual(row["text"], "Recently fixed escalator")
        self.assertTrue(row["resolved"])
        self.assertIsNotNone(row["end_time"])

    def test_one_fault_listed_at_several_stops_of_a_station_is_one_row(self):
        other_line = make_station(
            "Example Cross", "910GEXAMPLE", parent=self.station
        )
        first = make_fault(self.station, hours_ago=5)
        make_fault(other_line, hours_ago=2)

        (row,) = self.get()["issues"]

        self.assertEqual(row["id"], first.pk)
        self.assertEqual(row["station_naptan"], "940GEXAMPLE")
        # The earliest sighting is when the fault started.
        self.assertEqual(
            row["start_time"][:16], first.first_seen.isoformat()[:16]
        )

    def test_a_fault_still_listed_at_one_stop_is_not_resolved(self):
        other_line = make_station(
            "Example Cross", "910GEXAMPLE", parent=self.station
        )
        make_fault(self.station, hours_ago=5)
        make_fault(other_line, hours_ago=5, cleared_hours_ago=1)

        data = self.get()

        self.assertEqual(len(data["issues"]), 1)
        self.assertEqual(data["resolved"], [])

    def test_different_faults_at_one_station_stay_separate(self):
        make_fault(self.station, text="No up escalator due to a fault.")
        make_fault(self.station, text="No down escalator due to a fault.")

        self.assertEqual(len(self.get()["issues"]), 2)

    def test_a_station_we_do_not_hold_uses_the_feeds_name_and_code(self):
        make_fault(None, station_name="Elsewhere Underground Station")

        (row,) = self.get()["issues"]

        self.assertEqual(row["station_name"], "Elsewhere")
        self.assertEqual(row["station_naptan"], "940GELSEWHERE")

    def test_newest_faults_come_first(self):
        make_fault(self.station, text="Older escalator fault", hours_ago=9)
        make_fault(self.station, text="Newer escalator fault", hours_ago=1)

        texts = [row["text"] for row in self.get()["issues"]]

        self.assertEqual(texts, ["Newer escalator fault", "Older escalator fault"])

    def test_lift_issues_are_not_in_it(self):
        Incident.objects.create(
            station=self.station, text="Lift unavailable", information=False,
            resolved=False,
        )

        data = self.get()

        self.assertEqual(data["issues"] + data["planned"] + data["resolved"], [])


class EscalatorApiFreshnessTests(TestCase):
    def test_says_when_the_feed_was_last_read(self):
        read_at = timezone.now() - timedelta(minutes=4)
        mark_source_read(Report.SOURCE_TFLAPI_V1, read_at)

        data = self.client.get(URL).json()

        self.assertTrue(data["up_to_date"])
        self.assertEqual(data["last_checked"][:16], read_at.isoformat()[:16])
        self.assertIn("last_updated", data)

    def test_is_not_up_to_date_when_the_feed_has_not_been_read_lately(self):
        mark_source_read(
            Report.SOURCE_TFLAPI_V1, timezone.now() - timedelta(minutes=16)
        )

        self.assertFalse(self.client.get(URL).json()["up_to_date"])

    def test_is_not_up_to_date_before_the_feed_has_ever_been_read(self):
        data = self.client.get(URL).json()

        self.assertFalse(data["up_to_date"])
        self.assertIsNone(data["last_checked"])

    def test_the_lift_feed_being_read_does_not_count(self):
        mark_source_read(Report.SOURCE_TFLAPI_V2)

        self.assertFalse(self.client.get(URL).json()["up_to_date"])


class EscalatorApiLimitsTests(TestCase):
    def test_it_is_rate_limited_as_an_api(self):
        self.assertIn(URL, ratelimit.API_PATHS)

    def test_it_only_answers_reads(self):
        self.assertEqual(self.client.post(URL).status_code, 405)


class EscalatorApiCacheTests(TestCase):
    def test_it_is_cached_briefly_and_refreshed_by_the_update(self):
        from unittest.mock import patch

        from django.test import override_settings

        with override_settings(
            FUNCTIONS_SECRET_KEY="verysecret",
            CACHES={
                "default": {
                    "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
                    "LOCATION": "api-escalators-test",
                },
                "ratelimit": {
                    "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
                    "LOCATION": "ratelimit-api-escalators-test",
                },
            },
        ):
            from django.core.cache import cache

            self.addCleanup(cache.clear)
            station = make_station("Example Cross", "940GEXAMPLE")
            make_fault(station)

            first = self.client.get(URL)
            with self.assertNumQueries(0):
                second = self.client.get(URL)
            self.assertEqual(first.content, second.content)
            self.assertIn("no-store", second["Cache-Control"])

            EscalatorFault.objects.update(cleared_at=timezone.now())
            with patch("incidents.views.call_command"):
                self.client.post("/functions/update_incidents", {"key": "verysecret"})

            refreshed = self.client.get(URL).json()
            self.assertEqual(refreshed["issues"], [])
            self.assertEqual(len(refreshed["resolved"]), 1)
