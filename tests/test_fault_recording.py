"""Background records of lift outages (per lift) and escalator faults.

Nothing here is shown on the site yet: the records build up history for
later predictions. All stations, lifts and messages are made up.
"""

from datetime import datetime, timedelta, timezone as dt_timezone
from unittest.mock import patch

import requests
from django.db import connection
from django.test import TestCase
from django.test.utils import CaptureQueriesContext
from django.utils import timezone

from incidents.models import EscalatorFault, Report
from incidents.sources import tflapiv1
from incidents.sources.helpers import stale_feed_sources
from stations.models import LiftOutage, LiftStatusCheck, Station, StationRoutes
from stations.step_free.broken_lifts import update_broken_lifts

PUBLISHED = datetime(2026, 11, 1, tzinfo=dt_timezone.utc)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = "payload"

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


def make_station(name, naptan):
    return Station.objects.create(
        name=name, notes="", naptan_id=naptan, tube=True, national_rail=False
    )


def writes_to(queries, table):
    return [
        q["sql"] for q in queries.captured_queries
        if q["sql"].split(" ", 1)[0] in ("INSERT", "UPDATE", "DELETE") and table in q["sql"]
    ]


class LiftOutageRecordingTests(TestCase):
    def setUp(self):
        self.station = make_station("Example Cross", "940GEXAMPLE")
        StationRoutes.objects.create(
            station=self.station, tfl_station_id="HUBEXC", tfl_ids=["HUBEXC"],
            published=PUBLISHED, data={},
        )

    def _read(self, payload):
        with patch(
            "stations.step_free.broken_lifts.requests.get",
            return_value=FakeResponse(payload),
        ):
            return update_broken_lifts()

    def _item(self, station_id="HUBEXC", lifts=("Lift-6",),
              message="Made-up Station: lift out of service"):
        return {
            "stationUniqueId": station_id,
            "disruptedLiftUniqueIds": [f"{station_id}-{lift}" for lift in lifts],
            "message": message,
        }

    def test_each_lift_keeps_the_message_it_was_listed_with(self):
        self._read([
            self._item(lifts=("Lift-6",), message="First message"),
            self._item(lifts=("Lift-7",), message="Second message"),
        ])

        self.assertEqual(LiftOutage.objects.get(lift_id="HUBEXC-Lift-7").message, "Second message")

    def test_each_broken_lift_gets_an_open_record(self):
        self._read([self._item(lifts=("Lift-6", "Lift-7")), self._item("HUBXYZ", ("Lift-1",))])

        ours = LiftOutage.objects.get(lift_id="HUBEXC-Lift-6")
        self.assertEqual(ours.station, self.station)
        self.assertEqual(ours.tfl_station_id, "HUBEXC")
        self.assertEqual(ours.message, "Made-up Station: lift out of service")
        self.assertIsNone(ours.cleared_at)
        self.assertEqual(LiftOutage.objects.filter(cleared_at=None).count(), 3)
        # A lift at a station we do not hold is still recorded.
        self.assertIsNone(LiftOutage.objects.get(lift_id="HUBXYZ-Lift-1").station)

    def test_a_lift_that_is_still_broken_writes_nothing(self):
        self._read([self._item()])

        with CaptureQueriesContext(connection) as queries:
            self._read([self._item()])

        self.assertEqual(writes_to(queries, "liftoutage"), [])
        self.assertEqual(LiftOutage.objects.count(), 1)

    def test_a_lift_that_leaves_the_feed_is_closed_between_the_two_reads(self):
        self._read([self._item()])
        earlier = timezone.now() - timedelta(minutes=5)
        LiftStatusCheck.objects.update(checked_at=earlier)

        self._read([self._item("HUBXYZ", ("Lift-1",))])

        outage = LiftOutage.objects.get(lift_id="HUBEXC-Lift-6")
        self.assertEqual(outage.last_seen, earlier)
        self.assertGreater(outage.cleared_at, earlier)

    def test_a_lift_that_breaks_again_gets_a_second_record(self):
        self._read([self._item()])
        self._read([self._item("HUBXYZ", ("Lift-1",))])
        self._read([self._item()])

        self.assertEqual(LiftOutage.objects.filter(lift_id="HUBEXC-Lift-6").count(), 2)

    def test_an_empty_or_failed_read_closes_nothing(self):
        self._read([self._item()])
        StationRoutes.objects.update(broken_lift_ids=["HUBEXC-Lift-6"])

        self.assertFalse(self._read([]))
        with patch(
            "stations.step_free.broken_lifts.requests.get",
            side_effect=requests.ConnectionError("down"),
        ):
            self.assertFalse(update_broken_lifts())

        self.assertIsNone(LiftOutage.objects.get().cleared_at)
        # The station page keeps showing the lift as broken, too.
        self.assertEqual(
            StationRoutes.objects.get().broken_lift_ids, ["HUBEXC-Lift-6"]
        )

    def test_a_recording_failure_does_not_stop_the_station_pages(self):
        with patch(
            "stations.step_free.broken_lifts._record_lift_outages",
            side_effect=RuntimeError("unexpected"),
        ):
            self.assertTrue(self._read([self._item()]))

        self.assertEqual(
            StationRoutes.objects.get().broken_lift_ids, ["HUBEXC-Lift-6"]
        )


def stop_point_item(text, atco="940GEXAMPLE", appearance="RealTime", name="Example Cross"):
    return {
        "description": text,
        "atcoCode": atco,
        "commonName": f"{name} Underground Station",
        "appearance": appearance,
        "additionalInformation": "",
        "fromDate": "2026-11-24T11:40:00Z",
        "toDate": "2026-11-27T01:29:00Z",
    }


ESCALATOR = "Example Cross: reduced escalator service - no down escalator due to a fault."
LIFT = "Example Cross: No Step Free Access - Lift unavailable"


class EscalatorRecordingTests(TestCase):
    def setUp(self):
        self.station = make_station("Example Cross", "940GEXAMPLE")

    def _read(self, payload):
        with patch.object(tflapiv1.requests, "get", return_value=FakeResponse(payload)):
            tflapiv1.check()

    def test_escalator_messages_are_recorded_and_lift_ones_are_not(self):
        self._read([stop_point_item(ESCALATOR), stop_point_item(LIFT)])

        fault = EscalatorFault.objects.get()
        self.assertEqual(fault.station, self.station)
        self.assertEqual(fault.atco_code, "940GEXAMPLE")
        self.assertEqual(fault.station_name, "Example Cross Underground Station")
        self.assertEqual(fault.text, ESCALATOR)
        self.assertFalse(fault.planned)
        self.assertEqual(
            fault.source_start, datetime(2026, 11, 24, 11, 40, tzinfo=dt_timezone.utc)
        )
        self.assertIsNone(fault.cleared_at)
        # Recording escalators never makes a report or an issue.
        self.assertEqual(Report.objects.filter(text__icontains="escalator").count(), 0)

    def test_planned_work_and_unknown_stations_are_recorded(self):
        self._read([
            stop_point_item(
                "ELSEWHERE STATION: Until mid December, no down escalator.",
                atco="940GELSEWHERE", appearance="PlannedWork", name="Elsewhere",
            )
        ])

        fault = EscalatorFault.objects.get()
        self.assertTrue(fault.planned)
        self.assertIsNone(fault.station)
        self.assertEqual(fault.station_name, "Elsewhere Underground Station")

    def test_a_fault_that_is_still_listed_writes_nothing(self):
        self._read([stop_point_item(ESCALATOR)])

        with CaptureQueriesContext(connection) as queries:
            self._read([stop_point_item(ESCALATOR)])

        self.assertEqual(writes_to(queries, "escalatorfault"), [])
        self.assertEqual(EscalatorFault.objects.count(), 1)

    def test_a_fault_that_leaves_the_feed_is_closed_between_the_two_reads(self):
        self._read([stop_point_item(ESCALATOR)])
        first_read = timezone.now() - timedelta(minutes=5)
        from incidents.models import SourceCheck
        SourceCheck.objects.update(checked_at=first_read)

        self._read([stop_point_item(LIFT)])

        fault = EscalatorFault.objects.get()
        self.assertEqual(fault.last_seen, first_read)
        self.assertGreater(fault.cleared_at, first_read)

    def test_an_empty_feed_closes_nothing(self):
        self._read([stop_point_item(ESCALATOR)])

        self._read([])

        self.assertIsNone(EscalatorFault.objects.get().cleared_at)

    def test_a_recording_failure_does_not_stop_the_issue_update(self):
        with patch(
            "incidents.sources.escalators._record", side_effect=RuntimeError("unexpected")
        ):
            self._read([stop_point_item(ESCALATOR), stop_point_item(LIFT)])

        self.assertEqual(Report.objects.count(), 1)
        self.assertNotIn(Report.SOURCE_TFLAPI_V1, stale_feed_sources())
