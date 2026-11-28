"""A TfL feed going down, or coming back empty, must not leave issues stuck
open or close them all at once.

All stations, lifts and messages here are made up for the tests.
"""

from datetime import timedelta
from io import StringIO
from unittest.mock import patch

import requests
from django.core.management import call_command
from django.test import TestCase, override_settings
from django.utils import timezone

from incidents.management.commands.update_incidents import (
    LIFT_FEED_MAX_NEW_ISSUES,
    consolidate_incidents,
)
from incidents.ml import report_count_for_prediction
from incidents.models import Incident, Report, SourceCheck
from incidents.sources import tflapiv1, tflapiv2
from incidents.sources.helpers import mark_source_read, stale_feed_sources
from stations.models import Station

V1 = Report.SOURCE_TFLAPI_V1
V2 = Report.SOURCE_TFLAPI_V2


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


def make_station(name="Example Cross", naptan="940GEXAMPLE"):
    station = Station.objects.create(
        name=name, notes="", naptan_id=naptan, hub_naptan_id=f"HUB{naptan}",
        tube=True, national_rail=False,
    )
    station.parent_station = station
    station.save(update_fields=["parent_station"])
    return station


def make_report(station, source, resolved=False, text="Lift unavailable"):
    now = timezone.now()
    return Report.objects.create(
        station=station, text=text, information=False, source=source,
        start_time=now - timedelta(hours=2), resolved=resolved,
        end_time=now - timedelta(minutes=5) if resolved else None,
    )


def make_incident(station, reports, text="Lift unavailable"):
    incident = Incident.objects.create(
        station=station, text=text, information=False, resolved=False,
        start_time=timezone.now() - timedelta(hours=2), report_count=len(reports),
    )
    incident.reports.add(*reports)
    return incident


def read(source, minutes_ago=0):
    mark_source_read(source, timezone.now() - timedelta(minutes=minutes_ago))


class StaleFeedTests(TestCase):
    def test_a_feed_is_stale_until_read_and_again_after_fifteen_minutes(self):
        self.assertEqual(stale_feed_sources(), {V1, V2})
        read(V1, minutes_ago=14)
        read(V2, minutes_ago=16)
        self.assertEqual(stale_feed_sources(), {V2})

    def test_mark_source_read_returns_the_previous_read_time(self):
        first = timezone.now() - timedelta(minutes=5)
        self.assertIsNone(mark_source_read(V1, first))
        self.assertEqual(mark_source_read(V1), first)
        self.assertEqual(SourceCheck.objects.count(), 1)


class ClosingWhenAFeedIsDownTests(TestCase):
    def setUp(self):
        self.station = make_station()

    def test_issue_closes_when_the_working_feed_drops_it_and_the_other_is_down(self):
        read(V1)
        read(V2, minutes_ago=60)
        stuck = make_report(self.station, V2)
        incident = make_incident(
            self.station, [make_report(self.station, V1, resolved=True), stuck]
        )

        posts = consolidate_incidents()

        incident.refresh_from_db()
        stuck.refresh_from_db()
        self.assertTrue(incident.resolved)
        self.assertTrue(stuck.resolved)
        self.assertIsNotNone(stuck.end_time)
        self.assertEqual(posts, ["Step free access has been restored at Example Cross"])

        # The closed-off report must not come back as a new issue.
        self.assertEqual(consolidate_incidents(), [])
        self.assertEqual(Incident.objects.count(), 1)

    def test_issue_stays_open_while_a_working_feed_still_lists_it(self):
        read(V1)
        read(V2)
        incident = make_incident(
            self.station,
            [make_report(self.station, V1, resolved=True), make_report(self.station, V2)],
        )

        consolidate_incidents()

        incident.refresh_from_db()
        self.assertFalse(incident.resolved)

    def test_a_short_blip_does_not_close_anything(self):
        read(V1)
        read(V2, minutes_ago=10)
        incident = make_incident(
            self.station,
            [make_report(self.station, V1, resolved=True), make_report(self.station, V2)],
        )

        consolidate_incidents()

        incident.refresh_from_db()
        self.assertFalse(incident.resolved)

    def test_issue_known_only_to_the_feed_that_is_down_stays_open(self):
        read(V1)
        read(V2, minutes_ago=60)
        incident = make_incident(self.station, [make_report(self.station, V2)])

        consolidate_incidents()

        incident.refresh_from_db()
        self.assertFalse(incident.resolved)

    def test_nothing_closes_on_a_down_feed_when_every_feed_is_down(self):
        read(V1, minutes_ago=60)
        read(V2, minutes_ago=60)
        incident = make_incident(
            self.station,
            [make_report(self.station, V1, resolved=True), make_report(self.station, V2)],
        )

        consolidate_incidents()

        incident.refresh_from_db()
        self.assertFalse(incident.resolved)


class GivingUpOnTheLiftFeedTests(TestCase):
    """The lift feed is the second opinion. Once it has been unreadable for
    six hours, whatever it last said is dropped, without a public post."""

    def setUp(self):
        self.station = make_station()

    def test_issue_known_only_to_the_lift_feed_closes_quietly_after_six_hours(self):
        read(V1)
        read(V2, minutes_ago=6 * 60 + 1)
        report = make_report(self.station, V2)
        incident = make_incident(self.station, [report])

        posts = consolidate_incidents()

        incident.refresh_from_db()
        report.refresh_from_db()
        self.assertTrue(incident.resolved)
        self.assertTrue(report.resolved)
        # We do not know the lift is fixed, so we do not announce that it is.
        self.assertEqual(posts, [])

    def test_issue_known_only_to_the_lift_feed_stays_open_before_six_hours(self):
        read(V1)
        read(V2, minutes_ago=5 * 60)
        incident = make_incident(self.station, [make_report(self.station, V2)])

        consolidate_incidents()

        incident.refresh_from_db()
        self.assertFalse(incident.resolved)

    def test_an_issue_the_main_feed_still_lists_stays_open(self):
        read(V1)
        read(V2, minutes_ago=6 * 60 + 1)
        incident = make_incident(
            self.station,
            [make_report(self.station, V1), make_report(self.station, V2)],
        )

        consolidate_incidents()

        incident.refresh_from_db()
        self.assertFalse(incident.resolved)

    def test_the_main_feed_is_never_given_up_on(self):
        read(V1, minutes_ago=24 * 60)
        read(V2)
        report = make_report(self.station, V1)
        incident = make_incident(self.station, [report])

        consolidate_incidents()

        incident.refresh_from_db()
        report.refresh_from_db()
        self.assertFalse(incident.resolved)
        self.assertFalse(report.resolved)

    def test_leftover_reports_from_a_feed_never_read_open_no_issue(self):
        read(V1)
        leftover = make_report(self.station, V2)

        posts = consolidate_incidents()

        leftover.refresh_from_db()
        self.assertTrue(leftover.resolved)
        self.assertEqual(Incident.objects.count(), 0)
        self.assertEqual(posts, [])


class LiftFeedBurstTests(TestCase):
    """Many brand-new issues from the lift feed in one go means the feed has
    changed its wording, not that the lifts all broke at once."""

    def setUp(self):
        read(V1)
        read(V2)
        self.stations = [
            make_station(f"Example {n}", f"940GEXAMPLE{n}")
            for n in range(LIFT_FEED_MAX_NEW_ISSUES + 1)
        ]

    def test_too_many_new_issues_at_once_are_held_back(self):
        reports = [make_report(station, V2) for station in self.stations]

        posts = consolidate_incidents()

        self.assertEqual(posts, [])
        self.assertEqual(Incident.objects.count(), 0)
        # Held, not thrown away: the feed can still close them itself.
        self.assertEqual(
            Report.objects.filter(pk__in=[r.pk for r in reports], resolved=False).count(),
            len(reports),
        )

    def test_a_few_new_issues_go_through(self):
        for station in self.stations[:LIFT_FEED_MAX_NEW_ISSUES]:
            make_report(station, V2)

        posts = consolidate_incidents()

        self.assertEqual(len(posts), LIFT_FEED_MAX_NEW_ISSUES)
        self.assertEqual(Incident.objects.count(), LIFT_FEED_MAX_NEW_ISSUES)

    def test_reports_that_match_an_open_issue_are_not_new(self):
        incidents = [
            make_incident(station, [make_report(station, V1)])
            for station in self.stations
        ]
        for station in self.stations:
            make_report(station, V2)

        posts = consolidate_incidents()

        self.assertEqual(posts, [])
        self.assertEqual(Incident.objects.count(), len(incidents))
        for incident in incidents:
            self.assertEqual(incident.reports.filter(source=V2).count(), 1)

    def test_the_main_feed_is_not_capped(self):
        for station in self.stations:
            make_report(station, V1)

        consolidate_incidents()

        self.assertEqual(Incident.objects.count(), len(self.stations))


def stop_point_item(station, text="Lift unavailable"):
    return {
        "description": f"{station.name}: No Step Free Access - {text}",
        "atcoCode": station.naptan_id,
        "appearance": "RealTime",
        "additionalInformation": "",
    }


def lift_item(station, text="Lift unavailable", lifts=("L1",)):
    return {
        "stationUniqueId": station.naptan_id,
        "disruptedLiftUniqueIds": [f"{station.naptan_id}-{lift}" for lift in lifts],
        "message": f"{station.name}: No Step Free Access - {text}",
    }


class StopPointFeedReadTests(TestCase):
    def setUp(self):
        self.station = make_station()

    def _check(self, payload, status_code=200):
        with patch.object(
            tflapiv1.requests, "get", return_value=FakeResponse(payload, status_code)
        ):
            tflapiv1.check()

    def test_a_good_read_is_recorded(self):
        self._check([stop_point_item(self.station)])
        self.assertEqual(stale_feed_sources(), {V2})

    def test_an_empty_feed_is_an_outage_not_everything_fixed(self):
        report = make_report(self.station, V1)

        self._check([])

        report.refresh_from_db()
        self.assertFalse(report.resolved)
        self.assertIn(V1, stale_feed_sources())

    def test_an_error_page_is_not_a_good_read(self):
        report = make_report(self.station, V1)

        self._check({"message": "Rate limit exceeded"}, status_code=200)
        self._check([], status_code=503)

        report.refresh_from_db()
        self.assertFalse(report.resolved)
        self.assertIn(V1, stale_feed_sources())


class LiftFeedAsASourceTests(TestCase):
    """The lift feed has to say something on two reads in a row before we
    act on it, whether that is a new fault or a fault going away."""

    def setUp(self):
        self.station = make_station()

    def _check(self, payload):
        with patch(
            "stations.step_free.broken_lifts.requests.get",
            return_value=FakeResponse(payload),
        ):
            tflapiv2.check()

    def _open(self):
        return Report.objects.filter(source=V2, resolved=False)

    def test_a_fault_listed_once_makes_no_report_yet(self):
        self._check([lift_item(self.station)])

        self.assertEqual(Report.objects.filter(source=V2).count(), 0)
        self.assertEqual(stale_feed_sources(), {V1})

    def test_a_fault_listed_twice_makes_a_report_from_when_it_was_first_seen(self):
        self._check([lift_item(self.station)])
        first_seen = timezone.now()
        self._check([lift_item(self.station)])

        report = Report.objects.get(source=V2)
        self.assertEqual(report.text, "Lift unavailable")
        self.assertFalse(report.resolved)
        self.assertLessEqual(report.start_time, first_seen)

    def test_a_fault_listed_once_and_then_gone_never_makes_a_report(self):
        other = make_station("Elsewhere", "940GELSEWHERE")
        self._check([lift_item(self.station), lift_item(other)])
        self._check([lift_item(other)])
        self._check([lift_item(self.station), lift_item(other)])

        self.assertEqual(self._open().filter(station=self.station).count(), 0)
        self.assertEqual(self._open().filter(station=other).count(), 1)

    def test_a_fault_missing_from_one_read_stays_open(self):
        other = make_station("Elsewhere", "940GELSEWHERE")
        report = make_report(self.station, V2)

        self._check([lift_item(other)])

        report.refresh_from_db()
        self.assertFalse(report.resolved)

        # Back again on the next read: it was never really gone.
        self._check([lift_item(self.station), lift_item(other)])
        self._check([lift_item(other)])

        report.refresh_from_db()
        self.assertFalse(report.resolved)

    def test_a_fault_missing_from_two_reads_in_a_row_is_closed(self):
        other = make_station("Elsewhere", "940GELSEWHERE")
        report = make_report(self.station, V2)

        self._check([lift_item(other)])
        self._check([lift_item(other)])

        report.refresh_from_db()
        self.assertTrue(report.resolved)
        self.assertIsNotNone(report.end_time)

    def test_a_failed_read_between_two_good_ones_does_not_count(self):
        other = make_station("Elsewhere", "940GELSEWHERE")
        report = make_report(self.station, V2)

        self._check([lift_item(other)])
        self._check([])

        report.refresh_from_db()
        self.assertFalse(report.resolved)

    def test_a_fault_that_comes_back_gets_a_new_report(self):
        make_report(self.station, V2, resolved=True)
        make_report(self.station, V2, resolved=True)

        self._check([lift_item(self.station)])
        self._check([lift_item(self.station)])

        self.assertEqual(self._open().count(), 1)

    def test_an_empty_feed_is_an_outage_not_everything_fixed(self):
        report = make_report(self.station, V2)

        self._check([])
        self._check([])

        report.refresh_from_db()
        self.assertFalse(report.resolved)
        self.assertIn(V2, stale_feed_sources())

    def test_dated_and_planned_work_is_left_to_the_other_feed(self):
        payload = [
            lift_item(self.station, "Until Spring 2027, no step free access."),
            lift_item(self.station, "Planned maintenance on the lift."),
        ]
        self._check(payload)
        self._check(payload)

        self.assertEqual(Report.objects.filter(source=V2).count(), 0)
        self.assertNotIn(V2, stale_feed_sources())

    def test_malformed_items_and_failures_never_raise(self):
        payload = [{"message": 5}, "nonsense", lift_item(self.station)]
        self._check(payload)
        self._check(payload)
        self.assertEqual(Report.objects.filter(source=V2).count(), 1)

        with patch(
            "stations.step_free.broken_lifts.requests.get",
            side_effect=RuntimeError("unexpected"),
        ):
            tflapiv2.check()


class PredictionInputTests(TestCase):
    """Switching the lift feed on must not change what the model is told."""

    def setUp(self):
        self.station = make_station()

    def test_lift_feed_reports_are_not_counted(self):
        incident = make_incident(
            self.station,
            [make_report(self.station, V1), make_report(self.station, V2)],
        )

        self.assertEqual(report_count_for_prediction(incident), 1)

    def test_other_reports_are_all_counted(self):
        incident = make_incident(
            self.station,
            [
                make_report(self.station, V1),
                make_report(self.station, Report.SOURCE_USER),
                make_report(self.station, V2),
            ],
        )

        self.assertEqual(report_count_for_prediction(incident), 2)

    def test_an_issue_only_the_lift_feed_knows_counts_as_one(self):
        incident = make_incident(self.station, [make_report(self.station, V2)])

        self.assertEqual(report_count_for_prediction(incident), 1)

    def test_prefetched_reports_are_filtered_the_same_way(self):
        incident = make_incident(self.station, [])
        incident.prefetched_reports = [
            make_report(self.station, V1), make_report(self.station, V2),
        ]

        self.assertEqual(report_count_for_prediction(incident), 1)


class UpdateCommandLiftFeedTests(TestCase):
    def _run(self):
        command = "incidents.management.commands.update_incidents"
        with patch(f"{command}.check_tflv1"), \
                patch(f"{command}.check_tflv2") as lift_feed, \
                patch(f"{command}.consolidate_incidents", return_value=[]), \
                patch(f"{command}.update_last_updated"), \
                patch(f"{command}.update_broken_lifts"):
            call_command("update_incidents", stdout=StringIO())
        return lift_feed

    def test_lift_feed_is_a_source_by_default(self):
        self._run().assert_called_once_with()

    @override_settings(TFL_LIFT_FEED_AS_SOURCE=False)
    def test_lift_feed_can_be_switched_off(self):
        self._run().assert_not_called()
