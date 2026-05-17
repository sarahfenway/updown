import arrow
import json
import os
import tempfile

from datetime import datetime, timedelta, timezone as dt_timezone
from io import StringIO
from unittest.mock import MagicMock, patch

try:
    import numpy as np
except ImportError:  # pragma: no cover - local test env may not have ML deps
    np = None

from django.core.management import CommandError, call_command
from django.db import connection
from django.db.models import Count
from django.test import SimpleTestCase, TestCase, override_settings
from django.test.utils import CaptureQueriesContext
from django.utils import timezone

from incidents.management.commands.update_incidents import consolidate_incidents
from incidents.ml import predict_duration, prediction_outcome
from incidents.models import Incident, Report
from incidents.sources import tflapiv1, tflapiv2
from incidents.utils import (
    find_dates,
    fix_additional_info_grammar,
    get_last_updated,
    parse_date,
    remove_tfl_specifics,
    send_bluesky,
    send_tweet,
    update_last_updated,
)
from incidents.views import (
    _prediction_bucket_metrics,
    _prediction_confidence_label,
    _prediction_display_policy,
    _prediction_station_overrides,
    _prepare_incidents,
)
from stations.models import Station


class FakeResponse:
    def __init__(self, payload, status_code=200, text=None):
        self._payload = payload
        self.status_code = status_code
        self.text = text if text is not None else "payload"

    def json(self):
        return self._payload


class StationFactoryMixin:
    station_index = 0

    def create_parent_station(self, name="Bank", **overrides):
        self.__class__.station_index += 1
        defaults = {
            "name": name,
            "notes": "",
            "naptan_id": f"PARENT{self.station_index}",
            "hub_naptan_id": f"HUB{self.station_index}",
            "tube": True,
            "national_rail": False,
        }
        defaults.update(overrides)
        station = Station.objects.create(**defaults)
        station.parent_station = station
        station.save(update_fields=["parent_station"])
        return station

    def create_child_station(self, parent_station, name=None, **overrides):
        self.__class__.station_index += 1
        defaults = {
            "name": name or f"{parent_station.name} Platform",
            "notes": "",
            "naptan_id": f"CHILD{self.station_index}",
            "hub_naptan_id": parent_station.hub_naptan_id,
            "tube": True,
            "national_rail": False,
            "parent_station": parent_station,
        }
        defaults.update(overrides)
        return Station.objects.create(**defaults)

    def create_report(self, station, **overrides):
        defaults = {
            "station": station,
            "text": "Lift unavailable",
            "information": False,
            "start_time": timezone.now() - timedelta(hours=1),
            "end_time": None,
            "resolved": False,
            "source": Report.SOURCE_TFLAPI_V1,
        }
        defaults.update(overrides)
        return Report.objects.create(**defaults)

    def create_incident(self, station, reports=None, **overrides):
        defaults = {
            "station": station,
            "text": "Lift unavailable",
            "information": False,
            "start_time": timezone.now() - timedelta(hours=1),
            "end_time": None,
            "resolved": False,
        }
        defaults.update(overrides)
        incident = Incident.objects.create(**defaults)
        for report in reports or []:
            incident.reports.add(report)
        return incident


class UtilityTests(SimpleTestCase):
    def test_prediction_confidence_labels_are_conservative(self):
        self.assertIsNone(_prediction_confidence_label(None))
        self.assertEqual(_prediction_confidence_label(0.55), "maybe")
        self.assertEqual(_prediction_confidence_label(0.69), "maybe")
        self.assertEqual(_prediction_confidence_label(0.70), "likely")
        self.assertEqual(_prediction_confidence_label(0.84), "likely")
        self.assertEqual(_prediction_confidence_label(0.85), "very likely")

    def test_prediction_outcome_is_one_sided_fixed_by_bound(self):
        # We promised it would be fixed *by* 14:00.
        predicted_end = datetime(2026, 1, 1, 14, 0, tzinfo=dt_timezone.utc)

        # Resolved early — the "fixed by" promise was kept. Full success.
        self.assertEqual(
            prediction_outcome(
                predicted_end,
                datetime(2026, 1, 1, 9, 0, tzinfo=dt_timezone.utc),
            ),
            "exact",
        )
        # Resolved exactly on the promised time — also a full success.
        self.assertEqual(
            prediction_outcome(predicted_end, predicted_end),
            "exact",
        )
        # Overran, but only within the grace window — a soft miss.
        self.assertEqual(
            prediction_outcome(
                predicted_end,
                datetime(2026, 1, 1, 14, 20, tzinfo=dt_timezone.utc),
            ),
            "near",
        )
        # Overran well beyond grace — the promise broke.
        self.assertEqual(
            prediction_outcome(
                predicted_end,
                datetime(2026, 1, 1, 15, 1, tzinfo=dt_timezone.utc),
            ),
            "miss",
        )

    def test_remove_tfl_specifics_and_grammar_cleanup(self):
        cleaned = remove_tfl_specifics(
            "Call our Travel Information Centre on 0343 222 1234 for further help. "
            "we have asked a member of staff to help."
        )

        self.assertNotIn("0343 222 1234", cleaned)
        self.assertIn("TfL have asked a member of TfL staff", cleaned)
        self.assertEqual(
            fix_additional_info_grammar(
                "Use the lift.Please allow extra time for your journey.<b>Thanks</b>"
            ),
            "Use the lift. Please allow extra time for your journey. <b>Thanks</b>",
        )

    def test_parse_date_and_find_dates_cover_relative_and_multi_day_formats(self):
        with patch(
            "incidents.utils.arrow.utcnow", return_value=arrow.get("2026-03-26")
        ):
            early_march = parse_date("early March")
            start_date, end_date = find_dates(
                "Tuesday 21, Wednesday 22 and Thursday 23 October"
            )
            rollover_start, rollover_end = find_dates(
                "From 10 December until March, due to station works"
            )

        self.assertEqual(
            (early_march.year, early_march.month, early_march.day), (2026, 3, 10)
        )
        self.assertEqual((start_date.month, start_date.day), (10, 21))
        self.assertEqual((end_date.month, end_date.day), (10, 23))
        self.assertEqual((rollover_start.year, rollover_start.month), (2025, 12))
        self.assertEqual((rollover_end.year, rollover_end.month), (2026, 3))
        self.assertEqual(find_dates("Some random text"), (None, None))

    @override_settings(DEBUG=True)
    def test_send_helpers_print_in_debug(self):
        with patch("builtins.print") as mock_print:
            send_tweet("Bank update")
            send_bluesky("Bank update")

        mock_print.assert_any_call("Should have tweeted: Bank update")
        mock_print.assert_any_call("Should have posted on blue sky: Bank update")

    @override_settings(
        DEBUG=False,
        TWITTER_API_KEY="key",
        TWITTER_API_SECRET="secret",
        TWITTER_ACCESS_TOKEN="token",
        TWITTER_ACCESS_TOKEN_SECRET="token-secret",
    )
    def test_send_tweet_uses_tweepy_client_outside_debug(self):
        client = MagicMock()

        with patch("incidents.utils.tweepy.Client", return_value=client) as mock_client:
            send_tweet("Bank update")

        mock_client.assert_called_once_with(
            consumer_key="key",
            consumer_secret="secret",
            access_token="token",
            access_token_secret="token-secret",
        )
        client.create_tweet.assert_called_once_with(text="Bank update")

    @override_settings(
        DEBUG=False,
        BLUESKY_USER_NAME="user",
        BLUESKY_PASSWORD="password",
    )
    def test_send_bluesky_uses_atproto_client_outside_debug(self):
        client = MagicMock()
        profile = MagicMock(display_name="Up Down London")
        client.login.return_value = profile
        builder = MagicMock()
        builder.text.return_value = "formatted post"

        with patch("incidents.utils.atproto.Client", return_value=client), patch(
            "incidents.utils.atproto.client_utils.TextBuilder", return_value=builder
        ), patch("builtins.print"):
            send_bluesky("Bank update")

        client.login.assert_called_once_with("user", "password")
        client.send_post.assert_called_once_with("formatted post")

    def test_last_updated_round_trips_to_disk(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "incidents.utils.timezone.now",
            return_value=timezone.make_aware(timezone.datetime(2026, 3, 26, 9, 15)),
        ):
            current_dir = os.getcwd()
            try:
                os.chdir(temp_dir)
                update_last_updated()
                self.assertEqual(get_last_updated(), "09:15 26 Mar")
            finally:
                os.chdir(current_dir)


class SourceTests(StationFactoryMixin, TestCase):
    def test_tflapiv1_creates_current_planned_work_report(self):
        station = self.create_child_station(self.create_parent_station("Bank"))
        payload = [
            {
                "description": "Bank: No Step Free Access - From 1 March until 30 March, due to planned maintenance. Call us on 0343 222 1234 for further help.",
                "atcoCode": station.naptan_id,
                "appearance": "PlannedWork",
                "additionalInformation": "Use the side entrance.Please allow extra time for your journey.",
            }
        ]

        with patch.object(
            tflapiv1.requests, "get", return_value=FakeResponse(payload)
        ), patch("incidents.utils.arrow.utcnow", return_value=arrow.get("2026-03-26")):
            tflapiv1.check()

        report = Report.objects.get(source=Report.SOURCE_TFLAPI_V1)
        self.assertFalse(report.information)
        self.assertEqual((report.start_time.month, report.start_time.day), (3, 1))
        self.assertEqual(
            (
                timezone.localtime(report.end_time).month,
                timezone.localtime(report.end_time).day,
            ),
            (3, 30),
        )
        self.assertNotIn("0343 222 1234", report.text)
        self.assertIn("Please allow extra time for your journey.", report.text)

    def test_tflapiv1_falls_back_to_now_when_no_dates_are_present(self):
        station = self.create_child_station(self.create_parent_station("Waterloo"))
        payload = [
            {
                "description": "Waterloo: There will be no step free access because the lift is unavailable",
                "atcoCode": station.naptan_id,
                "appearance": "PlannedWork",
                "additionalInformation": "",
            }
        ]

        with patch.object(tflapiv1.requests, "get", return_value=FakeResponse(payload)):
            before = timezone.now()
            tflapiv1.check()
            after = timezone.now()

        report = Report.objects.get(source=Report.SOURCE_TFLAPI_V1)
        self.assertGreaterEqual(report.start_time, before)
        self.assertLessEqual(report.start_time, after)

    def test_tflapiv1_resolves_cleared_reports_and_skips_unknown_stations(self):
        station = self.create_child_station(
            self.create_parent_station("Liverpool Street")
        )
        existing = self.create_report(
            station,
            source=Report.SOURCE_TFLAPI_V1,
            text="Existing outage",
        )
        payload = [
            {
                "description": "Unknown: No Step Free Access - Lift unavailable",
                "atcoCode": "UNKNOWN",
                "appearance": "RealTime",
                "additionalInformation": "",
            }
        ]

        with patch.object(tflapiv1.requests, "get", return_value=FakeResponse(payload)):
            tflapiv1.check()

        existing.refresh_from_db()
        self.assertTrue(existing.resolved)
        self.assertIsNotNone(existing.end_time)
        self.assertEqual(Report.objects.count(), 1)

    def test_tflapiv1_does_not_duplicate_matching_existing_report(self):
        station = self.create_child_station(self.create_parent_station("Holborn"))
        existing = self.create_report(
            station,
            source=Report.SOURCE_TFLAPI_V1,
            text="Lift unavailable",
        )
        payload = [
            {
                "description": "Holborn: No Step Free Access - Lift unavailable",
                "atcoCode": station.naptan_id,
                "appearance": "RealTime",
                "additionalInformation": "",
            }
        ]

        with patch.object(tflapiv1.requests, "get", return_value=FakeResponse(payload)):
            tflapiv1.check()

        existing.refresh_from_db()
        self.assertEqual(
            Report.objects.filter(source=Report.SOURCE_TFLAPI_V1).count(), 1
        )
        self.assertFalse(existing.resolved)
        self.assertIsNone(existing.end_time)

    def test_tflapiv1_connection_error_leaves_existing_reports_untouched(self):
        station = self.create_child_station(self.create_parent_station("Paddington"))
        existing = self.create_report(
            station,
            source=Report.SOURCE_TFLAPI_V1,
            text="Lift unavailable",
        )

        with patch.object(
            tflapiv1.requests,
            "get",
            side_effect=tflapiv1.requests.exceptions.ConnectionError,
        ):
            tflapiv1.check()

        existing.refresh_from_db()
        self.assertFalse(existing.resolved)
        self.assertIsNone(existing.end_time)

    def test_tflapiv2_creates_reports_and_resolves_cleared_entries(self):
        station = self.create_child_station(self.create_parent_station("Euston"))
        stale_report = self.create_report(
            station,
            source=Report.SOURCE_TFLAPI_V2,
            text="Old outage",
        )
        payload = [
            {
                "stationUniqueId": station.naptan_id,
                "message": "Euston: No Step Free Access - Lift unavailable",
            }
        ]

        with patch.object(tflapiv2.requests, "get", return_value=FakeResponse(payload)):
            tflapiv2.check()

        stale_report.refresh_from_db()
        new_report = Report.objects.get(
            source=Report.SOURCE_TFLAPI_V2, text="Lift unavailable"
        )
        self.assertTrue(stale_report.resolved)
        self.assertEqual(new_report.station, station)
        self.assertFalse(new_report.resolved)

    def test_tflapiv2_does_not_duplicate_matching_existing_report(self):
        station = self.create_child_station(self.create_parent_station("Embankment"))
        existing = self.create_report(
            station,
            source=Report.SOURCE_TFLAPI_V2,
            text="Lift unavailable",
        )
        payload = [
            {
                "stationUniqueId": station.naptan_id,
                "message": "Embankment: No Step Free Access - Lift unavailable",
            }
        ]

        with patch.object(tflapiv2.requests, "get", return_value=FakeResponse(payload)):
            tflapiv2.check()

        existing.refresh_from_db()
        self.assertEqual(
            Report.objects.filter(source=Report.SOURCE_TFLAPI_V2).count(), 1
        )
        self.assertFalse(existing.resolved)
        self.assertIsNone(existing.end_time)

    def test_tflapiv2_connection_error_leaves_existing_reports_untouched(self):
        station = self.create_child_station(self.create_parent_station("Angel"))
        existing = self.create_report(
            station,
            source=Report.SOURCE_TFLAPI_V2,
            text="Lift unavailable",
        )

        with patch.object(
            tflapiv2.requests,
            "get",
            side_effect=tflapiv2.requests.exceptions.ConnectionError,
        ):
            tflapiv2.check()

        existing.refresh_from_db()
        self.assertFalse(existing.resolved)
        self.assertIsNone(existing.end_time)

    def test_tflapiv2_skips_unknown_stations(self):
        payload = [
            {
                "stationUniqueId": "UNKNOWN",
                "message": "Mystery: No Step Free Access - Lift unavailable",
            }
        ]

        with patch.object(tflapiv2.requests, "get", return_value=FakeResponse(payload)):
            tflapiv2.check()

        self.assertEqual(
            Report.objects.filter(source=Report.SOURCE_TFLAPI_V2).count(), 0
        )


class ConsolidationTests(StationFactoryMixin, TestCase):
    def test_model_string_representations_include_key_state(self):
        station = self.create_parent_station("Baker Street")
        report = self.create_report(station, source=Report.SOURCE_USER)
        incident = self.create_incident(station, reports=[report])

        self.assertIn("Baker Street", str(report))
        self.assertIn("User", str(report))
        self.assertIn("1 reports", str(incident))

    def test_consolidate_creates_incident_and_announces_user_report(self):
        parent = self.create_parent_station("Bank")
        child = self.create_child_station(parent)
        report = self.create_report(child, source=Report.SOURCE_USER)

        with patch(
            "incidents.management.commands.update_incidents.send_tweet"
        ) as send_tweet_mock, patch(
            "incidents.management.commands.update_incidents.send_bluesky"
        ) as send_bluesky_mock:
            consolidate_incidents()

        incident = Incident.objects.get()
        self.assertEqual(incident.station, parent)
        self.assertIn(report, incident.reports.all())
        send_tweet_mock.assert_called_once()
        send_bluesky_mock.assert_called_once()
        self.assertIn("This is a user report", send_tweet_mock.call_args.args[0])

    def test_consolidate_merges_similar_reports_into_existing_incident(self):
        parent = self.create_parent_station("Waterloo")
        first_report = self.create_report(
            parent,
            text="Lift unavailable on the northbound platform",
            start_time=timezone.now() - timedelta(hours=4),
        )
        incident = self.create_incident(
            parent,
            reports=[first_report],
            text=first_report.text,
            start_time=first_report.start_time,
        )
        second_report = self.create_report(
            parent,
            text="Lift unavailable on the northbound platform.",
            source=Report.SOURCE_TFLAPI_V2,
            start_time=timezone.now() - timedelta(hours=1),
        )

        with patch(
            "incidents.management.commands.update_incidents.send_tweet"
        ) as send_tweet_mock, patch(
            "incidents.management.commands.update_incidents.send_bluesky"
        ) as send_bluesky_mock:
            consolidate_incidents()

        incident.refresh_from_db()
        self.assertEqual(Incident.objects.count(), 1)
        self.assertEqual(incident.reports.count(), 2)
        self.assertEqual(incident.start_time, first_report.start_time)
        self.assertEqual(incident.text, first_report.text)
        send_tweet_mock.assert_not_called()
        send_bluesky_mock.assert_not_called()

    def test_consolidate_resolves_expired_single_user_reports_silently(self):
        parent = self.create_parent_station("Oxford Circus")
        report = self.create_report(
            parent,
            source=Report.SOURCE_USER,
            end_time=timezone.now() - timedelta(minutes=5),
        )
        incident = self.create_incident(parent, reports=[report])

        with patch(
            "incidents.management.commands.update_incidents.send_tweet"
        ) as send_tweet_mock, patch(
            "incidents.management.commands.update_incidents.send_bluesky"
        ) as send_bluesky_mock:
            consolidate_incidents()

        report.refresh_from_db()
        incident.refresh_from_db()
        self.assertTrue(report.resolved)
        self.assertTrue(incident.resolved)
        send_tweet_mock.assert_not_called()
        send_bluesky_mock.assert_not_called()

    def test_consolidate_announces_restoration_for_non_user_incidents(self):
        parent = self.create_parent_station("Victoria")
        report = self.create_report(
            parent,
            source=Report.SOURCE_TFLAPI_V1,
            resolved=True,
            end_time=timezone.now() - timedelta(minutes=1),
        )
        incident = self.create_incident(parent, reports=[report])

        with patch(
            "incidents.management.commands.update_incidents.send_tweet"
        ) as send_tweet_mock, patch(
            "incidents.management.commands.update_incidents.send_bluesky"
        ) as send_bluesky_mock:
            consolidate_incidents()

        incident.refresh_from_db()
        self.assertTrue(incident.resolved)
        self.assertIsNotNone(incident.end_time)
        send_tweet_mock.assert_called_once_with(
            "Step free access has been restored at Victoria"
        )
        send_bluesky_mock.assert_called_once_with(
            "Step free access has been restored at Victoria"
        )

    def test_consolidate_refreshes_prediction_after_attaching_report(self):
        parent = self.create_parent_station("Green Park")
        first_report = self.create_report(
            parent,
            text="Lift unavailable on the northbound platform",
            start_time=timezone.now() - timedelta(hours=4),
        )
        incident = self.create_incident(
            parent,
            reports=[first_report],
            text=first_report.text,
            start_time=first_report.start_time,
            estimated_duration=timedelta(minutes=30),
            prediction_confidence=0.1,
        )
        self.create_report(
            parent,
            text="Lift unavailable on the northbound platform.",
            source=Report.SOURCE_TFLAPI_V2,
            start_time=timezone.now() - timedelta(hours=1),
        )

        observed = {}

        def fake_predict(updated_incident):
            observed["reports_count"] = updated_incident.reports.count()
            return timedelta(hours=4), 0.8

        with patch(
            "incidents.management.commands.update_incidents.predict_duration",
            side_effect=fake_predict,
        ), patch(
            "incidents.management.commands.update_incidents.send_tweet"
        ) as send_tweet_mock, patch(
            "incidents.management.commands.update_incidents.send_bluesky"
        ) as send_bluesky_mock:
            consolidate_incidents()

        incident.refresh_from_db()
        self.assertEqual(observed["reports_count"], 2)
        self.assertEqual(incident.estimated_duration, timedelta(hours=4))
        self.assertEqual(incident.prediction_confidence, 0.8)
        send_tweet_mock.assert_not_called()
        send_bluesky_mock.assert_not_called()


class PredictionFeatureTests(StationFactoryMixin, TestCase):
    def test_predict_duration_uses_local_time_and_prior_station_history(self):
        if np is None:
            self.skipTest("numpy not installed")

        parent = self.create_parent_station("Canary Wharf")
        self.create_incident(
            parent,
            text="Earlier outage",
            resolved=True,
            start_time=datetime(2026, 7, 1, 7, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 1, 8, 0, tzinfo=dt_timezone.utc),
        )
        incident = Incident(
            station=parent,
            text="Faulty lift. Call 0343 222 1234 for help.",
            information=False,
            start_time=datetime(2026, 7, 1, 9, 30, tzinfo=dt_timezone.utc),
            resolved=False,
        )

        captured = {}

        class FakeModel:
            classes_ = np.array([0])
            feature_names_in_ = np.array(
                [
                    "station_id",
                    "information",
                    "hour_of_day",
                    "day_of_week",
                    "month",
                    "is_weekend",
                    "start_block",
                    "has_faulty_lift",
                    "has_planned_maintenance",
                    "has_staff_issue",
                    "tube",
                    "dlr",
                    "national_rail",
                    "crossrail",
                    "overground",
                    "access_via_lift",
                    "num_reports",
                    "days_since_last_incident",
                    "concurrent_incidents",
                    "station_mean_duration",
                    "station_incident_count",
                    "station_mean_offset",
                ],
                dtype=object,
            )

            def predict_proba(self, df):
                captured["features"] = df.iloc[0].to_dict()
                return np.array([[1.0]])

        with patch(
            "incidents.ml._load_model",
            return_value={"model": FakeModel(), "metadata": {"feature_version": 2}},
        ):
            duration, confidence = predict_duration(incident)

        # The model is certain (P=1.0 on offset 0), so the one-sided bound
        # is the end of the start block (daytime, ends 15:00 BST), landing
        # one second inside it. Start is 10:30 BST → 14:59:59 BST.
        self.assertEqual(duration, timedelta(hours=4, minutes=29, seconds=59))
        self.assertEqual(confidence, 0.95)
        self.assertEqual(captured["features"]["hour_of_day"], 10)
        self.assertEqual(captured["features"]["month"], 7)
        self.assertEqual(captured["features"]["start_block"], 1)
        self.assertEqual(captured["features"]["has_faulty_lift"], 1)
        self.assertEqual(captured["features"]["station_incident_count"], 1)
        self.assertEqual(captured["features"]["station_mean_duration"], 60.0)
        self.assertEqual(captured["features"]["station_mean_offset"], 0.0)
        self.assertEqual(captured["features"]["num_reports"], 1)
        self.assertEqual(captured["features"]["concurrent_incidents"], 0)

    def test_predict_duration_uses_annotated_num_reports(self):
        if np is None:
            self.skipTest("numpy not installed")

        parent = self.create_parent_station("Canary Wharf")
        report_one = self.create_report(parent, text="Lift unavailable")
        report_two = self.create_report(parent, text="Second report")
        incident = self.create_incident(
            parent,
            text="Faulty lift",
            reports=[report_one, report_two],
        )
        incident = Incident.objects.annotate(
            num_reports=Count("reports", distinct=True)
        ).get(pk=incident.pk)

        captured = {}

        class FakeModel:
            classes_ = np.array([0])
            feature_names_in_ = np.array(
                [
                    "station_id",
                    "information",
                    "hour_of_day",
                    "day_of_week",
                    "month",
                    "is_weekend",
                    "start_block",
                    "has_faulty_lift",
                    "has_planned_maintenance",
                    "has_staff_issue",
                    "tube",
                    "dlr",
                    "national_rail",
                    "crossrail",
                    "overground",
                    "access_via_lift",
                    "num_reports",
                    "days_since_last_incident",
                    "concurrent_incidents",
                    "station_mean_duration",
                    "station_incident_count",
                    "station_mean_offset",
                ],
                dtype=object,
            )

            def predict_proba(self, df):
                captured["features"] = df.iloc[0].to_dict()
                return np.array([[1.0]])

        with patch(
            "incidents.ml._load_model",
            return_value={"model": FakeModel(), "metadata": {"feature_version": 2}},
        ):
            predict_duration(incident)

        self.assertEqual(captured["features"]["num_reports"], 2)

    def test_predict_duration_blends_in_historical_station_baseline(self):
        if np is None:
            self.skipTest("numpy not installed")

        parent = self.create_parent_station("Roding Valley")
        incident = Incident(
            station=parent,
            text="Faulty lift",
            information=False,
            start_time=datetime(2026, 7, 1, 9, 30, tzinfo=dt_timezone.utc),
            resolved=False,
        )

        class FakeModel:
            classes_ = np.array([0, 1])
            feature_names_in_ = np.array(
                [
                    "station_id",
                    "information",
                    "hour_of_day",
                    "day_of_week",
                    "month",
                    "is_weekend",
                    "start_block",
                    "has_faulty_lift",
                    "has_planned_maintenance",
                    "has_staff_issue",
                    "tube",
                    "dlr",
                    "national_rail",
                    "crossrail",
                    "overground",
                    "access_via_lift",
                    "num_reports",
                    "days_since_last_incident",
                    "concurrent_incidents",
                    "station_mean_duration",
                    "station_incident_count",
                    "station_mean_offset",
                ],
                dtype=object,
            )

            def predict_proba(self, df):
                return np.array([[0.45, 0.55]])

        empty_tail = [0] * 19
        historical_baselines = {
            "global": {"count": 20, "counts": [10, 10, *empty_tail]},
            "category": {
                "faulty_lift": {"count": 10, "counts": [9, 1, *empty_tail]}
            },
            "network_category": {},
            "station": {
                parent.id: {"count": 20, "counts": [20, 0, *empty_tail]}
            },
            "station_category": {
                (parent.id, "faulty_lift"): {
                    "count": 12,
                    "counts": [12, 0, *empty_tail],
                }
            },
        }

        with patch(
            "incidents.ml._load_model",
            return_value={
                "model": FakeModel(),
                "metadata": {"feature_version": 3},
                "historical_baselines": historical_baselines,
            },
        ):
            duration, confidence = predict_duration(incident)

        # Blended probs are [0.578, 0.422] over offsets [0, 1]. The 75%
        # cumulative bound falls on offset 1, so the prediction is "fixed
        # by" the end of the next (evening) block: 20:00 BST minus a second,
        # from a 10:30 BST start. Confidence is the cumulative mass (~1.0,
        # clamped to 0.95).
        self.assertEqual(duration, timedelta(hours=9, minutes=29, seconds=59))
        self.assertEqual(confidence, 0.95)


@override_settings(ROOT_URLCONF="updown.urls")
class ViewAndCommandTests(StationFactoryMixin, TestCase):
    def test_predict_durations_command_can_resume_in_small_batches(self):
        parent = self.create_parent_station("Bank")
        first = self.create_incident(parent, text="First outage", resolved=False)
        second = self.create_incident(parent, text="Second outage", resolved=False)
        stdout = StringIO()

        with patch(
            "incidents.management.commands.predict_durations._load_model",
            return_value=object(),
        ), patch(
            "incidents.management.commands.predict_durations.predict_duration",
            return_value=(timedelta(minutes=30), 0.7),
        ):
            call_command(
                "predict_durations",
                "--overwrite",
                "--after-id",
                str(first.id),
                "--limit",
                "1",
                "--chunk-size",
                "1",
                stdout=stdout,
            )

        first.refresh_from_db()
        second.refresh_from_db()
        self.assertIsNone(first.estimated_duration)
        self.assertEqual(second.estimated_duration, timedelta(minutes=30))
        self.assertEqual(second.prediction_confidence, 0.7)
        self.assertIn(f"Last processed incident id: {second.id}", stdout.getvalue())

    def test_prepare_incidents_only_shows_predictions_from_stronger_buckets(self):
        parent = self.create_parent_station("Waterloo")
        base = datetime(2026, 7, 1, 7, 0, tzinfo=dt_timezone.utc)

        # One-sided scoring: a prediction is correct when the incident was
        # resolved at or before the predicted "fixed by" time. "Wrong"
        # therefore means it resolved *later* than we promised.
        #
        # Bucket 60 (conf 0.65): 20 predictions, 16 correct → 80% accuracy,
        # which clears the display gate and lands in the "likely" band.
        for i in range(16):
            self.create_incident(
                parent,
                text=f"Resolved good {i}",
                resolved=True,
                start_time=base + timedelta(days=i),
                end_time=base + timedelta(days=i, hours=1),
                estimated_duration=timedelta(hours=2),
                prediction_confidence=0.65,
            )
        for i in range(4):
            self.create_incident(
                parent,
                text=f"Resolved good late {i}",
                resolved=True,
                start_time=base + timedelta(days=20 + i),
                end_time=base + timedelta(days=20 + i, hours=12),
                estimated_duration=timedelta(hours=1),
                prediction_confidence=0.65,
            )
        # Bucket 40 (conf 0.45): 15 predictions, 7 correct → ~47% accuracy,
        # well below the gate, so it stays hidden.
        for i in range(7):
            self.create_incident(
                parent,
                text=f"Resolved low ok {i}",
                resolved=True,
                start_time=base + timedelta(days=40 + i),
                end_time=base + timedelta(days=40 + i, hours=1),
                estimated_duration=timedelta(hours=2),
                prediction_confidence=0.45,
            )
        for i in range(8):
            self.create_incident(
                parent,
                text=f"Resolved low late {i}",
                resolved=True,
                start_time=base + timedelta(days=60 + i),
                end_time=base + timedelta(days=60 + i, hours=12),
                estimated_duration=timedelta(hours=1),
                prediction_confidence=0.45,
            )

        low = self.create_incident(
            parent,
            text="Low confidence outage",
            estimated_duration=timedelta(hours=2),
            prediction_confidence=0.45,
        )
        high = self.create_incident(
            parent,
            text="High confidence outage",
            estimated_duration=timedelta(hours=2),
            prediction_confidence=0.65,
        )
        policy = _prediction_bucket_metrics(
            timezone.now() - timedelta(days=365)
        )["buckets"]

        incidents = _prepare_incidents(
            Incident.objects.filter(id__in=[low.id, high.id]).order_by("id"),
            prediction_policy=policy,
        )

        self.assertFalse(incidents[0].show_prediction)
        self.assertIsNone(incidents[0].prediction_label)
        self.assertTrue(incidents[1].show_prediction)
        self.assertEqual(incidents[1].prediction_label, "likely")
        self.assertEqual(incidents[1].prediction_accuracy_pct, 80)

    def test_prepare_incidents_allows_40_bucket_for_station_with_strong_history(self):
        weak_station = self.create_parent_station("Camden Road", overground=True)
        strong_station = self.create_parent_station("Roding Valley")
        now = timezone.now()

        for i in range(15):
            start_time = now - timedelta(days=30 + i, hours=2)
            self.create_incident(
                weak_station,
                text=f"Weak bucket miss {i}",
                resolved=True,
                start_time=start_time,
                end_time=start_time + timedelta(days=1, hours=1),
                estimated_duration=timedelta(hours=1),
                prediction_confidence=0.45,
            )

        for i in range(20):
            start_time = now - timedelta(days=60 + i, hours=2)
            self.create_incident(
                strong_station,
                text=f"Strong history good {i}",
                resolved=True,
                start_time=start_time,
                end_time=start_time + timedelta(hours=1),
                estimated_duration=timedelta(hours=1),
                prediction_confidence=0.65,
            )

        current = self.create_incident(
            strong_station,
            text="Current moderate confidence outage",
            resolved=False,
            start_time=now - timedelta(hours=1),
            estimated_duration=timedelta(hours=2),
            prediction_confidence=0.45,
        )

        policy = _prediction_display_policy(now)
        overrides = _prediction_station_overrides(
            Incident.objects.filter(id=current.id).select_related(
                "station", "station__parent_station"
            ),
            policy,
            now,
        )

        incidents = _prepare_incidents(
            Incident.objects.filter(id=current.id).select_related(
                "station", "station__parent_station"
            ),
            prediction_policy=policy,
            station_prediction_overrides=overrides,
            now=now,
        )

        self.assertFalse(policy[40]["show_prediction"])
        self.assertTrue(incidents[0].show_prediction)
        self.assertTrue(incidents[0].used_station_prediction_override)
        self.assertEqual(incidents[0].prediction_label, "maybe")
        self.assertEqual(incidents[0].prediction_accuracy_pct, 100)

    def test_prediction_display_policy_falls_back_when_recent_bucket_is_sparse(self):
        parent = self.create_parent_station("Waterloo")
        now = timezone.now()

        for i in range(10):
            start_time = now - timedelta(days=10 + i, hours=2)
            self.create_incident(
                parent,
                text=f"Recent good {i}",
                resolved=True,
                start_time=start_time,
                end_time=start_time + timedelta(hours=1),
                estimated_duration=timedelta(hours=1),
                prediction_confidence=0.65,
            )
        for i in range(5):
            start_time = now - timedelta(days=40 + i, hours=2)
            self.create_incident(
                parent,
                text=f"Fallback good {i}",
                resolved=True,
                start_time=start_time,
                end_time=start_time + timedelta(hours=1),
                estimated_duration=timedelta(hours=1),
                prediction_confidence=0.65,
            )

        policy = _prediction_display_policy(now)

        self.assertTrue(policy[60]["show_prediction"])
        self.assertEqual(policy[60]["window_days"], 90)
        self.assertTrue(policy[60]["used_fallback_window"])

    def test_prediction_display_policy_keeps_recent_bad_bucket_hidden(self):
        parent = self.create_parent_station("Waterloo")
        now = timezone.now()

        for i in range(15):
            start_time = now - timedelta(days=10 + i, hours=2)
            # One-sided miss: we said "fixed by" an hour but it actually
            # took twelve, so the promise broke.
            self.create_incident(
                parent,
                text=f"Recent miss {i}",
                resolved=True,
                start_time=start_time,
                end_time=start_time + timedelta(hours=12),
                estimated_duration=timedelta(hours=1),
                prediction_confidence=0.65,
            )
        for i in range(20):
            start_time = now - timedelta(days=60 + i, hours=2)
            self.create_incident(
                parent,
                text=f"Older good {i}",
                resolved=True,
                start_time=start_time,
                end_time=start_time + timedelta(hours=1),
                estimated_duration=timedelta(hours=1),
                prediction_confidence=0.65,
            )

        policy = _prediction_display_policy(now)

        self.assertFalse(policy[60]["show_prediction"])
        self.assertEqual(policy[60]["window_days"], 30)
        self.assertFalse(policy[60]["used_fallback_window"])

    def test_prepare_incidents_hides_overdue_current_prediction(self):
        parent = self.create_parent_station("Bank")
        base = datetime(2026, 7, 1, 7, 0, tzinfo=dt_timezone.utc)
        for i in range(15):
            self.create_incident(
                parent,
                text=f"Resolved good {i}",
                resolved=True,
                start_time=base + timedelta(days=i),
                end_time=base + timedelta(days=i, hours=1),
                estimated_duration=timedelta(hours=1),
                prediction_confidence=0.65,
            )

        overdue = self.create_incident(
            parent,
            text="Overdue outage",
            resolved=False,
            start_time=timezone.now() - timedelta(days=2),
            estimated_duration=timedelta(hours=2),
            prediction_confidence=0.65,
        )
        policy = _prediction_bucket_metrics(
            timezone.now() - timedelta(days=365)
        )["buckets"]

        incidents = _prepare_incidents(
            Incident.objects.filter(id=overdue.id),
            prediction_policy=policy,
        )

        self.assertFalse(incidents[0].show_prediction)
        self.assertIsNone(incidents[0].beta_current_status)

    def test_prepare_incidents_uses_more_than_a_week_for_long_range_prediction(self):
        parent = self.create_parent_station("Bank")
        base = datetime(2026, 7, 1, 7, 0, tzinfo=dt_timezone.utc)
        for i in range(15):
            self.create_incident(
                parent,
                text=f"Resolved good {i}",
                resolved=True,
                start_time=base + timedelta(days=i),
                end_time=base + timedelta(days=i, hours=1),
                estimated_duration=timedelta(hours=1),
                prediction_confidence=0.65,
            )

        long_range = self.create_incident(
            parent,
            text="Long range outage",
            resolved=False,
            start_time=timezone.now(),
            estimated_duration=timedelta(days=8),
            prediction_confidence=0.65,
        )
        policy = _prediction_bucket_metrics(
            timezone.now() - timedelta(days=365)
        )["buckets"]

        incidents = _prepare_incidents(
            Incident.objects.filter(id=long_range.id),
            prediction_policy=policy,
        )

        self.assertTrue(incidents[0].show_prediction)
        self.assertEqual(incidents[0].beta_current_status, "More than a week")

    def test_prepare_incidents_marks_resolved_miss_as_wrong(self):
        parent = self.create_parent_station("Bank")
        base = datetime(2026, 7, 1, 7, 0, tzinfo=dt_timezone.utc)
        for i in range(15):
            self.create_incident(
                parent,
                text=f"Resolved good {i}",
                resolved=True,
                start_time=base + timedelta(days=i),
                end_time=base + timedelta(days=i, hours=1),
                estimated_duration=timedelta(hours=1),
                prediction_confidence=0.65,
            )

        missed = self.create_incident(
            parent,
            text="Missed outage",
            resolved=True,
            start_time=timezone.now() - timedelta(days=2),
            end_time=timezone.now() - timedelta(days=1),
            estimated_duration=timedelta(hours=2),
            prediction_confidence=0.65,
        )
        policy = _prediction_bucket_metrics(
            timezone.now() - timedelta(days=365)
        )["buckets"]

        incidents = _prepare_incidents(
            Incident.objects.filter(id=missed.id),
            prediction_policy=policy,
        )

        self.assertTrue(incidents[0].show_prediction)
        self.assertEqual(incidents[0].prediction_outcome, "miss")
        self.assertEqual(incidents[0].beta_resolved_status, "Wrong")

    def test_detail_renders_home_lists(self):
        parent = self.create_parent_station("Bank")
        active = self.create_incident(parent, text="Active outage", resolved=False)
        resolved = self.create_incident(
            parent,
            text="Resolved outage",
            resolved=True,
            end_time=timezone.now() - timedelta(hours=1),
        )
        information = self.create_incident(
            parent,
            text="Planned works",
            information=True,
        )

        with patch("incidents.views.get_last_updated", return_value="09:15 26 Mar"):
            response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "home.html")
        self.assertEqual(list(response.context["issues"]), [active])
        self.assertEqual(list(response.context["resolved"]), [resolved])
        self.assertEqual(list(response.context["information"]), [information])
        self.assertEqual(response.context["last_updated"], "09:15 26 Mar")

    def test_beta_renders_inline_status_next_to_event_time(self):
        parent = self.create_parent_station("Bank")
        base = datetime(2026, 7, 1, 7, 0, tzinfo=dt_timezone.utc)
        for i in range(15):
            self.create_incident(
                parent,
                text=f"Resolved good {i}",
                resolved=True,
                start_time=base + timedelta(days=i),
                end_time=base + timedelta(days=i, hours=1),
                estimated_duration=timedelta(hours=1),
                prediction_confidence=0.65,
            )

        active = self.create_incident(
            parent,
            text="Active outage",
            resolved=False,
            start_time=timezone.now() - timedelta(hours=1),
            estimated_duration=timedelta(hours=2),
            prediction_confidence=0.65,
        )
        resolved = self.create_incident(
            parent,
            text="Resolved outage",
            resolved=True,
            end_time=timezone.now() - timedelta(hours=1),
        )
        information = self.create_incident(
            parent,
            text="Planned works",
            information=True,
        )

        with patch("incidents.views.get_last_updated", return_value="09:15 26 Mar"):
            response = self.client.get("/beta/")

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "beta.html")
        self.assertEqual(list(response.context["issues"]), [active])
        self.assertIn(resolved, list(response.context["resolved"]))
        self.assertEqual(list(response.context["information"]), [information])
        self.assertNotContains(response, "Current Status")
        self.assertContains(response, "Current Issues")
        self.assertContains(response, "Resolved Issues")
        self.assertContains(response, "Information")
        self.assertContains(response, "beta-inline-status-current")
        self.assertContains(response, "icon-magic")
        self.assertContains(response, "beta-inline-meter")
        self.assertContains(response, "The meter shows how certain we feel")

    def test_detail_avoids_n_plus_one_queries_for_reports(self):
        now = timezone.now()
        stations = [
            self.create_parent_station("Bank"),
            self.create_parent_station("Waterloo"),
            self.create_parent_station("Victoria"),
        ]

        for index, station in enumerate(stations):
            issue = self.create_incident(
                station,
                text=f"Active outage {index}",
                resolved=False,
                start_time=now - timedelta(hours=index + 1),
            )
            issue.reports.add(
                self.create_report(
                    station,
                    source=Report.SOURCE_USER,
                    end_time=now + timedelta(hours=1),
                )
            )

        with patch("incidents.views.get_last_updated", return_value="09:15 26 Mar"):
            with CaptureQueriesContext(connection) as queries:
                response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertLessEqual(len(queries), 8)

    def test_detail_switches_to_stp_view_for_special_host(self):
        stp_station = self.create_parent_station(
            "St Pancras", hub_naptan_id="HUBKGX", naptan_id="STP"
        )
        self.create_incident(
            stp_station,
            text="No step free access to the Thameslink due to a faulty lift",
        )

        with patch("incidents.views.get_last_updated", return_value="09:15 26 Mar"):
            response = self.client.get(
                "/", HTTP_HOST="www.isstpthameslinkliftbroken.com"
            )

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "stp.html")
        self.assertTrue(response.context["yes_or_no"])

    def test_api_incidents_returns_station_metadata(self):
        parent = self.create_parent_station("Waterloo", naptan_id="WAT")
        self.create_incident(parent, text="Active outage")
        self.create_incident(
            parent,
            text="Resolved outage",
            resolved=True,
            end_time=timezone.now() - timedelta(hours=1),
        )
        self.create_incident(parent, text="Planned works", information=True)

        response = self.client.get("/api/incidents/")
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["issues"][0]["station_name"], "Waterloo")
        self.assertEqual(payload["issues"][0]["station_naptan"], "WAT")
        self.assertEqual(len(payload["resolved"]), 1)
        self.assertEqual(len(payload["information"]), 1)

    def test_api_training_data_uses_local_time_and_clean_text(self):
        parent = self.create_parent_station("Liverpool Street", naptan_id="LST")
        report_one = self.create_report(
            parent,
            start_time=datetime(2026, 7, 1, 9, 30, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 1, 11, 0, tzinfo=dt_timezone.utc),
            resolved=True,
        )
        report_two = self.create_report(
            parent,
            text="Second report",
            start_time=datetime(2026, 7, 1, 9, 35, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 1, 11, 0, tzinfo=dt_timezone.utc),
            resolved=True,
        )
        self.create_incident(
            parent,
            text="Faulty lift. Call 0343 222 1234 for help.",
            resolved=True,
            start_time=datetime(2026, 7, 1, 9, 30, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 1, 11, 0, tzinfo=dt_timezone.utc),
            reports=[report_one, report_two],
        )
        self.create_incident(
            parent,
            text="Staff helping passengers",
            resolved=True,
            start_time=datetime(2026, 7, 1, 10, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 1, 12, 0, tzinfo=dt_timezone.utc),
        )

        response = self.client.get("/api/training-data/?key=verysecret")
        payload = json.loads(b"".join(response.streaming_content).decode("utf-8"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["incidents"][0]["hour_of_day"], 10)
        self.assertEqual(payload["incidents"][0]["month"], 7)
        self.assertNotIn("0343 222 1234", payload["incidents"][0]["text"])
        self.assertTrue(payload["incidents"][0]["has_faulty_lift"])
        self.assertEqual(payload["incidents"][0]["num_reports"], 2)
        self.assertEqual(payload["incidents"][0]["concurrent_incidents"], 0)
        self.assertEqual(payload["incidents"][1]["concurrent_incidents"], 1)

    def test_detail_and_api_exclude_resolved_incidents_older_than_twelve_hours(self):
        parent = self.create_parent_station("Tottenham Court Road", naptan_id="TCR")
        recent = self.create_incident(
            parent,
            text="Recently resolved outage",
            resolved=True,
            end_time=timezone.now() - timedelta(hours=11, minutes=59),
        )
        self.create_incident(
            parent,
            text="Old resolved outage",
            resolved=True,
            end_time=timezone.now() - timedelta(hours=12, minutes=1),
        )

        detail_response = self.client.get("/")
        api_response = self.client.get("/api/incidents/")

        self.assertEqual(list(detail_response.context["resolved"]), [recent])
        self.assertEqual(len(api_response.json()["resolved"]), 1)
        self.assertEqual(
            api_response.json()["resolved"][0]["text"], "Recently resolved outage"
        )

    def test_api_stations_returns_distinct_parent_station_entries(self):
        parent = self.create_parent_station(
            "Liverpool Street", naptan_id="LST", hub_naptan_id="HUBLST"
        )
        self.create_child_station(
            parent, name="Liverpool Street Underground", tube=True
        )
        self.create_child_station(
            parent, name="Liverpool Street Rail", national_rail=True
        )

        response = self.client.get("/api/stations/")
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            payload["stations"],
            [{"station_name": "Liverpool Street", "station_naptan": "LST"}],
        )

    def test_alexa_formats_station_names(self):
        first = self.create_parent_station("Bank & Monument")
        second = self.create_parent_station("Waterloo")
        self.create_incident(first, text="Active outage")
        self.create_incident(second, text="Active outage")

        response = self.client.get("/problems.txt")

        self.assertEqual(
            response.content.decode(),
            "There are step free access issues at: Bank and Monument and Waterloo",
        )

    def test_alexa_handles_zero_and_single_issue_states(self):
        no_issues_response = self.client.get("/problems.txt")
        self.assertIn(
            "There are currently no reported step free access issues",
            no_issues_response.content.decode(),
        )

        self.create_incident(
            self.create_parent_station("Brixton"), text="Active outage"
        )
        one_issue_response = self.client.get("/problems.txt")

        self.assertEqual(
            one_issue_response.content.decode(),
            "There are step free access issues at: Brixton",
        )

    @override_settings(FUNCTIONS_SECRET_KEY="secret")
    def test_update_incidents_view_requires_correct_secret(self):
        with patch("incidents.views.call_command") as call_command_mock:
            denied = self.client.post("/functions/update_incidents", {"key": "wrong"})
            allowed = self.client.post("/functions/update_incidents", {"key": "secret"})

        self.assertEqual(denied.status_code, 404)
        self.assertEqual(allowed.status_code, 204)
        call_command_mock.assert_called_once_with("update_incidents")

    def test_stats_computes_counts_and_percentages(self):
        parent = self.create_parent_station("Green Park")
        now = timezone.now()
        self.create_incident(
            parent,
            text="unavailability of station staff",
            start_time=now - timedelta(hours=5),
            end_time=now - timedelta(hours=3),
            resolved=True,
        )
        self.create_incident(
            parent,
            text="faulty lift",
            start_time=now - timedelta(hours=3),
            end_time=now - timedelta(hours=2),
            resolved=True,
        )
        self.create_incident(
            parent,
            text="planned maintenance",
            start_time=now - timedelta(hours=2),
            end_time=now - timedelta(hours=1, minutes=30),
            resolved=True,
        )

        with patch("incidents.views.get_last_updated", return_value="09:15 26 Mar"):
            response = self.client.get("/stats/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["total_count"], 3)
        self.assertEqual(response.context["station_staff_count"], 1)
        self.assertEqual(response.context["faulty_lift_count"], 1)
        self.assertEqual(response.context["planned_maintenance_count"], 1)
        self.assertEqual(response.context["station_staff_count_percentage"], 33.33)
        self.assertEqual(response.context["last_updated"], "09:15 26 Mar")

    def test_stats_handles_zero_incidents(self):
        with patch("incidents.views.get_last_updated", return_value="09:15 26 Mar"):
            response = self.client.get("/stats/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["total_count"], 0)
        self.assertEqual(response.context["total_delays"], "0:0:0:0")
        self.assertEqual(response.context["station_staff_count_percentage"], 0)
        self.assertEqual(response.context["faulty_lift_delays_percentage"], 0)
        self.assertEqual(response.context["planned_maintenance_delays_percentage"], 0)
        self.assertEqual(response.context["last_updated"], "09:15 26 Mar")

    def test_stats_includes_recent_prediction_checks_table(self):
        parent = self.create_parent_station("Green Park")
        start_time = datetime(2026, 7, 1, 7, 0, tzinfo=dt_timezone.utc)
        self.create_incident(
            parent,
            text="faulty lift",
            start_time=start_time,
            end_time=start_time + timedelta(hours=1),
            resolved=True,
            estimated_duration=timedelta(hours=1),
            prediction_confidence=0.65,
        )

        with patch("incidents.views.get_last_updated", return_value="09:15 26 Mar"):
            response = self.client.get("/stats/")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Recent Prediction Checks")
        self.assertContains(response, "Green Park")
        self.assertContains(response, "65%")
        rows = response.context["recent_prediction_rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["station_name"], "Green Park")
        self.assertEqual(rows[0]["outcome_text"], "Right")

    def test_stats_includes_current_prediction_visibility_breakdown(self):
        parent = self.create_parent_station("Green Park")
        base = timezone.now() - timedelta(days=10)

        for i in range(15):
            self.create_incident(
                parent,
                text=f"Resolved good {i}",
                resolved=True,
                start_time=base + timedelta(days=i),
                end_time=base + timedelta(days=i, hours=1),
                estimated_duration=timedelta(hours=1),
                prediction_confidence=0.65,
            )
            self.create_incident(
                parent,
                text=f"Resolved weak {i}",
                resolved=True,
                start_time=base + timedelta(days=i, hours=3),
                end_time=base + timedelta(days=i, hours=12),
                estimated_duration=timedelta(hours=1),
                prediction_confidence=0.25,
            )

        self.create_incident(
            parent,
            text="Visible outage",
            resolved=False,
            start_time=timezone.now() - timedelta(hours=1),
            estimated_duration=timedelta(hours=2),
            prediction_confidence=0.65,
        )
        self.create_incident(
            parent,
            text="Past due outage",
            resolved=False,
            start_time=timezone.now() - timedelta(hours=4),
            estimated_duration=timedelta(hours=1),
            prediction_confidence=0.65,
        )
        self.create_incident(
            parent,
            text="Sparse bucket outage",
            resolved=False,
            start_time=timezone.now() - timedelta(hours=1),
            estimated_duration=timedelta(hours=2),
            prediction_confidence=0.45,
        )
        self.create_incident(
            parent,
            text="Low accuracy outage",
            resolved=False,
            start_time=timezone.now() - timedelta(hours=1),
            estimated_duration=timedelta(hours=2),
            prediction_confidence=0.25,
        )
        self.create_incident(
            parent,
            text="Missing prediction outage",
            resolved=False,
        )

        with patch("incidents.views.get_last_updated", return_value="09:15 26 Mar"):
            response = self.client.get("/stats/")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Current Prediction Visibility")
        self.assertContains(response, "Hidden: too few similar examples")
        self.assertContains(
            response, "Hidden: similar predictions not accurate enough"
        )
        self.assertEqual(response.context["current_issue_count"], 5)
        self.assertEqual(response.context["current_prediction_visible_count"], 1)
        self.assertEqual(response.context["current_prediction_hidden_past_due_count"], 1)
        self.assertEqual(
            response.context["current_prediction_hidden_sparse_count"], 1
        )
        self.assertEqual(
            response.context["current_prediction_hidden_low_accuracy_count"], 1
        )
        self.assertEqual(response.context["current_prediction_hidden_missing_count"], 1)

    def test_stats_includes_hidden_low_accuracy_diagnostics(self):
        tube_station = self.create_parent_station("Green Park", tube=True)
        dlr_station = self.create_parent_station(
            "West India Quay",
            tube=False,
            dlr=True,
            national_rail=False,
            crossrail=False,
            overground=False,
        )
        base = timezone.now() - timedelta(days=20)

        for i in range(15):
            self.create_incident(
                tube_station,
                text=f"Resolved weak tube {i}",
                resolved=True,
                start_time=base + timedelta(days=i),
                end_time=base + timedelta(days=i + 1, hours=4),
                estimated_duration=timedelta(hours=1),
                prediction_confidence=0.25,
            )
            self.create_incident(
                dlr_station,
                text=f"Resolved weak dlr {i}",
                resolved=True,
                start_time=base + timedelta(days=i, hours=1),
                end_time=base + timedelta(days=i + 1, hours=5),
                estimated_duration=timedelta(hours=1),
                prediction_confidence=0.35,
            )

        self.create_incident(
            tube_station,
            text="faulty lift",
            resolved=False,
            start_time=timezone.now() - timedelta(hours=1),
            estimated_duration=timedelta(hours=2),
            prediction_confidence=0.25,
        )
        self.create_incident(
            dlr_station,
            text="planned maintenance",
            resolved=False,
            start_time=timezone.now() - timedelta(hours=1),
            estimated_duration=timedelta(hours=2),
            prediction_confidence=0.35,
        )
        self.create_incident(
            dlr_station,
            text="unavailability of station staff",
            resolved=False,
            start_time=timezone.now() - timedelta(hours=1),
            estimated_duration=timedelta(hours=2),
            prediction_confidence=0.35,
        )

        with patch("incidents.views.get_last_updated", return_value="09:15 26 Mar"):
            response = self.client.get("/stats/")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Hidden Low-Accuracy Diagnostics")
        self.assertContains(response, "By confidence bucket")
        self.assertContains(response, "By category")
        self.assertContains(response, "By network")
        self.assertContains(response, "By station")

        bucket_rows = {
            row["label"]: row["count"]
            for row in response.context["current_hidden_low_accuracy_bucket_rows"]
        }
        self.assertEqual(bucket_rows, {"20-30%": 1, "30-40%": 2})

        category_rows = {
            row["label"]: row["count"]
            for row in response.context["current_hidden_low_accuracy_category_rows"]
        }
        self.assertEqual(
            category_rows,
            {"Faulty lift": 1, "Planned maintenance": 1, "Staff issue": 1},
        )

        network_rows = {
            row["label"]: row["count"]
            for row in response.context["current_hidden_low_accuracy_network_rows"]
        }
        self.assertEqual(network_rows, {"Tube": 1, "DLR": 2})

        station_rows = {
            row["station_name"]: row
            for row in response.context["current_hidden_low_accuracy_station_rows"]
        }
        self.assertEqual(station_rows["Green Park"]["current_hidden_count"], 1)
        self.assertEqual(station_rows["Green Park"]["recent_prediction_count"], 15)
        self.assertEqual(station_rows["Green Park"]["recent_accuracy_pct"], 0)
        self.assertTrue(station_rows["Green Park"]["has_enough_history"])
        self.assertEqual(station_rows["West India Quay"]["current_hidden_count"], 2)
        self.assertEqual(station_rows["West India Quay"]["recent_prediction_count"], 15)
        self.assertEqual(station_rows["West India Quay"]["recent_accuracy_pct"], 0)
        self.assertTrue(station_rows["West India Quay"]["has_enough_history"])

    def test_stats_uses_small_constant_number_of_queries(self):
        parent = self.create_parent_station("Green Park")
        now = timezone.now()
        self.create_incident(
            parent,
            text="unavailability of station staff",
            start_time=now - timedelta(hours=5),
            end_time=now - timedelta(hours=3),
            resolved=True,
        )
        self.create_incident(
            parent,
            text="faulty lift",
            start_time=now - timedelta(hours=3),
            end_time=now - timedelta(hours=2),
            resolved=True,
        )
        self.create_incident(
            parent,
            text="planned maintenance",
            start_time=now - timedelta(hours=2),
            end_time=now - timedelta(hours=1, minutes=30),
            resolved=True,
        )

        with patch("incidents.views.get_last_updated", return_value="09:15 26 Mar"):
            with CaptureQueriesContext(connection) as queries:
                response = self.client.get("/stats/")

        self.assertEqual(response.status_code, 200)
        self.assertLessEqual(len(queries), 3)

    def test_faq_and_privacy_pages_render(self):
        faq_response = self.client.get("/faq/")
        privacy_response = self.client.get("/privacy/")

        self.assertEqual(faq_response.status_code, 200)
        self.assertEqual(privacy_response.status_code, 200)
        self.assertTemplateUsed(faq_response, "faq.html")
        self.assertTemplateUsed(privacy_response, "privacy.html")

    def test_update_incidents_command_runs_and_wraps_failures(self):
        stdout = StringIO()

        with patch(
            "incidents.management.commands.update_incidents.check_tflv1"
        ) as check_tflv1_mock, patch(
            "incidents.management.commands.update_incidents.consolidate_incidents"
        ) as consolidate_mock, patch(
            "incidents.management.commands.update_incidents.update_last_updated"
        ) as update_last_updated_mock:
            call_command("update_incidents", stdout=stdout)

        check_tflv1_mock.assert_called_once_with()
        consolidate_mock.assert_called_once_with()
        update_last_updated_mock.assert_called_once_with()
        self.assertIn("Successfully updated incident list", stdout.getvalue())

        with patch(
            "incidents.management.commands.update_incidents.check_tflv1",
            side_effect=Exception("boom"),
        ):
            with self.assertRaises(CommandError):
                call_command("update_incidents")
