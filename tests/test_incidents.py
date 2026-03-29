import arrow
import os
import tempfile

from datetime import timedelta
from io import StringIO
from unittest.mock import MagicMock, patch

from django.core.management import CommandError, call_command
from django.test import SimpleTestCase, TestCase, override_settings
from django.utils import timezone

from incidents.management.commands.update_incidents import consolidate_incidents
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


@override_settings(ROOT_URLCONF="updown.urls")
class ViewAndCommandTests(StationFactoryMixin, TestCase):
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
