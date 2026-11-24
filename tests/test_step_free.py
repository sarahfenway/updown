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


from stations.step_free.tfl_data import FeedError, load_feed

TCR = "HUBTCR"
TCR_POINT = "HUBTCR-1000235-"


class FeedParsingTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.feed = load_feed(fixture_zip())
        cls.tcr = cls.feed.stations[TCR]

    def test_reads_publish_date_and_stations(self):
        self.assertEqual(self.feed.published, date(2026, 10, 20))
        self.assertEqual(sorted(self.feed.stations), ["910GHAKNYNM", "HUBBAL", "HUBTCR"])

    def test_platform_labels_and_train_access(self):
        platforms = self.tcr["platforms"]
        self.assertEqual(
            platforms["HUBTCR-Plat02-EB-central"]["label"],
            "Central line – Eastbound Platform 2",
        )
        northern = platforms["HUBTCR-Plat03-NB-northern"]
        self.assertEqual(northern["step"], "0–50 mm")
        self.assertEqual(northern["gap"], "0–85 mm")

    def test_only_covered_platforms_are_kept(self):
        self.assertEqual(
            set(self.feed.stations["HUBBAL"]["platforms"]),
            {"HUBBAL-Plat01-NB-northern", "HUBBAL-Plat02-SB-northern"},
        )

    def test_bus_stops_are_dropped(self):
        codes = [place["code"] for place in self.tcr["places"].values()]
        self.assertFalse([code for code in codes if code.lower().startswith("bus")])

    def test_places_get_readable_unique_names(self):
        places = self.tcr["places"]
        self.assertEqual(places[TCR_POINT + "BookH-13"]["name"], "Ticket hall (BookH)")
        self.assertEqual(places[TCR_POINT + "DSBH-27"]["name"], "Ticket hall (DSBH)")
        self.assertEqual(places[TCR_POINT + "D-NPE-21"]["name"], "Entrance (D-NPE)")
        self.assertEqual(places[TCR_POINT + "UpCon-17"]["name"], "Upper concourse")

    def test_lift_with_an_intermediate_stop_links_each_stop(self):
        lift_d = [link for link in self.tcr["links"] if link.get("lift_id") == "HUBTCR-Lift-4"]
        self.assertEqual(
            {frozenset((link["a"], link["b"])) for link in lift_d},
            {
                frozenset((TCR_POINT + "BookH-13", TCR_POINT + "UpCon-17")),
                frozenset((TCR_POINT + "UpCon-17", TCR_POINT + "NORPL-22")),
            },
        )
        self.assertEqual({link["lift_name"] for link in lift_d}, {"Lift D"})

    def test_street_level_entrances_are_starts(self):
        self.assertEqual(
            self.feed.stations["910GHAKNYNM"]["starts"],
            [
                "910GHAKNYNM-Outside",
                "910GHAKNYNM-1001128-Entr-3",
                "910GHAKNYNM-1001128-IENT-8",
            ],
        )

    def test_codes_for_matching(self):
        self.assertIn("940GZZLUTCR", self.feed.codes[TCR])
        self.assertIn("HUBTCR", self.feed.codes[TCR])

    def test_layout_is_stored_for_every_drawn_place(self):
        layout = self.tcr["layout"]
        self.assertEqual(layout["levels"], [0, -1, -2, -3, -5, -6])
        for pid, place in self.tcr["places"].items():
            if place["kind"] != "street":
                self.assertIn(pid, layout["positions"])

    def test_missing_file_is_a_feed_error(self):
        with self.assertRaises(FeedError):
            load_feed(fixture_zip(skip={"Lifts.csv"}))

    def test_not_a_zip_is_a_feed_error(self):
        with self.assertRaises(FeedError):
            load_feed(b"not a zip")


from stations.step_free.routing import BLOCKED, NO_ROUTE, OK, find_route

CENTRAL_EB = "HUBTCR-Plat02-EB-central"
NORTHERN_NB = "HUBTCR-Plat03-NB-northern"


class RoutingTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.feed = load_feed(fixture_zip())
        cls.tcr = cls.feed.stations[TCR]

    def test_route_to_central_line_eastbound(self):
        result = find_route(self.tcr, CENTRAL_EB)

        self.assertEqual(result.status, OK)
        self.assertEqual(result.places[0], "HUBTCR-Outside")
        self.assertEqual(result.places[-1], CENTRAL_EB)
        self.assertEqual(result.steps, [
            "Enter at Entrance (D-NPE)",
            "Take Lift B down to Ticket hall (BookH)",
            "Take Lift E down to Upper concourse",
            "Take Lift F down to Central line eastbound platform",
            "Go to Eastbound Platform 2",
            "Eastbound Platform 2: staff ramp available",
        ])

    def test_one_lift_ride_through_a_stop_counts_as_one_lift(self):
        # Lift D goes ticket hall -> upper concourse -> Northern line, so
        # B then D (two lifts) beats B, C, G (three).
        result = find_route(self.tcr, NORTHERN_NB)

        self.assertEqual(result.steps[:3], [
            "Enter at Entrance (D-NPE)",
            "Take Lift B down to Ticket hall (BookH)",
            "Take Lift D down to Northern line platforms",
        ])
        self.assertEqual(
            result.steps[-1],
            "Northbound Platform 3: step to train 0–50 mm, gap 0–85 mm",
        )

    def test_broken_lift_with_an_alternative_is_avoided(self):
        result = find_route(self.tcr, NORTHERN_NB, ["HUBTCR-Lift-4"])  # Lift D

        self.assertEqual(result.status, OK)
        self.assertFalse([step for step in result.steps if "Lift D" in step])
        self.assertIn("Take Lift G down to Northern line platforms", result.steps)

    def test_broken_lift_on_the_only_route_blocks_it(self):
        result = find_route(self.tcr, CENTRAL_EB, ["HUBTCR-Lift-5"])  # Lift F

        self.assertEqual(result.status, BLOCKED)
        self.assertEqual(result.blocking_lifts, ["Lift F"])
        self.assertEqual(result.steps, [])

    def test_hackney_downs_side_entrance_reaches_platform_1(self):
        hackney = self.feed.stations["910GHAKNYNM"]
        result = find_route(hackney, "910GHAKNYNM-Plat01-SB-london-overground|national-rail")

        self.assertEqual(result.steps, [
            "Enter at Entrance (IENT)",
            "Take Lift 1 up to Platform level (RPL-1)",
            "Go to Southbound Platform 1",
            "Southbound Platform 1: staff ramp available",
        ])

    def test_balham_northern_line_has_no_step_free_route(self):
        balham = self.feed.stations["HUBBAL"]
        for platform_id in balham["platforms"]:
            with self.subTest(platform=platform_id):
                self.assertEqual(find_route(balham, platform_id).status, NO_ROUTE)

    def test_ramp_and_small_lift_wording(self):
        routes = {
            "places": {
                "out": {"name": "Street", "level": 0, "kind": "street"},
                "e": {"name": "Main entrance", "level": 0, "kind": "entrance"},
                "h": {"name": "Ticket hall", "level": -1, "kind": "place"},
                "p": {"name": "Platform 1", "level": -1, "kind": "platform"},
            },
            "links": [
                {"a": "out", "b": "e", "kind": "walkway"},
                {"a": "e", "b": "h", "kind": "lift", "lift_id": "L1",
                 "lift_name": "Lift 1", "small": True},
                {"a": "h", "b": "p", "kind": "ramp"},
            ],
            "platforms": {"p": {"name": "Platform 1", "step": None, "gap": None,
                                "manual_ramp": False}},
            "starts": ["out"],
        }

        self.assertEqual(find_route(routes, "p").steps, [
            "Enter at Main entrance",
            "Take Lift 1 (small lift) down to Ticket hall",
            "Use the ramp to Platform 1",
        ])


from io import StringIO

from django.core.management import CommandError, call_command

from stations.step_free.importer import ImportAborted, import_feed


def make_fixture_stations():
    """Our records for the three fixture stations, as update_stations makes them."""
    tcr = make_station("Tottenham Court Road", hub_naptan_id="HUBTCR",
                       naptan_id="940GZZLUTCR", tube=True, crossrail=True)
    hackney = make_station("Hackney Downs", naptan_id="910GHAKNYNM", overground=True)
    balham = make_station("Balham", hub_naptan_id="HUBBAL",
                          naptan_id="940GZZLUBLM", tube=True)
    return tcr, hackney, balham


class ImportTests(TestCase):
    def setUp(self):
        self.tcr, self.hackney, self.balham = make_fixture_stations()

    def test_stores_routes_on_main_records(self):
        result = import_feed(load_feed(fixture_zip()), min_stations=1)

        self.assertEqual(result.matched, 3)
        self.tcr.refresh_from_db()
        self.assertEqual(self.tcr.tfl_station_id, "HUBTCR")
        self.assertEqual(self.tcr.step_free_routes_published, date(2026, 10, 20))
        self.assertIn(CENTRAL_EB, self.tcr.step_free_routes["platforms"])
        self.assertIn("positions", self.tcr.step_free_routes["layout"])

    def test_child_record_codes_match_to_the_parent(self):
        self.tcr.hub_naptan_id = None
        self.tcr.naptan_id = None
        self.tcr.save()
        child = make_station("Tottenham Court Road", naptan_id="910GTOTCTRD",
                             crossrail=True, parent_station=self.tcr)

        import_feed(load_feed(fixture_zip()), min_stations=1)

        self.tcr.refresh_from_db()
        child.refresh_from_db()
        self.assertEqual(self.tcr.tfl_station_id, "HUBTCR")
        self.assertIsNone(child.step_free_routes)

    def test_name_match_when_no_code_matches(self):
        self.hackney.naptan_id = "SOMETHING-ELSE"
        self.hackney.save()

        import_feed(load_feed(fixture_zip()), min_stations=1)

        self.hackney.refresh_from_db()
        self.assertEqual(self.hackney.tfl_station_id, "910GHAKNYNM")

    def test_duplicate_records_store_on_the_lower_id(self):
        duplicate = make_station("Balham", hub_naptan_id="HUBBAL", tube=True)

        import_feed(load_feed(fixture_zip()), min_stations=1)

        self.balham.refresh_from_db()
        duplicate.refresh_from_db()
        self.assertEqual(self.balham.tfl_station_id, "HUBBAL")
        self.assertIsNone(duplicate.tfl_station_id)

    def test_too_few_stations_aborts_and_keeps_old_data(self):
        self.tcr.step_free_routes = {"old": True}
        self.tcr.save()

        with self.assertRaises(ImportAborted):
            import_feed(load_feed(fixture_zip()), min_stations=200)

        self.tcr.refresh_from_db()
        self.assertEqual(self.tcr.step_free_routes, {"old": True})

    def test_two_tfl_stations_for_one_of_ours_are_merged(self):
        # TfL lists West Hampstead's Tube and Overground stations separately
        # under one hub code; we count them as one station. Simulate that by
        # giving one record codes from two fixture stations.
        Station.objects.filter(pk=self.balham.pk).delete()
        self.tcr.naptan_id = "940GZZLUBLM"  # Balham's Tube code
        self.tcr.save()

        import_feed(load_feed(fixture_zip()), min_stations=1)

        self.tcr.refresh_from_db()
        routes = self.tcr.step_free_routes
        self.assertEqual(routes["tfl_ids"], ["HUBBAL", "HUBTCR"])
        self.assertIn(CENTRAL_EB, routes["platforms"])
        self.assertIn("HUBBAL-Plat01-NB-northern", routes["platforms"])
        self.assertIn("HUBTCR-Outside", routes["starts"])
        self.assertIn("HUBBAL-Outside", routes["starts"])
        for pid, place in routes["places"].items():
            if place["kind"] != "street":
                self.assertIn(pid, routes["layout"]["positions"])
        self.assertEqual(find_route(routes, CENTRAL_EB).status, OK)

    def test_stations_missing_from_the_file_are_cleared(self):
        gone = make_station("Old Street", hub_naptan_id="HUBOLD", tube=True,
                            step_free_routes={"old": True}, tfl_station_id="HUBOLD")

        result = import_feed(load_feed(fixture_zip()), min_stations=1)

        gone.refresh_from_db()
        self.assertIsNone(gone.step_free_routes)
        self.assertIsNone(gone.tfl_station_id)
        self.assertEqual(result.cleared, 1)


@patch("stations.step_free.importer.MIN_STATIONS", 1)
class ImportCommandTests(TestCase):
    def setUp(self):
        self.tcr, _, _ = make_fixture_stations()

    def run_command(self, *args):
        out = StringIO()
        call_command("import_station_routes", *args, stdout=out)
        return out.getvalue()

    def test_downloads_and_imports(self):
        with patch("stations.management.commands.import_station_routes.download_zip",
                   return_value=fixture_zip()):
            output = self.run_command()

        self.assertIn("Imported step-free routes for 3 stations", output)
        self.tcr.refresh_from_db()
        self.assertEqual(self.tcr.tfl_station_id, "HUBTCR")

    def test_unchanged_publish_date_skips_the_import(self):
        self.tcr.step_free_routes_published = date(2026, 10, 20)
        self.tcr.save()

        with patch("stations.management.commands.import_station_routes.download_zip",
                   return_value=fixture_zip()):
            output = self.run_command()

        self.assertIn("unchanged", output)
        self.tcr.refresh_from_db()
        self.assertIsNone(self.tcr.tfl_station_id)

    def test_broken_file_fails_and_keeps_data(self):
        self.tcr.step_free_routes = {"old": True}
        self.tcr.save()

        with patch("stations.management.commands.import_station_routes.download_zip",
                   return_value=fixture_zip(skip={"Platforms.csv"})):
            with self.assertRaises(CommandError):
                self.run_command()

        self.tcr.refresh_from_db()
        self.assertEqual(self.tcr.step_free_routes, {"old": True})


import requests

from stations.step_free.broken_lifts import update_broken_lifts


class FakeLiftResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.HTTPError("bad status")

    def json(self):
        return self._payload


class BrokenLiftTests(TestCase):
    def setUp(self):
        self.tcr = make_station("Tottenham Court Road", tube=True, tfl_station_id="HUBTCR",
                                step_free_routes={"tfl_ids": ["HUBTCR"]})
        self.bank = make_station("Bank", tube=True, tfl_station_id="HUBBAN",
                                 step_free_routes={"tfl_ids": ["HUBBAN"]},
                                 broken_lift_ids=["HUBBAN-Lift-1"])

    def test_merged_station_gets_lifts_from_every_tfl_station(self):
        west_hampstead = make_station(
            "West Hampstead", tube=True, tfl_station_id="910GWHMDSTD",
            step_free_routes={"tfl_ids": ["910GWHMDSTD", "940GZZLUWHP"]},
        )
        payload = [
            {"stationUniqueId": "910GWHMDSTD", "disruptedLiftUniqueIds": ["910GWHMDSTD-Lift-1"]},
            {"stationUniqueId": "940GZZLUWHP", "disruptedLiftUniqueIds": ["940GZZLUWHP-Lift-2"]},
        ]
        with patch("stations.step_free.broken_lifts.requests.get",
                   return_value=FakeLiftResponse(payload)):
            update_broken_lifts()

        west_hampstead.refresh_from_db()
        self.assertEqual(west_hampstead.broken_lift_ids,
                         ["910GWHMDSTD-Lift-1", "940GZZLUWHP-Lift-2"])

    def test_stores_broken_lift_ids_per_station(self):
        payload = [
            {"stationUniqueId": "HUBTCR", "disruptedLiftUniqueIds": ["HUBTCR-Lift-5"],
             "message": "Tottenham Court Road: ..."},
            {"stationUniqueId": "HUBXYZ", "disruptedLiftUniqueIds": ["HUBXYZ-Lift-1"],
             "message": "Not one of ours"},
        ]
        with patch("stations.step_free.broken_lifts.requests.get",
                   return_value=FakeLiftResponse(payload)):
            self.assertTrue(update_broken_lifts())

        self.tcr.refresh_from_db()
        self.bank.refresh_from_db()
        self.assertEqual(self.tcr.broken_lift_ids, ["HUBTCR-Lift-5"])
        self.assertEqual(self.bank.broken_lift_ids, [])
        self.assertIsNotNone(self.tcr.broken_lifts_checked_at)

    def test_failed_fetch_keeps_previous_data(self):
        with patch("stations.step_free.broken_lifts.requests.get",
                   side_effect=requests.ConnectionError("down")):
            self.assertFalse(update_broken_lifts())

        self.bank.refresh_from_db()
        self.assertEqual(self.bank.broken_lift_ids, ["HUBBAN-Lift-1"])
        self.assertIsNone(self.bank.broken_lifts_checked_at)

    def test_unexpected_payload_keeps_previous_data(self):
        with patch("stations.step_free.broken_lifts.requests.get",
                   return_value=FakeLiftResponse({"error": "nope"})):
            self.assertFalse(update_broken_lifts())

        self.bank.refresh_from_db()
        self.assertEqual(self.bank.broken_lift_ids, ["HUBBAN-Lift-1"])


from django.test import override_settings

from incidents.models import Incident
from stations.step_free.diagram import build_diagram, level_name


class DiagramTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tcr = load_feed(fixture_zip()).stations[TCR]

    def test_level_names(self):
        self.assertEqual(level_name(0), "Street level")
        self.assertEqual(level_name(-1), "1 level below street")
        self.assertEqual(level_name(-3), "3 levels below street")
        self.assertEqual(level_name(2), "2 levels above street")

    def test_one_box_per_drawn_place_and_one_band_per_level(self):
        diagram = build_diagram(self.tcr)
        drawn = [p for p in self.tcr["places"].values() if p["kind"] != "street"]
        self.assertEqual(len(diagram["boxes"]), len(drawn))
        self.assertEqual(len(diagram["bands"]), len(self.tcr["layout"]["levels"]))

    def test_broken_lift_is_marked(self):
        diagram = build_diagram(self.tcr, ["HUBTCR-Lift-5"])
        broken = [line for line in diagram["lines"] if "broken" in line["cls"]]
        self.assertTrue(broken)
        self.assertTrue(all("out of service" in line["label"] for line in broken))

    def test_route_is_highlighted(self):
        route = find_route(self.tcr, CENTRAL_EB)
        diagram = build_diagram(self.tcr, route=route)
        on_route = [box for box in diagram["boxes"] if "on-route" in box["cls"]]
        # Every place on the route except the street node, which is not drawn.
        self.assertEqual(len(on_route), len(route.places) - 1)


@override_settings(ROOT_URLCONF="updown.urls")
class StepFreePageTests(TestCase):
    def setUp(self):
        self.tcr, self.hackney, self.balham = make_fixture_stations()
        import_feed(load_feed(fixture_zip()), min_stations=1)
        self.tcr.refresh_from_db()

    def set_lift_status(self, broken, age=timedelta(minutes=1)):
        Station.objects.filter(pk=self.tcr.pk).update(
            broken_lift_ids=broken, broken_lifts_checked_at=timezone.now() - age
        )

    def test_index_lists_stations_and_redirects_to_choice(self):
        response = self.client.get("/step-free/")
        self.assertContains(response, "Tottenham Court Road")
        self.assertContains(response, 'name="robots" content="noindex"')

        response = self.client.get("/step-free/", {"station": "tottenham-court-road"})
        self.assertRedirects(response, "/step-free/tottenham-court-road/",
                             fetch_redirect_response=False)

    def test_station_page_shows_diagram_and_directions(self):
        self.set_lift_status([])
        response = self.client.get("/step-free/tottenham-court-road/",
                                   {"platform": CENTRAL_EB})

        self.assertContains(response, "<svg")
        self.assertContains(response, "Take Lift F down to Central line eastbound platform")
        self.assertContains(response, 'name="robots" content="noindex"')
        # No route data reaches the page: no place ids, lift ids or JSON keys.
        # (base.html's site-wide jQuery scripts are unrelated and stay.)
        content = response.content.decode()
        self.assertNotIn("HUBTCR-1000235-", content)
        self.assertNotIn("HUBTCR-Lift-", content)
        self.assertNotIn('"places"', content)

    def test_broken_lift_blocks_route_and_is_named(self):
        self.set_lift_status(["HUBTCR-Lift-5"])
        response = self.client.get("/step-free/tottenham-court-road/",
                                   {"platform": CENTRAL_EB})

        self.assertContains(response, "No step-free route to this platform right now")
        self.assertContains(response, "Lift F")
        self.assertContains(response, "out of service")

    def test_stale_lift_status_is_unavailable(self):
        self.set_lift_status(["HUBTCR-Lift-5"], age=timedelta(minutes=31))
        response = self.client.get("/step-free/tottenham-court-road/",
                                   {"platform": CENTRAL_EB})

        self.assertContains(response, "Live lift status is unavailable")
        self.assertContains(response, "Take Lift F down to Central line eastbound platform")
        self.assertNotContains(response, "out of service")

    def test_no_route_in_tfl_data(self):
        response = self.client.get("/step-free/balham/",
                                   {"platform": "HUBBAL-Plat01-NB-northern"})
        # Literal template text, so Django does not escape the apostrophe.
        self.assertContains(response, "TfL's data shows no step-free route to this platform")

    def test_current_issues_are_shown(self):
        Incident.objects.create(station=self.tcr, information=False,
                                text="Step free access is not available to the Central line.")
        response = self.client.get("/step-free/tottenham-court-road/")
        self.assertContains(response, "Step free access is not available to the Central line.")

    def test_bad_platform_is_ignored(self):
        response = self.client.get("/step-free/tottenham-court-road/", {"platform": "nope"})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "<svg")
        self.assertNotContains(response, "Take Lift")

    def test_unknown_station_is_404(self):
        self.assertEqual(self.client.get("/step-free/nowhere/").status_code, 404)

    def test_different_stations_with_one_name_each_get_a_page(self):
        # Like the two Bethnal Greens (Central line; Weaver line): separate
        # stations that happen to share a name.
        feed = load_feed(fixture_zip())
        make_station("Bethnal Green", tube=True, tfl_station_id="HUBTCR",
                     step_free_routes=feed.stations["HUBTCR"])
        make_station("Bethnal Green", overground=True, tfl_station_id="910GHAKNYNM",
                     step_free_routes=feed.stations["910GHAKNYNM"])

        index = self.client.get("/step-free/")
        self.assertContains(index, "Bethnal Green (Central line, Elizabeth line, Northern line)")
        self.assertContains(index, "Bethnal Green (Weaver line)")

        tube = self.client.get("/step-free/bethnal-green-central-line/")
        overground = self.client.get("/step-free/bethnal-green-weaver-line/")
        self.assertContains(tube, "Central line – Eastbound Platform 2")
        self.assertContains(overground, "Weaver line – Southbound Platform 1")
        # The plain name still works, for links made before the suffix.
        self.assertEqual(self.client.get("/step-free/bethnal-green/").status_code, 200)

    def test_station_without_route_data(self):
        make_station("Oval", hub_naptan_id="HUBOVL", tube=True)
        response = self.client.get("/step-free/oval/")
        self.assertContains(response, "TfL has not published step-free route data for this station.")
