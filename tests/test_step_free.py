import io
import zipfile
from datetime import datetime, timedelta, timezone as dt_timezone
from pathlib import Path
from unittest.mock import patch

from django.db import OperationalError, connection
from django.test import SimpleTestCase, TestCase, override_settings
from django.test.utils import CaptureQueriesContext
from django.utils import timezone

from stations.models import LiftStatusCheck, Station, StationRoutes

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "stationdata"


# FeedStartDate in tests/fixtures/stationdata/FeedInfo.csv.
PUBLISHED = datetime(2026, 10, 20, 13, 56, tzinfo=dt_timezone.utc)


def fixture_zip(skip=()):
    """A zip shaped like TfL's station data file, holding three made-up
    stations written by hand for these tests (no TfL data)."""
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


def make_routes(station, data=None, tfl_ids=None, broken=None, published=PUBLISHED):
    tfl_ids = tfl_ids or ["HUBOTH"]
    return StationRoutes.objects.create(
        station=station, tfl_station_id=tfl_ids[0], tfl_ids=tfl_ids,
        published=published, data=data or {"tfl_ids": tfl_ids}, broken_lift_ids=broken or [],
    )


def routes_of(station):
    return StationRoutes.objects.filter(station=station).first()


class StationRoutesModelTests(TestCase):
    def test_route_data_round_trips(self):
        station = make_station("Bank", tube=True)
        make_routes(station, data={"tfl_id": "HUBOTH", "places": {}}, broken=["HUBOTH-Lift-1"])

        # A fresh object, so plain Python attributes can't fake a pass.
        stored = StationRoutes.objects.get(pk=station.pk)

        self.assertEqual(stored.data["tfl_id"], "HUBOTH")
        self.assertEqual(stored.published, PUBLISHED)
        self.assertEqual(stored.broken_lift_ids, ["HUBOTH-Lift-1"])

    def test_station_rows_do_not_carry_route_data(self):
        # Station rows are loaded on every page and API request; the 3-22 KB
        # route JSON must not ride along.
        self.assertFalse({"data", "step_free_routes", "broken_lift_ids"}
                         & {field.name for field in Station._meta.concrete_fields})


from stations.step_free.labels import is_dropped, place_label


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

    def test_ticket_halls_and_mezzanines(self):
        for code in ("TTH", "NTH", "WTH", "ELTH"):
            with self.subTest(code=code):
                self.assertEqual(place_label(code), "Ticket hall")
        self.assertEqual(place_label("Mezz"), "Mezzanine")
        self.assertIsNone(place_label("SOUTH"))
        self.assertIsNone(place_label("NORTH"))

    def test_unknown_code_has_no_label(self):
        self.assertIsNone(place_label("BCQ"))

    def test_codes_that_only_look_like_line_names(self):
        # Real codes in TfL's file: CEntr is an entrance; North, ELVER and
        # ELM are not the Northern or Elizabeth lines (ELTH, a ticket hall,
        # is covered by the ticket-hall rule).
        self.assertEqual(place_label("CEntr"), "Entrance")
        self.assertEqual(place_label("ELEnt"), "Entrance")
        self.assertEqual(place_label("Centr"), "Central line level")
        self.assertEqual(place_label("PICCA"), "Piccadilly line level")
        self.assertEqual(place_label("ELWL"), "Elizabeth line level")
        for code in ("North", "ELVER", "ELM"):
            with self.subTest(code=code):
                self.assertIsNone(place_label(code))

    def test_bus_and_taxi_points_are_dropped(self):
        self.assertTrue(is_dropped("Bus C"))
        self.assertTrue(is_dropped("BUS"))
        self.assertTrue(is_dropped("Taxi"))
        self.assertFalse(is_dropped("BookH"))


from stations.step_free.tfl_data import (
    DATA_FORMAT, FeedError, _merge_platform_areas, _name_places, load_feed, merge_routes,
)

EXC = "HUBEXC"
EXC_POINT = "HUBEXC-1000001-"


class FeedParsingTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.feed = load_feed(fixture_zip())
        cls.exc = cls.feed.stations[EXC]

    def test_reads_publish_date_and_stations(self):
        self.assertEqual(self.feed.published, PUBLISHED)
        self.assertEqual(sorted(self.feed.stations), ["910GSMPLDN", "HUBEXC", "HUBTST"])

    def test_platform_labels_and_train_access(self):
        platforms = self.exc["platforms"]
        self.assertEqual(
            platforms["HUBEXC-Plat02-EB-central"]["label"],
            "Central line – Eastbound Platform 2",
        )
        northern = platforms["HUBEXC-Plat03-NB-northern"]
        self.assertEqual(northern["step"], "0–50 mm")
        self.assertEqual(northern["gap"], "0–85 mm")

    def test_only_covered_platforms_are_kept(self):
        self.assertEqual(
            set(self.feed.stations["HUBTST"]["platforms"]),
            {"HUBTST-Plat01-NB-northern", "HUBTST-Plat02-SB-northern"},
        )

    def test_bus_stops_are_dropped(self):
        codes = [place["code"] for place in self.exc["places"].values()]
        self.assertFalse([code for code in codes if code.lower().startswith("bus")])

    def test_places_get_readable_unique_names(self):
        places = self.exc["places"]
        self.assertEqual(places[EXC_POINT + "BookH-5"]["name"], "North-west ticket hall")
        self.assertEqual(places[EXC_POINT + "DSBH-4"]["name"], "East ticket hall")
        self.assertEqual(places[EXC_POINT + "A-NOR-1"]["name"], "North-west entrance")
        self.assertEqual(places[EXC_POINT + "UpCon-6"]["name"], "Upper concourse")

    def test_lift_with_an_intermediate_stop_links_each_stop(self):
        lift_d = [link for link in self.exc["links"] if link.get("lift_id") == "HUBEXC-Lift-4"]
        self.assertEqual(
            {frozenset((link["a"], link["b"])) for link in lift_d},
            {
                frozenset((EXC_POINT + "BookH-5", EXC_POINT + "UpCon-6")),
                frozenset((EXC_POINT + "UpCon-6", EXC_POINT + "NORPL-9")),
            },
        )
        self.assertEqual({link["lift_name"] for link in lift_d}, {"Lift D"})

    def test_street_level_entrances_are_starts(self):
        self.assertEqual(
            self.feed.stations["910GSMPLDN"]["starts"],
            [
                "910GSMPLDN-Outside",
                "910GSMPLDN-1000002-Entr-3",
                "910GSMPLDN-1000002-IENT-8",
            ],
        )

    def test_codes_for_matching(self):
        self.assertIn("940GZZLUEXC", self.feed.codes[EXC])
        self.assertIn("HUBEXC", self.feed.codes[EXC])

    def test_platforms_with_no_step_free_link_get_no_level(self):
        # Testbury's Northern line platforms have no step-free link at all;
        # they mustn't be drawn at street level.
        testbury = self.feed.stations["HUBTST"]
        for platform_id in ("HUBTST-Plat01-NB-northern", "HUBTST-Plat02-SB-northern"):
            self.assertIsNone(testbury["places"][platform_id]["level"])

    def test_merged_stations_keep_names_distinct_and_sources_unchanged(self):
        sample_downs, testbury = self.feed.stations["910GSMPLDN"], self.feed.stations["HUBTST"]
        before = repr(sample_downs), repr(testbury)

        merged = merge_routes([sample_downs, testbury])

        self.assertEqual((repr(sample_downs), repr(testbury)), before)
        self.assertEqual(merged["tfl_ids"], ["910GSMPLDN", "HUBTST"])
        lift_names = {link["lift_name"] for link in merged["links"] if link["kind"] == "lift"}
        self.assertEqual(lift_names, {"Lift 1 – Weaver line", "Lift 1 – Northern line"})
        names = [p["name"] for p in merged["places"].values() if p["kind"] != "street"]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("South-west platform level", names)

    def test_area_joined_to_its_platform_by_a_lift_or_ramp_is_not_merged(self):
        places = {
            "area": {"kind": "place", "code": "RPL", "name": "Platform level", "level": 1},
            "plat": {"kind": "platform", "code": "", "name": "Platform 1", "level": 1},
        }
        links = [{"a": "area", "b": "plat", "kind": "walkway"},
                 {"a": "area", "b": "plat", "kind": "lift", "lift_id": "L", "lift_name": "Lift 1"}]
        platforms = {"plat": {"label": "Weaver line – Platform 1"}}
        self.assertEqual(_merge_platform_areas(places, links, platforms), {})

    def test_platform_areas_serving_one_platform_merge_into_it(self):
        # TfL lists most platforms twice: the platform-area point and the
        # platform itself. Where an area serves one platform, they're one box.
        sample_downs = self.feed.stations["910GSMPLDN"]
        self.assertEqual(sample_downs["merged"], {
            "910GSMPLDN-1000002-RPL-1-2": "910GSMPLDN-Plat01-SB-london-overground|national-rail",
            "910GSMPLDN-1000002-RPL-2-4": "910GSMPLDN-Plat02-NB-london-overground|national-rail",
        })
        self.assertEqual(
            sample_downs["places"]["910GSMPLDN-Plat01-SB-london-overground|national-rail"]["name"],
            "Weaver line – Southbound Platform 1",
        )
        # Where areas reach several platforms (King's Cross's NorNB and NorSB
        # both reach platforms 7 and 8), the code's line and direction say
        # which platform an area is.
        self.assertEqual(self.exc["merged"]["HUBEXC-1000001-CenEB-7"], "HUBEXC-Plat02-EB-central")
        self.assertEqual(self.exc["merged"]["HUBEXC-1000001-CenWB-8"], "HUBEXC-Plat01-WB-central")
        # An area with no direction that serves several platforms stays.
        self.assertNotIn("HUBEXC-1000001-ELEL-10", self.exc["merged"])

    def test_platform_boarding_and_notes(self):
        platform = self.exc["platforms"]["HUBEXC-Plat03-NB-northern"]
        self.assertEqual(platform["boarding"], "2 centre doors on car 5")
        self.assertEqual(platform["notes"], ["Use the lift by the north entrance for this platform."])
        self.assertIsNone(self.exc["platforms"]["HUBEXC-Plat01-WB-central"]["boarding"])

    def test_lift_notes(self):
        lift_g = next(link for link in self.exc["links"] if link.get("lift_id") == "HUBEXC-Lift-7")
        self.assertEqual(lift_g["notes"], "Fits one wheelchair user at a time.")

    def test_station_facilities(self):
        facilities = self.exc["facilities"]
        self.assertEqual(facilities["blue_badge_spaces"], 2)
        self.assertTrue(facilities["taxi_rank"])
        # Only accessible toilets are kept.
        self.assertEqual(facilities["toilets"], [{
            "location": "Ticket hall", "inside_gates": True, "radar_key": True,
            "fee": False, "baby_changing": True, "hours": "Daily until 23:00",
        }])
        self.assertEqual(self.feed.stations["HUBTST"]["facilities"]["toilets"], [])

    def test_toilets_file_is_optional(self):
        feed = load_feed(fixture_zip(skip={"Toilets.csv"}))
        self.assertEqual(feed.stations["HUBEXC"]["facilities"]["toilets"], [])

    def test_unlabelled_places_get_unique_names(self):
        places = {
            "a": {"kind": "place", "code": "BCQ", "area_id": "1", "name": None},
            "b": {"kind": "place", "code": "BCQ", "area_id": "2", "name": None},
            "c": {"kind": "entrance", "code": "XYZ", "area_id": "3", "name": None},
            "d": {"kind": "entrance", "code": "XYZ", "area_id": "4", "name": None},
        }
        _name_places(places)
        self.assertEqual([p["name"] for p in places.values()],
                         ["Passageway 1", "Passageway 2", "Entrance 1", "Entrance 2"])

    def test_places_are_named_by_compass_direction_never_by_code(self):
        names = [p["name"] for p in self.exc["places"].values() if p["kind"] not in ("street", "platform")]
        self.assertIn("North-west entrance", names)
        self.assertIn("South-east entrance", names)
        for name in names:
            self.assertNotRegex(name, r"\(|BookH|DSBH|A-NOR|ELEL")

    def test_blank_lift_id_gets_one_of_its_own(self):
        lifts = (b"StationUniqueId,LiftUniqueId,LiftId,NationalRailUprn,LiftName,FriendlyName,"
                 b"FromAreas,IntermediateAreas,IntermediateAreas2,ToAreas,LimitedCapacityLift,LiftNotes\n"
                 b"910GSMPLDN,,1,,,Lift 1,910GSMPLDN-1000002-IENT-8,,,910GSMPLDN-1000002-RPL-1-2,False,\n")
        feed = load_feed(self._with_file(fixture_zip(), "Lifts.csv", lifts))
        lift_ids = {link["lift_id"] for link in feed.stations["910GSMPLDN"]["links"]
                    if link["kind"] == "lift"}
        self.assertEqual(lift_ids, {"910GSMPLDN-lift-row0"})

    def test_unreadable_files_are_feed_errors(self):
        good = fixture_zip()
        cases = {
            "short row": self._with_file(good, "StationPoints.csv", b"UniqueId,StationUniqueId\nX\n"),
            "not utf-8": self._with_file(good, "Stations.csv", b"UniqueId,Name\n\x96\x97,Bad\n"),
            "no date": self._with_file(good, "FeedInfo.csv", b"FeedPublisherName\nX\n"),
        }
        for label, data in cases.items():
            with self.subTest(label), self.assertRaises(FeedError):
                load_feed(data)

    @staticmethod
    def _with_file(data, name, content):
        source = zipfile.ZipFile(io.BytesIO(data))
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for item in source.namelist():
                archive.writestr(item, content if item == name else source.read(item))
        return buffer.getvalue()

    def test_missing_file_is_a_feed_error(self):
        with self.assertRaises(FeedError):
            load_feed(fixture_zip(skip={"Lifts.csv"}))

    def test_not_a_zip_is_a_feed_error(self):
        with self.assertRaises(FeedError):
            load_feed(b"not a zip")


from stations.step_free.routing import BLOCKED, NO_INFO, NO_ROUTE, OK, find_route

CENTRAL_EB = "HUBEXC-Plat02-EB-central"
NORTHERN_NB = "HUBEXC-Plat03-NB-northern"


class RoutingTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.feed = load_feed(fixture_zip())
        cls.exc = cls.feed.stations[EXC]

    def test_route_to_central_line_eastbound(self):
        result = find_route(self.exc, CENTRAL_EB)

        self.assertEqual(result.status, OK)
        self.assertEqual(result.places[0], "HUBEXC-Outside")
        self.assertEqual(result.places[-1], CENTRAL_EB)
        self.assertEqual(result.steps, [
            "Enter at North-west entrance",
            "Take Lift A down to North-west ticket hall",
            "Take Lift D down to Upper concourse",
            "Take Lift F down to Central line – Eastbound Platform 2",
            "Eastbound Platform 2: staff ramp available",
        ])

    def test_one_lift_ride_through_a_stop_counts_as_one_lift(self):
        # Lift D goes ticket hall -> upper concourse -> Northern line, so
        # B then D (two lifts) beats B, C, G (three).
        result = find_route(self.exc, NORTHERN_NB)

        self.assertEqual(result.steps[:3], [
            "Enter at North-west entrance",
            "Take Lift A down to North-west ticket hall",
            "Take Lift D down to Northern line platforms",
        ])
        self.assertEqual(
            result.steps[-1],
            "Northbound Platform 3: step to train 0–50 mm, gap 0–85 mm",
        )

    def test_broken_lift_with_an_alternative_is_avoided(self):
        result = find_route(self.exc, NORTHERN_NB, ["HUBEXC-Lift-4"])  # Lift D

        self.assertEqual(result.status, OK)
        self.assertFalse([step for step in result.steps if "Lift D" in step])
        self.assertIn("Take Lift G (small lift) up to Northern line platforms", result.steps)

    def test_broken_lift_on_the_only_route_blocks_it(self):
        result = find_route(self.exc, CENTRAL_EB, ["HUBEXC-Lift-6"])  # Lift F

        self.assertEqual(result.status, BLOCKED)
        self.assertEqual(result.blocking_lifts, ["Lift F"])
        self.assertEqual(result.steps, [])

    def test_sample_downs_side_entrance_reaches_platform_1(self):
        sample_downs = self.feed.stations["910GSMPLDN"]
        result = find_route(sample_downs, "910GSMPLDN-Plat01-SB-london-overground|national-rail")

        self.assertEqual(result.steps, [
            "Enter at East entrance",
            "Take Lift 1 up to Weaver line – Southbound Platform 1",
            "Southbound Platform 1: staff ramp available",
        ])
        # TfL's data doesn't link that entrance to the street: say so.
        self.assertTrue(result.unlinked_entrance)

    def test_street_routes_beat_unlinked_entrances(self):
        # The unlinked entrance "side" reaches the hall in one link, the
        # street in two; the street-linked route must still win.
        routes = self._routes(
            places={"side": ("Side entrance", 0, "entrance"), "main": ("Main entrance", 0, "entrance"),
                    "hall": ("Ticket hall", 0, "place")},
            links=[("out", "main", "walkway"), ("main", "hall", "walkway"), ("side", "hall", "walkway"),
                   ("hall", "p", "walkway")],
            starts=["out", "side"],
        )
        result = find_route(routes, "p")
        self.assertEqual(result.steps[0], "Enter at Main entrance")
        self.assertFalse(result.unlinked_entrance)

    def test_lift_straight_from_the_street_is_a_step(self):
        routes = self._routes(
            places={"hall": ("Ticket hall", -1, "place")},
            links=[("out", "hall", "lift"), ("hall", "p", "walkway")],
            starts=["out"],
        )
        self.assertEqual(find_route(routes, "p").steps, [
            "Start at street level",
            "Take Lift 1 down to Ticket hall",
            "Go to Platform 1",
        ])

    @staticmethod
    def _routes(places, links, starts):
        all_places = {"out": {"name": "Street", "level": 0, "kind": "street"},
                      "p": {"name": "Platform 1", "level": -1, "kind": "platform"}}
        for pid, (name, level, kind) in places.items():
            all_places[pid] = {"name": name, "level": level, "kind": kind}
        all_links = []
        for a, b, kind in links:
            link = {"a": a, "b": b, "kind": kind}
            if kind == "lift":
                link.update(lift_id="L1", lift_name="Lift 1", small=False)
            all_links.append(link)
        return {"places": all_places, "links": all_links, "starts": starts,
                "platforms": {"p": {"name": "Platform 1", "step": None, "gap": None,
                                    "manual_ramp": False}}}

    def test_testbury_northern_line_has_no_step_free_route(self):
        testbury = self.feed.stations["HUBTST"]
        self.assertEqual(find_route(testbury, "HUBTST-Plat01-NB-northern").status, NO_ROUTE)

    def test_unsurveyed_platform_is_no_information_not_no_route(self):
        testbury = self.feed.stations["HUBTST"]
        self.assertEqual(find_route(testbury, "HUBTST-Plat02-SB-northern").status, NO_INFO)

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
    exc = make_station("Example Cross", hub_naptan_id="HUBEXC",
                       naptan_id="940GZZLUEXC", tube=True, crossrail=True)
    sample_downs = make_station("Sample Downs", naptan_id="910GSMPLDN", overground=True)
    testbury = make_station("Testbury", hub_naptan_id="HUBTST",
                          naptan_id="940GZZLUTST", tube=True)
    return exc, sample_downs, testbury


class ImportTests(TestCase):
    def setUp(self):
        self.exc, self.sample_downs, self.testbury = make_fixture_stations()

    def test_stores_routes_for_main_records(self):
        result = import_feed(load_feed(fixture_zip()), min_stations=1)

        self.assertEqual(result.matched, 3)
        routes = routes_of(self.exc)
        self.assertEqual(routes.tfl_station_id, "HUBEXC")
        self.assertEqual(routes.published, PUBLISHED)
        self.assertIn(CENTRAL_EB, routes.data["platforms"])
        self.assertEqual(routes.data["format"], DATA_FORMAT)

    def test_child_record_codes_match_to_the_parent(self):
        # A name the TfL file doesn't use, so only the child's code can match.
        self.exc.name = "Example Cross (main record)"
        self.exc.hub_naptan_id = None
        self.exc.naptan_id = None
        self.exc.save()
        child = make_station("Example Cross", naptan_id="910GEXMPLX",
                             crossrail=True, parent_station=self.exc)

        import_feed(load_feed(fixture_zip()), min_stations=1)

        self.assertEqual(routes_of(self.exc).tfl_station_id, "HUBEXC")
        self.assertIsNone(routes_of(child))

    def test_name_match_when_no_code_matches(self):
        self.sample_downs.naptan_id = "SOMETHING-ELSE"
        self.sample_downs.save()

        import_feed(load_feed(fixture_zip()), min_stations=1)

        self.assertEqual(routes_of(self.sample_downs).tfl_station_id, "910GSMPLDN")

    def test_duplicate_records_store_on_the_lower_id_and_are_logged(self):
        duplicate = make_station("Testbury", hub_naptan_id="HUBTST", tube=True)

        with self.assertLogs("stations.step_free.importer", "WARNING") as logs:
            import_feed(load_feed(fixture_zip()), min_stations=1)

        self.assertEqual(routes_of(self.testbury).tfl_station_id, "HUBTST")
        self.assertIsNone(routes_of(duplicate))
        self.assertIn("share code HUBTST", "\n".join(logs.output))

    def test_too_few_stations_aborts_and_keeps_old_data(self):
        make_routes(self.exc, data={"old": True})

        with self.assertRaises(ImportAborted):
            import_feed(load_feed(fixture_zip()), min_stations=200)

        self.assertEqual(routes_of(self.exc).data, {"old": True})

    def test_two_tfl_stations_for_one_of_ours_are_merged(self):
        # TfL lists West Hampstead's Tube and Overground stations separately
        # under one hub code; we count them as one station. Simulate that by
        # giving one record codes from two fixture stations.
        Station.objects.filter(pk=self.testbury.pk).delete()
        self.exc.naptan_id = "940GZZLUTST"  # Testbury's Tube code
        self.exc.save()

        import_feed(load_feed(fixture_zip()), min_stations=1)

        stored = routes_of(self.exc)
        routes = stored.data
        self.assertEqual(stored.tfl_ids, ["HUBEXC", "HUBTST"])
        self.assertIn(CENTRAL_EB, routes["platforms"])
        self.assertIn("HUBTST-Plat01-NB-northern", routes["platforms"])
        self.assertIn("HUBEXC-Outside", routes["starts"])
        self.assertIn("HUBTST-Outside", routes["starts"])
        self.assertEqual(find_route(routes, CENTRAL_EB).status, OK)

    def test_stations_missing_from_the_file_are_cleared(self):
        gone = make_station("Nowhere Street", hub_naptan_id="HUBNOW", tube=True)
        make_routes(gone, data={"old": True}, tfl_ids=["HUBNOW"])

        result = import_feed(load_feed(fixture_zip()), min_stations=1)

        self.assertIsNone(routes_of(gone))
        self.assertEqual(result.cleared, 1)

    def test_reimport_keeps_broken_lifts_until_the_next_check(self):
        import_feed(load_feed(fixture_zip()), min_stations=1)
        StationRoutes.objects.filter(pk=self.exc.pk).update(broken_lift_ids=["HUBEXC-Lift-6"])

        import_feed(load_feed(fixture_zip()), min_stations=1)

        self.assertEqual(routes_of(self.exc).broken_lift_ids, ["HUBEXC-Lift-6"])


@patch("stations.step_free.importer.MIN_STATIONS", 1)
class ImportCommandTests(TestCase):
    def setUp(self):
        self.exc, _, self.testbury = make_fixture_stations()

    def run_command(self, data=None, *args):
        out = StringIO()
        with patch("stations.management.commands.import_station_routes.download_zip",
                   return_value=data or fixture_zip()):
            call_command("import_station_routes", *args, stdout=out)
        return out.getvalue()

    def test_downloads_and_imports(self):
        output = self.run_command()

        self.assertIn("Imported step-free routes for 3 stations", output)
        self.assertEqual(routes_of(self.exc).tfl_station_id, "HUBEXC")

    def test_unchanged_publish_time_skips_the_import(self):
        make_routes(self.exc, published=PUBLISHED, data={"format": DATA_FORMAT})

        output = self.run_command()

        self.assertIn("unchanged", output)
        self.assertIsNone(routes_of(self.testbury))

    def test_stored_data_in_an_older_format_is_reimported(self):
        # Same publish time, but data stored before the format existed.
        make_routes(self.exc, published=PUBLISHED, data={"tfl_ids": ["HUBEXC"]})

        output = self.run_command()

        self.assertIn("Imported step-free routes", output)
        self.assertEqual(routes_of(self.exc).data["format"], DATA_FORMAT)

    def test_same_day_republish_is_imported(self):
        make_routes(self.exc, published=PUBLISHED - timedelta(hours=2))

        output = self.run_command()

        self.assertIn("Imported step-free routes", output)
        self.assertEqual(routes_of(self.testbury).tfl_station_id, "HUBTST")

    def test_unmatched_stations_are_printed(self):
        self.testbury.delete()

        output = self.run_command()

        self.assertIn("TfL stations with no matching station: HUBTST", output)

    def test_broken_file_fails_and_keeps_data(self):
        make_routes(self.exc, data={"old": True}, published=PUBLISHED - timedelta(days=7))

        with self.assertRaises(CommandError):
            self.run_command(fixture_zip(skip={"Platforms.csv"}))

        self.assertEqual(routes_of(self.exc).data, {"old": True})


@override_settings(ROOT_URLCONF="updown.urls", FUNCTIONS_SECRET_KEY="secret")
class ImportEndpointTests(TestCase):
    def test_requires_the_functions_key(self):
        with patch("stations.views.call_command") as call_command_mock:
            denied = self.client.post("/functions/import_station_routes", {"key": "wrong"})
            allowed = self.client.post("/functions/import_station_routes", {"key": "secret"})

        self.assertEqual(denied.status_code, 404)
        self.assertEqual(allowed.status_code, 204)
        self.assertEqual(call_command_mock.call_args.args, ("import_station_routes",))

    def test_failed_import_is_a_server_error(self):
        with patch("stations.views.call_command", side_effect=CommandError("boom")):
            response = self.client.post("/functions/import_station_routes", {"key": "secret"})

        self.assertEqual(response.status_code, 500)


import requests

from stations.step_free.broken_lifts import lift_status_checked_at, update_broken_lifts


class FakeLiftResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.HTTPError("bad status")

    def json(self):
        return self._payload


def lift_feed(payload):
    return patch("stations.step_free.broken_lifts.requests.get",
                 return_value=FakeLiftResponse(payload))


class BrokenLiftTests(TestCase):
    def setUp(self):
        self.exc = make_station("Example Cross", tube=True)
        make_routes(self.exc, tfl_ids=["HUBEXC"])
        self.bank = make_station("Bank", tube=True)
        make_routes(self.bank, tfl_ids=["HUBOTH"], broken=["HUBOTH-Lift-1"])

    def test_stores_broken_lift_ids_per_station(self):
        payload = [
            {"stationUniqueId": "HUBEXC", "disruptedLiftUniqueIds": ["HUBEXC-Lift-6"],
             "message": "Example Cross: ..."},
            {"stationUniqueId": "HUBXYZ", "disruptedLiftUniqueIds": ["HUBXYZ-Lift-1"],
             "message": "Not one of ours"},
        ]
        with lift_feed(payload):
            self.assertTrue(update_broken_lifts())

        self.assertEqual(routes_of(self.exc).broken_lift_ids, ["HUBEXC-Lift-6"])
        self.assertEqual(routes_of(self.bank).broken_lift_ids, [])
        self.assertIsNotNone(lift_status_checked_at())

    def test_merged_station_gets_lifts_from_every_tfl_station(self):
        two_parts = make_station("Two Parts", tube=True)
        make_routes(two_parts, tfl_ids=["910GTWOPRT", "940GZZLUTWO"])
        payload = [
            {"stationUniqueId": "910GTWOPRT", "disruptedLiftUniqueIds": ["910GTWOPRT-Lift-1"]},
            {"stationUniqueId": "940GZZLUTWO", "disruptedLiftUniqueIds": ["940GZZLUTWO-Lift-2"]},
        ]
        with lift_feed(payload):
            update_broken_lifts()

        self.assertEqual(routes_of(two_parts).broken_lift_ids,
                         ["910GTWOPRT-Lift-1", "940GZZLUTWO-Lift-2"])

    def test_malformed_items_are_skipped(self):
        payload = [
            {"stationUniqueId": 940, "disruptedLiftUniqueIds": ["X"]},
            {"stationUniqueId": "HUBEXC", "disruptedLiftUniqueIds": "HUBEXC-Lift-6"},
            {"stationUniqueId": "HUBEXC", "disruptedLiftUniqueIds": [None, {"id": 1}, "HUBEXC-Lift-1"]},
            "not an item",
        ]
        with lift_feed(payload):
            self.assertTrue(update_broken_lifts())

        self.assertEqual(routes_of(self.exc).broken_lift_ids, ["HUBEXC-Lift-1"])

    def test_unchanged_rows_are_not_rewritten(self):
        payload = [{"stationUniqueId": "HUBOTH", "disruptedLiftUniqueIds": ["HUBOTH-Lift-1"]}]
        with lift_feed(payload), CaptureQueriesContext(connection) as queries:
            update_broken_lifts()

        writes = [q["sql"] for q in queries.captured_queries
                  if q["sql"].startswith("UPDATE") and "stationroutes" in q["sql"]]
        self.assertEqual(writes, [])
        self.assertIsNotNone(lift_status_checked_at())

    def test_failed_fetch_keeps_previous_data(self):
        with patch("stations.step_free.broken_lifts.requests.get",
                   side_effect=requests.ConnectionError("down")):
            self.assertFalse(update_broken_lifts())

        self.assertEqual(routes_of(self.bank).broken_lift_ids, ["HUBOTH-Lift-1"])
        self.assertIsNone(lift_status_checked_at())

    def test_unexpected_payload_keeps_previous_data(self):
        with lift_feed({"error": "nope"}):
            self.assertFalse(update_broken_lifts())

        self.assertEqual(routes_of(self.bank).broken_lift_ids, ["HUBOTH-Lift-1"])

    def test_database_error_does_not_raise(self):
        payload = [{"stationUniqueId": "HUBEXC", "disruptedLiftUniqueIds": ["HUBEXC-Lift-6"]}]
        with lift_feed(payload), patch.object(StationRoutes.objects, "values_list",
                                         side_effect=OperationalError("database is locked")):
            self.assertFalse(update_broken_lifts())


class UpdatePipelineTests(TestCase):
    def test_lift_status_runs_last_and_cannot_stop_the_update(self):
        calls = []
        command = "incidents.management.commands.update_incidents"
        with patch(f"{command}.check_tflv1", side_effect=lambda: calls.append("fetch")), \
                patch(f"{command}.consolidate_incidents",
                      side_effect=lambda: calls.append("consolidate") or []), \
                patch(f"{command}.update_last_updated",
                      side_effect=lambda: calls.append("last_updated")), \
                patch("stations.step_free.broken_lifts.requests.get",
                      side_effect=RuntimeError("unexpected")):
            call_command("update_incidents", stdout=StringIO())

        self.assertEqual(calls, ["fetch", "consolidate", "last_updated"])



from incidents.models import Incident
from stations.step_free.journey import build_journey, level_name


class JourneyTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.feed = load_feed(fixture_zip())
        cls.exc = cls.feed.stations[EXC]

    def journey(self, routes, platform_id, known=True):
        return build_journey(routes, find_route(routes, platform_id), known)

    @staticmethod
    def stops(level):
        return [item["name"] for item in level["items"] if item.get("stop")]

    def test_level_names(self):
        self.assertEqual(level_name(0), "Street level")
        self.assertEqual(level_name(-1), "1 level below street")
        self.assertEqual(level_name(-3), "3 levels below street")
        self.assertEqual(level_name(2), "2 levels above street")
        self.assertEqual(level_name(None), "No step-free route")

    def test_one_block_per_level_with_the_lift_between(self):
        journey = self.journey(self.exc, CENTRAL_EB)
        levels = journey["levels"]
        self.assertEqual([level["name"] for level in levels],
                         ["Street level", "1 level below street", "2 levels below street",
                          "3 levels below street"])
        self.assertEqual([self.stops(level) for level in levels],
                         [["North-west entrance"], ["North-west ticket hall"], ["Upper concourse"],
                          ["Central line – Eastbound Platform 2"]])
        self.assertEqual([(level["then"]["text"], level["then"]["detail"]) for level in levels[:-1]],
                         [("Lift A", "down 1 level"), ("Lift D", "down 1 level"), ("Lift F", "down 1 level")])
        self.assertIsNone(levels[-1]["then"])
        self.assertEqual({level["then"]["status"] for level in levels[:-1]}, {"Working"})

    def test_walks_on_one_level_stay_in_its_block(self):
        journey = self.journey(self.exc, "HUBEXC-Plat03-NB-northern")
        last = journey["levels"][-1]
        self.assertEqual(last["name"], "4 levels below street")
        self.assertEqual([item.get("name") or item["text"] for item in last["items"]],
                         ["Northern line platforms", "Walk", "Northern line – Northbound Platform 3"])
        # A lift that skips levels says how many.
        self.assertEqual(journey["levels"][-2]["then"]["detail"], "down 3 levels")

    def test_platform_stop_has_line_colour_and_boarding_notes(self):
        journey = self.journey(self.exc, "HUBEXC-Plat03-NB-northern")
        platform = journey["levels"][-1]["items"][-1]
        self.assertEqual(platform["kind"], "platform")
        self.assertEqual(platform["colour"], "#000000")
        self.assertEqual(platform["notes"], ["Step to train 0–50 mm", "Gap 0–85 mm"])

    def test_merged_platform_area_is_the_platform(self):
        sample_downs = self.feed.stations["910GSMPLDN"]
        journey = self.journey(sample_downs, "910GSMPLDN-Plat01-SB-london-overground|national-rail")
        self.assertEqual([self.stops(level) for level in journey["levels"]],
                         [["East entrance"], ["Weaver line – Southbound Platform 1"]])
        self.assertEqual(journey["levels"][0]["then"]["detail"], "up 2 levels")

    def test_lift_status_left_out_when_unknown(self):
        journey = self.journey(self.exc, CENTRAL_EB, known=False)
        self.assertEqual({level["then"]["status"] for level in journey["levels"][:-1]}, {""})

    def test_no_route_draws_nothing(self):
        journey = build_journey(self.exc, find_route(self.exc, CENTRAL_EB, ["HUBEXC-Lift-6"]))
        self.assertEqual(journey, {"levels": [], "place_panels": {}, "lift_panels": {}})
        self.assertEqual(build_journey(self.exc, None)["levels"], [])

    def test_every_place_and_lift_has_a_panel(self):
        journey = self.journey(self.exc, CENTRAL_EB)
        panels = {panel["id"] for panel in build_panels(self.exc, journey)}
        linked = [item["panel"] for level in journey["levels"] for item in level["items"] if item["panel"]]
        linked += [level["then"]["panel"] for level in journey["levels"] if level["then"]]
        self.assertEqual(len(linked), 7)
        self.assertEqual(set(linked), panels)

    def test_panel_ids_are_stable_and_hide_tfl_ids(self):
        journey = self.journey(self.exc, CENTRAL_EB)
        reordered = dict(self.exc, places=dict(reversed(list(self.exc["places"].items()))))
        again = self.journey(reordered, CENTRAL_EB)
        self.assertEqual(journey["place_panels"], again["place_panels"])
        for panel_id in list(journey["place_panels"].values()) + list(journey["lift_panels"].values()):
            self.assertRegex(panel_id, r"^(place|lift)-[0-9a-f]{10}$")


from stations.step_free.details import build_panels, facility_lines, station_summary


class DetailsTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.feed = load_feed(fixture_zip())
        cls.exc = cls.feed.stations[EXC]

    def panels(self, routes, broken=(), known=True):
        """{title: items} for a panel on every place and lift."""
        merged = routes.get("merged", {})
        every = {
            "place_panels": {pid: pid for pid, place in routes["places"].items()
                             if place["kind"] != "street" and pid not in merged},
            "lift_panels": {link["lift_id"]: link["lift_id"] for link in routes["links"]
                            if link["kind"] == "lift"},
        }
        return {panel["title"]: panel["items"] for panel in build_panels(routes, every, broken, known)}

    def test_platform_panel(self):
        items = self.panels(self.exc)["Northern line – Northbound Platform 3"]
        self.assertIn("Step-free from the street", items)
        self.assertIn("Step to the train: 0–50 mm", items)
        self.assertIn("Gap to the train: 0–85 mm", items)
        self.assertIn("Board at: 2 centre doors on car 5", items)
        self.assertIn("Use the lift by the north entrance for this platform.", items)

    def test_unsurveyed_platform_panel(self):
        panels = self.panels(self.feed.stations["HUBTST"])
        self.assertIn("TfL hasn't surveyed this platform for step-free access",
                      panels["Northern line – Southbound Platform 2"])

    def test_lift_panel_shows_stops_notes_and_status(self):
        items = self.panels(self.exc, ["HUBEXC-Lift-7"])["Lift G"]
        self.assertIn("Stops: Northern line platforms, Elizabeth line level", items)
        self.assertIn("Small lift", items)
        self.assertIn("Fits one wheelchair user at a time.", items)
        self.assertIn("Out of service now", items)

    def test_lift_status_unknown_when_stale(self):
        panels = self.panels(self.exc, (), known=False)
        self.assertIn("Live lift status unavailable", panels["Lift G"])

    def test_place_panel_lists_lifts_and_entrance_link(self):
        panels = self.panels(self.exc, ["HUBEXC-Lift-1"])
        items = panels["North-west ticket hall"]
        self.assertIn("1 level below street", items)
        self.assertIn("Lift A up to North-west entrance – out of service", items)
        self.assertIn("Lift C down to Elizabeth line level", items)
        self.assertIn("Linked to the street in TfL's data", panels["North-west entrance"])

    def test_street_linked_entrance_with_an_ent_code_is_linked(self):
        # Sample Downs' "Entr" is linked to the street; its code starts with
        # ENT, which puts it among the route starts too.
        sample_downs = self.feed.stations["910GSMPLDN"]
        self.assertIn("Linked to the street in TfL's data", self.panels(sample_downs)["South-west entrance"])

    def test_platform_panel_reflects_a_broken_lift(self):
        items = self.panels(self.exc, ["HUBEXC-Lift-6"])["Central line – Eastbound Platform 2"]  # Lift F
        self.assertIn("No step-free route right now: Lift F out of service", items)
        self.assertNotIn("Step-free from the street", items)

    def test_merged_platform_panel_lists_lifts_that_reach_it(self):
        sample_downs = self.feed.stations["910GSMPLDN"]
        items = self.panels(sample_downs, ["910GSMPLDN-Lift-1"])["Weaver line – Southbound Platform 1"]
        self.assertIn("Lift 1 down to East entrance – out of service", items)

    def test_platform_reached_only_through_an_unlinked_entrance_is_caveated(self):
        sample_downs = self.feed.stations["910GSMPLDN"]
        self.assertIn("Step-free only through an entrance TfL's data doesn't link to the street, "
                      "which may have steps", self.panels(sample_downs)["Weaver line – Southbound Platform 1"])
        summary = station_summary(sample_downs)
        self.assertEqual(summary["step_free"], [])
        self.assertEqual(summary["step_free_unlinked"], ["Weaver line – Southbound Platform 1"])

    def test_toilet_with_unknown_gate_position(self):
        lines = facility_lines({"blue_badge_spaces": None, "taxi_rank": False, "toilets": [{
            "location": "Concourse", "inside_gates": None, "radar_key": False, "fee": False,
            "baby_changing": False, "hours": ""}]})
        self.assertEqual(lines, ["Accessible toilet: Concourse"])

    def test_unlinked_entrance_says_so(self):
        sample_downs = self.feed.stations["910GSMPLDN"]
        self.assertIn("Not linked to the street in TfL's data, so it may have steps",
                      self.panels(sample_downs)["East entrance"])

    def test_station_summary(self):
        summary = station_summary(self.exc)
        self.assertIn("Central line – Eastbound Platform 2", summary["step_free"])
        self.assertEqual(summary["not_step_free"], [])
        testbury = station_summary(self.feed.stations["HUBTST"])
        self.assertEqual(testbury["step_free"], [])
        self.assertEqual(testbury["not_step_free"], ["Northern line – Northbound Platform 1"])
        self.assertEqual(testbury["no_information"], ["Northern line – Southbound Platform 2"])


@override_settings(ROOT_URLCONF="updown.urls")
class StepFreePageTests(TestCase):
    def setUp(self):
        self.exc, self.sample_downs, self.testbury = make_fixture_stations()
        import_feed(load_feed(fixture_zip()), min_stations=1)

    def set_lift_status(self, broken, age=timedelta(minutes=1)):
        StationRoutes.objects.filter(pk=self.exc.pk).update(broken_lift_ids=broken)
        LiftStatusCheck.objects.update_or_create(
            pk=1, defaults={"checked_at": timezone.now() - age})

    def test_index_lists_stations_and_redirects_to_choice(self):
        response = self.client.get("/step-free/")
        self.assertContains(response, "Example Cross")
        self.assertContains(response, 'name="robots" content="noindex"')

        response = self.client.get("/step-free/", {"station": "example-cross"})
        self.assertRedirects(response, "/step-free/example-cross/",
                             fetch_redirect_response=False)

    def test_choice_with_no_letters_shows_the_index(self):
        for value in ("!!!", " ", "東京"):
            with self.subTest(value=value):
                response = self.client.get("/step-free/", {"station": value})
                self.assertEqual(response.status_code, 200)
                self.assertContains(response, "Choose a station")

    def test_station_page_shows_the_route_level_by_level(self):
        self.set_lift_status([])
        response = self.client.get("/step-free/example-cross/",
                                   {"platform": CENTRAL_EB})

        self.assertContains(response, 'class="sf-journey"')
        self.assertContains(response, "3 levels below street")
        self.assertContains(response, "Lift F")
        self.assertContains(response, '<span class="sf-pill ok">Working</span>')
        self.assertContains(response, 'href="/step-free/"')
        self.assertContains(response, 'name="robots" content="noindex"')
        # No route data reaches the page: no place ids, lift ids or JSON keys.
        # (base.html's site-wide jQuery scripts are unrelated and stay.)
        content = response.content.decode()
        self.assertNotIn("HUBEXC-1000001-", content)
        self.assertNotIn("HUBEXC-Lift-", content)
        self.assertNotIn('"places"', content)

    def test_broken_lift_blocks_route_and_is_named(self):
        self.set_lift_status(["HUBEXC-Lift-6"])
        response = self.client.get("/step-free/example-cross/",
                                   {"platform": CENTRAL_EB})

        self.assertContains(response, "No step-free route to this platform right now")
        self.assertContains(response, "Out of service: Lift F.")
        self.assertContains(response, "Out of service now: Lift F.")

    def test_stale_lift_status_is_unavailable(self):
        self.set_lift_status(["HUBEXC-Lift-6"], age=timedelta(minutes=31))
        response = self.client.get("/step-free/example-cross/",
                                   {"platform": CENTRAL_EB})

        self.assertContains(response, "Live lift status is unavailable")
        self.assertContains(response, 'class="sf-journey"')
        self.assertNotContains(response, "out of service")
        self.assertNotContains(response, '<span class="sf-pill ok">Working</span>')

    def test_no_route_in_tfl_data(self):
        response = self.client.get("/step-free/testbury/",
                                   {"platform": "HUBTST-Plat01-NB-northern"})
        # Literal template text, so Django does not escape the apostrophe.
        self.assertContains(response, "TfL's data shows no step-free route to this platform")

    def test_places_open_detail_panels_without_javascript(self):
        response = self.client.get("/step-free/example-cross/",
                                   {"platform": "HUBEXC-Plat03-NB-northern"})
        content = response.content.decode()
        self.assertRegex(content, r'href="#place-[0-9a-f]{10}"')
        self.assertRegex(content, r'href="#lift-[0-9a-f]{10}"')
        # tabindex lets the browser move focus to the opened panel.
        self.assertRegex(content, r'<section class="detail-panel" id="place-[0-9a-f]{10}" tabindex="-1"')
        self.assertIn('class="sf-i"', content)
        self.assertIn("Board at: 2 centre doors on car 5", content)
        self.assertIn(":target", content)
        # Still no route data in the page.
        self.assertNotIn("HUBEXC-1000001-", content)
        self.assertNotIn("HUBEXC-Lift-", content)

    def test_summary_and_facilities(self):
        response = self.client.get("/step-free/example-cross/")
        self.assertContains(response, "Step-free from the street to:")
        self.assertContains(response, "Blue Badge parking (2 spaces)")
        self.assertContains(response, "Taxi rank outside")
        self.assertContains(response, "Accessible toilet: Ticket hall, inside the ticket gates, "
                                      "RADAR key, baby changing, Daily until 23:00")

    def test_station_reachable_only_through_an_unlinked_entrance(self):
        response = self.client.get("/step-free/sample-downs/")
        self.assertContains(response, "Possibly step-free")
        self.assertNotContains(response, "No step-free access at this station")

    def test_station_with_no_survey_information_does_not_claim_no_access(self):
        stored = routes_of(self.testbury)
        for platform in stored.data["platforms"].values():
            platform["route_info"] = False
        stored.save()
        response = self.client.get("/step-free/testbury/")
        self.assertContains(response, "TfL hasn't published step-free information for this station")
        self.assertNotContains(response, "No step-free access at this station")

    def test_station_with_no_step_free_route_says_so(self):
        response = self.client.get("/step-free/testbury/")
        self.assertContains(response, "No step-free access at this station")

    def test_unsurveyed_platform_says_so(self):
        response = self.client.get("/step-free/testbury/",
                                   {"platform": "HUBTST-Plat02-SB-northern"})
        self.assertContains(response, "TfL hasn't published step-free route information")

    def test_issues_on_a_duplicate_record_of_the_station_are_shown(self):
        duplicate = make_station("Example Cross", tube=True)
        Incident.objects.create(station=duplicate, information=False,
                                text="Lift out of order at the other record.")
        response = self.client.get("/step-free/example-cross/")
        self.assertContains(response, "Lift out of order at the other record.")

    def test_unlinked_entrance_route_carries_a_warning(self):
        response = self.client.get(
            "/step-free/sample-downs/",
            {"platform": "910GSMPLDN-Plat01-SB-london-overground|national-rail"})
        self.assertContains(response, "Check before you travel")

    def test_current_issues_are_shown_with_advice_formatted(self):
        Incident.objects.create(station=self.exc, information=False,
                                text="Lift out of order.<p><i>Use Lift B instead.</i></p>")
        response = self.client.get("/step-free/example-cross/")
        self.assertContains(response, "Lift out of order.")
        self.assertContains(response, "Use Lift B instead.")
        self.assertNotContains(response, "&lt;p&gt;")

    def test_bad_platform_is_ignored(self):
        response = self.client.get("/step-free/example-cross/", {"platform": "nope"})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Choose a platform")
        self.assertNotContains(response, 'class="sf-journey"')

    def test_unknown_station_is_404(self):
        self.assertEqual(self.client.get("/step-free/nowhere/").status_code, 404)

    def test_different_stations_with_one_name_each_get_a_page(self):
        # Like the two Bethnal Greens (Central line; Weaver line): separate
        # stations that happen to share a name.
        feed = load_feed(fixture_zip())
        tube = make_station("Bethnal Green", tube=True)
        make_routes(tube, data=feed.stations["HUBEXC"], tfl_ids=["HUBEXC"])
        overground = make_station("Bethnal Green", overground=True)
        make_routes(overground, data=feed.stations["910GSMPLDN"], tfl_ids=["910GSMPLDN"])

        index = self.client.get("/step-free/")
        self.assertContains(index, "Bethnal Green (Central line, Elizabeth line, Northern line)")
        self.assertContains(index, "Bethnal Green (Weaver line)")

        tube_page = self.client.get("/step-free/bethnal-green-central-line/")
        overground_page = self.client.get("/step-free/bethnal-green-weaver-line/")
        self.assertContains(tube_page, "Central line – Eastbound Platform 2")
        self.assertContains(overground_page, "Weaver line – Southbound Platform 1")
        # The plain name still works, for links made before the suffix.
        self.assertEqual(self.client.get("/step-free/bethnal-green/").status_code, 200)

    def test_station_without_route_data(self):
        make_station("Oval", hub_naptan_id="HUBOVL", tube=True)
        response = self.client.get("/step-free/oval/")
        self.assertContains(response, "TfL has not published step-free route data for this station.")
