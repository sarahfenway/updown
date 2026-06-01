from unittest.mock import patch
from io import StringIO

from django.core.management import CommandError, call_command
from django.test import SimpleTestCase, TestCase

from stations.models import Station
from stations.utils import (
    cleanup_station_name,
    find_station,
    find_station_from_naptan,
    update_station_list,
)


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class StationUtilityTests(SimpleTestCase):
    def test_cleanup_station_name_removes_suffixes_and_preserves_kensington(self):
        self.assertEqual(cleanup_station_name("Bank Underground Station"), "Bank")
        self.assertEqual(
            cleanup_station_name("Kensington (Olympia) Underground Station"),
            "Kensington Olympia",
        )
        self.assertEqual(cleanup_station_name("Paddington Crossrail"), "Paddington")


class StationQueryTests(TestCase):
    def test_alternate_names_round_trip_as_plain_list(self):
        station = Station.objects.create(
            name="Baker Street",
            notes="",
            alternate_names=["Baker St", "Baker Street Underground"],
            tube=True,
            national_rail=False,
        )

        station.refresh_from_db()

        self.assertEqual(
            station.alternate_names,
            ["Baker St", "Baker Street Underground"],
        )

    def test_find_station_and_find_station_from_naptan(self):
        parent = Station.objects.create(
            name="Bank",
            notes="",
            naptan_id="BNK",
            hub_naptan_id="HUBBNK",
            tube=True,
            national_rail=False,
        )
        parent.parent_station = parent
        parent.save(update_fields=["parent_station"])
        child = Station.objects.create(
            name="Bank Platforms",
            notes="",
            naptan_id="BNK1",
            hub_naptan_id="HUBBNK",
            tube=True,
            national_rail=False,
            parent_station=parent,
        )

        self.assertEqual(find_station("Bank Underground Station"), parent)
        self.assertEqual(find_station_from_naptan("BNK1"), child)
        self.assertEqual(find_station_from_naptan("HUBBNK"), parent)
        self.assertIsNone(find_station("Made Up Station"))

    def test_update_station_list_creates_pages_and_reuses_parent_station(self):
        payloads = {
            ("tube", 1): {
                "stopPoints": [
                    {
                        "commonName": "Bank Underground Station",
                        "stationNaptan": "BNK",
                        "hubNaptanCode": "HUBBNK",
                    }
                ],
                "pageSize": 1,
                "total": 2,
            },
            ("tube", 2): {
                "stopPoints": [
                    {
                        "commonName": "Monument Underground Station",
                        "stationNaptan": "MON",
                        "hubNaptanCode": "HUBBNK",
                    }
                ],
                "pageSize": 1,
                "total": 2,
            },
            ("dlr", 1): {"stopPoints": [], "pageSize": 0, "total": 0},
            ("overground", 1): {
                "stopPoints": [
                    {
                        "commonName": "Bank Underground Station",
                        "stationNaptan": "BNK",
                        "hubNaptanCode": "HUBBNK",
                    }
                ],
                "pageSize": 1,
                "total": 1,
            },
            ("elizabeth-line", 1): {"stopPoints": [], "pageSize": 0, "total": 0},
            ("national-rail", 1): {"stopPoints": [], "pageSize": 0, "total": 0},
        }

        def fake_get(url):
            mode = url.split("/Mode/")[1].split("?")[0]
            page = int(url.split("&page=")[1])
            return FakeResponse(payloads[(mode, page)])

        with patch("stations.utils.requests.get", side_effect=fake_get):
            update_station_list()

        bank = Station.objects.get(name="Bank")
        monument = Station.objects.get(name="Monument")

        self.assertTrue(bank.tube)
        self.assertTrue(bank.overground)
        self.assertEqual(Station.objects.filter(name="Bank").count(), 1)
        self.assertEqual(bank.parent_station, bank)
        self.assertEqual(monument.parent_station, bank)

    def test_update_station_list_sets_crossrail_and_national_rail_flags(self):
        payloads = {
            ("tube", 1): {"stopPoints": [], "pageSize": 0, "total": 0},
            ("dlr", 1): {"stopPoints": [], "pageSize": 0, "total": 0},
            ("overground", 1): {"stopPoints": [], "pageSize": 0, "total": 0},
            ("elizabeth-line", 1): {
                "stopPoints": [
                    {
                        "commonName": "Paddington Crossrail",
                        "stationNaptan": "PAD",
                        "hubNaptanCode": "HUBPAD",
                    }
                ],
                "pageSize": 1,
                "total": 1,
            },
            ("national-rail", 1): {
                "stopPoints": [
                    {
                        "commonName": "Paddington Rail Station",
                        "stationNaptan": "PAD",
                        "hubNaptanCode": "HUBPAD",
                    }
                ],
                "pageSize": 1,
                "total": 1,
            },
        }

        def fake_get(url):
            mode = url.split("/Mode/")[1].split("?")[0]
            page = int(url.split("&page=")[1])
            return FakeResponse(payloads[(mode, page)])

        with patch("stations.utils.requests.get", side_effect=fake_get):
            update_station_list()

        paddington = Station.objects.get(name="Paddington")
        self.assertTrue(paddington.crossrail)
        self.assertTrue(paddington.national_rail)
        self.assertEqual(paddington.parent_station, paddington)

    def test_update_stations_command_runs_and_wraps_failures(self):
        stdout = StringIO()

        with patch(
            "stations.management.commands.update_stations.update_station_list"
        ) as update_station_list_mock:
            call_command("update_stations", stdout=stdout)

        update_station_list_mock.assert_called_once_with()
        self.assertIn("Successfully updated station list", stdout.getvalue())

        with patch(
            "stations.management.commands.update_stations.update_station_list",
            side_effect=Exception("boom"),
        ):
            with self.assertRaises(CommandError):
                call_command("update_stations")
