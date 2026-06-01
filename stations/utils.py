import requests
from django.conf import settings
from django.db.models import Q

from stations.models import Station


def cleanup_station_name(station_name):
    to_remove = [
        "Rail Station",
        "Underground Station",
        "Underground Stn",
        "Overground Station",
        "DLR Station",
        "Station",
        "(London)",
        "(",
        ")",
        "Dist&Picc Line",
        "Circle Line",
        "Bakerloo Line",
        "Central Line",
        "H&C Line-Underground",
        "Crossrail",
    ]

    for item in to_remove:
        station_name = station_name.replace(item, "")

    # Kensington (Olympia) is the only station with parentheses in its name on the tube map.
    if station_name == "Kensington Olympia":
        station_name = "Kensington (Olympia)"

    return station_name.strip()


def update_station_list():
    modes = [
        "tube",
        "dlr",
        "overground",
        "elizabeth-line",
        "national-rail",
    ]

    station_status_uri = f"https://api.tfl.gov.uk/StopPoint/Mode/%s?app_id={settings.TFL_API_ID}&app_key={settings.TFL_API_KEY}&page=%d"

    MODE_TRANSFORM = {
        "tube": "tube",
        "dlr": "dlr",
        "overground": "overground",
        "elizabeth-line": "crossrail",
        "national-rail": "national_rail",
    }

    try:
        for mode in modes:
            end_of_pages = False
            page = 1
            data_points_returned = 0

            while not end_of_pages:
                url = station_status_uri % (mode, page)

                r = requests.get(url)
                data = r.json()

                for stop_point in data.get("stopPoints", []):
                    station_name = cleanup_station_name(stop_point["commonName"])

                    station, _ = Station.objects.get_or_create(
                        name=station_name,
                        naptan_id=stop_point.get("stationNaptan"),
                        hub_naptan_id=stop_point.get("hubNaptanCode"),
                    )
                    setattr(station, MODE_TRANSFORM[mode], True)

                    parent_station = station
                    if station.hub_naptan_id:
                        try:
                            parent_station = Station.objects.filter(
                                hub_naptan_id=station.hub_naptan_id
                            ).first()
                        except Station.DoesNotExist:
                            pass
                    station.parent_station = parent_station
                    station.save()

                data_points_returned += data.get("pageSize")
                if data.get("total") > data_points_returned:
                    page += 1
                else:
                    end_of_pages = True

    except Exception as e:
        raise


def find_station(possible_name):
    clean_station_name = cleanup_station_name(possible_name).lower()

    station = Station.objects.filter(name__iexact=clean_station_name).first()

    return station


def find_station_from_naptan(naptan_id):
    station = Station.objects.filter(
        Q(naptan_id=naptan_id) | Q(hub_naptan_id=naptan_id)
    ).first()

    return station
