"""Parse TfL's station data zip into per-station step-free route data.

The zip is ``https://api.tfl.gov.uk/stationdata/tfl-stationdata-detailed.zip``.
It describes each station as places joined by step-free links only (level
walkways, ramps and lifts). This module is pure: bytes in, plain dicts out.
"""

import csv
import io
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime

from .labels import is_dropped, place_label
from .layout import compute_layout

REQUIRED_FILES = (
    "FeedInfo.csv",
    "Stations.csv",
    "StationPoints.csv",
    "SameLevelPaths.csv",
    "RampRoutes.csv",
    "Lifts.csv",
    "Platforms.csv",
    "PlatformServices.csv",
)

# Mode tokens that appear at the end of a platform's UniqueId, e.g.
# "HUBTCR-Plat02-EB-central" or "...-london-overground|national-rail".
COVERED_MODES = {
    "bakerloo", "central", "circle", "district", "hammersmith-city", "jubilee",
    "metropolitan", "northern", "piccadilly", "victoria", "waterloo-city",
    "dlr", "london-overground", "elizabeth",
}

LINE_NAMES = {
    "bakerloo": "Bakerloo line",
    "central": "Central line",
    "circle": "Circle line",
    "district": "District line",
    "hammersmith-city": "Hammersmith & City line",
    "jubilee": "Jubilee line",
    "metropolitan": "Metropolitan line",
    "northern": "Northern line",
    "piccadilly": "Piccadilly line",
    "victoria": "Victoria line",
    "waterloo-city": "Waterloo & City line",
    "dlr": "DLR",
    "elizabeth": "Elizabeth line",
    "liberty": "Liberty line",
    "lioness": "Lioness line",
    "mildmay": "Mildmay line",
    "suffragette": "Suffragette line",
    "weaver": "Weaver line",
    "windrush": "Windrush line",
    "london-overground": "Overground",
}


class FeedError(Exception):
    """The file is missing, unreadable or incomplete."""


@dataclass
class Feed:
    published: date
    # tfl station id -> route dict (see build_feed for its shape)
    stations: dict = field(default_factory=dict)
    # tfl station id -> every NaPTAN-style code that identifies it
    codes: dict = field(default_factory=dict)


def read_zip(data):
    """Return {file name: [row dict, ...]} for the files we need."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise FeedError("not a zip file") from exc
    names = set(archive.namelist())
    missing = [name for name in REQUIRED_FILES if name not in names]
    if missing:
        raise FeedError(f"missing {', '.join(missing)}")
    tables = {}
    for name in REQUIRED_FILES:
        text = archive.read(name).decode("utf-8-sig")
        tables[name] = list(csv.DictReader(io.StringIO(text)))
    return tables


def read_published(tables):
    rows = tables["FeedInfo.csv"]
    try:
        return datetime.fromisoformat(rows[0]["FeedStartDate"].strip()).date()
    except (IndexError, KeyError, ValueError) as exc:
        raise FeedError("FeedInfo.csv has no readable FeedStartDate") from exc


def load_feed(data):
    """Parse the zip bytes into a Feed. Raises FeedError."""
    return build_feed(read_zip(data))


def _true(value):
    return (value or "").strip().upper() == "TRUE"


def _platform_modes(platform_id):
    parts = platform_id.split("-", 3)
    return set(parts[3].split("|")) if len(parts) == 4 else set()


def _range(low, high):
    low, high = (low or "").strip(), (high or "").strip()
    if low and high:
        return f"{low}–{high} mm"
    return None


def build_feed(tables):
    try:
        return _build_feed(tables)
    except (KeyError, ValueError) as exc:
        raise FeedError(f"unexpected file contents: {exc!r}") from exc


def _build_feed(tables):
    published = read_published(tables)
    station_rows = {row["UniqueId"].strip(): row for row in tables["Stations.csv"]}

    # Every node id belongs to exactly one station.
    node_station = {}
    places = {}  # station id -> {place id: place}

    for sid, row in station_rows.items():
        outside = (row.get("OutsideStationUniqueId") or "").strip() or f"{sid}-Outside"
        places[sid] = {outside: {"name": "Street", "level": 0, "kind": "street",
                                 "lon": None, "code": "", "area_id": ""}}
        node_station[outside] = sid

    for row in tables["StationPoints.csv"]:
        sid = row["StationUniqueId"].strip()
        code = row["AreaName"].strip()
        if sid not in places or is_dropped(code):
            continue
        pid = row["UniqueId"].strip()
        lon = row.get("Lon", "").strip()
        places[sid][pid] = {
            "name": None, "level": int(row["Level"].strip() or 0), "kind": "place",
            "lon": float(lon) if lon else None, "code": code,
            "area_id": row.get("AreaId", "").strip(),
        }
        node_station[pid] = sid

    services = {}
    for row in tables["PlatformServices.csv"]:
        services.setdefault(row["PlatformUniqueId"].strip(), []).append(row)

    platforms = {}  # station id -> {platform id: platform}
    codes = {sid: {sid} for sid in station_rows}
    for sid, row in station_rows.items():
        if (row.get("HubNaptanCode") or "").strip():
            codes[sid].add(row["HubNaptanCode"].strip())

    for row in tables["Platforms.csv"]:
        sid = row["StationUniqueId"].strip()
        pid = row["UniqueId"].strip()
        if sid not in places:
            continue
        if (row.get("PlatformNaptanCode") or "").strip():
            codes[sid].add(row["PlatformNaptanCode"].strip())
        for service in services.get(pid, []):
            if (service.get("StopAreaNaptanCode") or "").strip():
                codes[sid].add(service["StopAreaNaptanCode"].strip())
        if not _true(row["IsCustomerFacing"]) or not (_platform_modes(pid) & COVERED_MODES):
            continue
        rows = services.get(pid, [])
        lines = []
        for service in rows:
            line = service["Line"].strip()
            if line in LINE_NAMES and line != "national-rail" and LINE_NAMES[line] not in lines:
                lines.append(LINE_NAMES[line])
        if not lines:
            lines = [LINE_NAMES[m] for m in sorted(_platform_modes(pid) & COVERED_MODES)]
        name = row["FriendlyName"].strip()
        step = gap = None
        manual_ramp = False
        for service in rows:
            step = step or _range(service.get("MinStep"), service.get("MaxStep"))
            gap = gap or _range(service.get("MinGap"), service.get("MaxGap"))
            manual_ramp = manual_ramp or _true(service.get("LevelAccessByManualRamp"))
        platforms.setdefault(sid, {})[pid] = {
            "name": name,
            "label": f"{' / '.join(lines)} – {name}",
            "lines": lines,
            "step": step,
            "gap": gap,
            "manual_ramp": manual_ramp,
            "entrance": (row.get("AccessibleEntranceName") or "").strip() or None,
        }
        places[sid][pid] = {"name": name, "level": None, "kind": "platform",
                            "lon": None, "code": "", "area_id": ""}
        node_station[pid] = sid

    links = {sid: {} for sid in places}

    def add(a, b, kind, lift=None):
        a, b = a.strip(), b.strip()
        sid = node_station.get(a)
        if sid is None or node_station.get(b) != sid or a == b:
            return
        lift_id = lift["id"] if lift else None
        key = (min(a, b), max(a, b), kind, lift_id)
        if key not in links[sid]:
            link = {"a": a, "b": b, "kind": kind}
            if lift:
                link.update(lift_id=lift["id"], lift_name=lift["name"], small=lift["small"])
            links[sid][key] = link

    for row in tables["SameLevelPaths.csv"]:
        add(row["From"], row["To"], "walkway")
    for row in tables["RampRoutes.csv"]:
        add(row["From"], row["To"], "ramp")
    for row in tables["Lifts.csv"]:
        lift = {
            "id": row["LiftUniqueId"].strip(),
            "name": (row.get("FriendlyName") or "").strip() or (row.get("LiftName") or "").strip() or "Lift",
            "small": _true(row.get("LimitedCapacityLift")),
        }
        chain = [row["FromAreas"], row["IntermediateAreas"], row["IntermediateAreas2"], row["ToAreas"]]
        chain = [[a for a in group.split("|") if a.strip()] for group in chain if (group or "").strip()]
        for group_from, group_to in zip(chain, chain[1:]):
            for a in group_from:
                for b in group_to:
                    add(a, b, "lift", lift)

    stations = {}
    for sid, station_places in places.items():
        if not platforms.get(sid):
            continue
        station_links = list(links[sid].values())
        outside = next(pid for pid, p in station_places.items() if p["kind"] == "street")
        neighbours = {}
        for link in station_links:
            neighbours.setdefault(link["a"], []).append((link["b"], link["kind"]))
            neighbours.setdefault(link["b"], []).append((link["a"], link["kind"]))

        # Platforms take the level and longitude of the place they open onto.
        for pid in platforms[sid]:
            near = [station_places[n] for n, kind in neighbours.get(pid, [])
                    if kind == "walkway" and station_places[n]["kind"] == "place"]
            station_places[pid]["level"] = near[0]["level"] if near else 0
            station_places[pid]["lon"] = near[0]["lon"] if near else None

        # Places reached straight from the street are entrances.
        for other, kind in neighbours.get(outside, []):
            if station_places[other]["kind"] == "place":
                station_places[other]["kind"] = "entrance"

        starts = [outside] + sorted(
            pid for pid, p in station_places.items()
            if p["kind"] in ("place", "entrance") and p["level"] == 0
            and p["code"].upper().lstrip("I").startswith("ENT")
        )
        for pid in starts[1:]:
            station_places[pid]["kind"] = "entrance"

        _name_places(station_places)
        stations[sid] = {
            "tfl_id": sid,
            "tfl_ids": [sid],
            "name": station_rows[sid]["Name"].strip(),
            "places": station_places,
            "links": station_links,
            "platforms": platforms[sid],
            "starts": starts,
            "layout": compute_layout(station_places),
        }

    return Feed(published=published, stations=stations,
                codes={sid: codes[sid] for sid in stations})


def merge_routes(routes_list):
    """One station's route data from several TfL stations.

    TfL sometimes lists one interchange as separate stations sharing a hub
    code (West Hampstead's Tube and Overground). We count it as one station,
    so their places, links, platforms and starts are combined. They share no
    links, so each part keeps its own street node and routes.
    """
    routes_list = sorted(routes_list, key=lambda routes: routes["tfl_id"])
    places, links, platforms, starts = {}, [], {}, []
    for routes in routes_list:
        places.update(routes["places"])
        links.extend(routes["links"])
        platforms.update(routes["platforms"])
        starts.extend(routes["starts"])
    _name_places(places)
    return {
        "tfl_id": routes_list[0]["tfl_id"],
        "tfl_ids": [routes["tfl_id"] for routes in routes_list],
        "name": routes_list[0]["name"],
        "places": places,
        "links": links,
        "platforms": platforms,
        "starts": starts,
        "layout": compute_layout(places),
    }


def _name_places(station_places):
    """Give every non-platform place a readable, unique name."""
    groups = {}
    for pid, place in station_places.items():
        if place["kind"] in ("street", "platform"):
            continue
        label = place_label(place["code"])
        if label is None:
            label = f"Entrance ({place['code']})" if place["kind"] == "entrance" else place["code"]
            place["name"] = label
            continue
        place["name"] = label
        groups.setdefault(label, []).append(pid)
    for label, ids in groups.items():
        if len(ids) < 2:
            continue
        codes = [station_places[pid]["code"] for pid in ids]
        for pid in ids:
            place = station_places[pid]
            suffix = place["code"]
            if codes.count(place["code"]) > 1:
                suffix = f"{place['code']} {place['area_id']}"
            place["name"] = f"{label} ({suffix})"
