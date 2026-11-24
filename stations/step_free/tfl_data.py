"""Parse TfL's station data zip into per-station step-free route data.

The zip is ``https://api.tfl.gov.uk/stationdata/tfl-stationdata-detailed.zip``.
It describes each station as places joined by step-free links only (level
walkways, ramps and lifts). This module is pure: bytes in, plain dicts out.
"""

import copy
import csv
import io
import math
import re
import zipfile
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

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

# Read when present; the pages just show less without them.
OPTIONAL_FILES = ("Toilets.csv",)

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


MAX_UNPACKED_BYTES = 50 * 1024 * 1024

# Bump when the stored route data changes shape or meaning, so the import
# re-runs even if TfL's file hasn't changed (stored data without it, or
# with an older number, is out of date).
DATA_FORMAT = 3


class FeedError(Exception):
    """The file is missing, unreadable or incomplete."""


@dataclass
class Feed:
    published: datetime  # FeedStartDate, timezone-aware
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
    # The real file unpacks to about 1.3 MB; refuse anything absurd before
    # reading it into memory.
    wanted = REQUIRED_FILES + tuple(n for n in OPTIONAL_FILES if n in names)
    if sum(archive.getinfo(name).file_size for name in wanted) > MAX_UNPACKED_BYTES:
        raise FeedError("station data files are unexpectedly large")
    tables = {name: [] for name in OPTIONAL_FILES}
    for name in REQUIRED_FILES + tuple(n for n in OPTIONAL_FILES if n in names):
        try:
            text = archive.read(name).decode("utf-8-sig")
        except (zipfile.BadZipFile, zlib.error, EOFError, NotImplementedError,
                RuntimeError, UnicodeDecodeError) as exc:
            raise FeedError(f"can't read {name} ({type(exc).__name__})") from exc
        tables[name] = list(csv.DictReader(io.StringIO(text)))
    return tables


def read_published(tables):
    rows = tables["FeedInfo.csv"]
    try:
        published = datetime.fromisoformat(rows[0]["FeedStartDate"].strip())
    except (IndexError, KeyError, ValueError, AttributeError, TypeError) as exc:
        raise FeedError("FeedInfo.csv has no readable FeedStartDate") from exc
    # The whole timestamp, not just the date, so a same-day republish is seen.
    return published if published.tzinfo else published.replace(tzinfo=timezone.utc)


def load_feed(data):
    """Parse the zip bytes into a Feed. Raises FeedError."""
    return build_feed(read_zip(data))


def _true(value):
    return (value or "").strip().upper() == "TRUE"


def _true_or_none(value):
    value = (value or "").strip().upper()
    return True if value == "TRUE" else False if value == "FALSE" else None


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
    except (KeyError, ValueError, AttributeError, TypeError) as exc:
        # AttributeError/TypeError: csv fills a short row's missing fields
        # with None.
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
        lon = (row.get("Lon") or "").strip()
        lat = (row.get("Lat") or "").strip()
        places[sid][pid] = {
            "name": None, "level": int((row["Level"] or "").strip() or 0), "kind": "place",
            "lon": float(lon) if lon else None, "lat": float(lat) if lat else None, "code": code,
            "area_id": (row.get("AreaId") or "").strip(),
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
        step = gap = boarding = None
        manual_ramp = False
        notes = []
        for service in rows:
            step = step or _range(service.get("MinStep"), service.get("MaxStep"))
            gap = gap or _range(service.get("MinGap"), service.get("MaxGap"))
            manual_ramp = manual_ramp or _true(service.get("LevelAccessByManualRamp"))
            boarding = boarding or (service.get("LocationOfLevelAccess") or "").strip() or None
            note = (service.get("AdditionalAccessibilityInformation") or "").strip()
            if note and note not in notes:
                notes.append(note)
        platforms.setdefault(sid, {})[pid] = {
            "name": name,
            "label": f"{' / '.join(lines)} – {name}",
            "lines": lines,
            "step": step,
            "gap": gap,
            "manual_ramp": manual_ramp,
            # Where on the train level access is ("2 centre doors on car 5").
            "boarding": boarding,
            # TfL's written notes for the platform.
            "notes": notes,
            "entrance": (row.get("AccessibleEntranceName") or "").strip() or None,
            # False where TfL hasn't surveyed the platform: "no route" then
            # means "no information", not "not step-free".
            "route_info": _true(row.get("HasStepFreeRouteInformation")),
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
                link.update(lift_id=lift["id"], lift_name=lift["name"], small=lift["small"],
                            notes=lift["notes"])
            links[sid][key] = link

    for row in tables["SameLevelPaths.csv"]:
        add(row["From"], row["To"], "walkway")
    for row in tables["RampRoutes.csv"]:
        add(row["From"], row["To"], "ramp")
    for number, row in enumerate(tables["Lifts.csv"]):
        lift = {
            # A blank id would read as "not in a lift" to the route search.
            "id": (row["LiftUniqueId"] or "").strip() or f"{row['StationUniqueId'].strip()}-lift-row{number}",
            "name": (row.get("FriendlyName") or "").strip() or (row.get("LiftName") or "").strip() or "Lift",
            "small": _true(row.get("LimitedCapacityLift")),
            "notes": (row.get("LiftNotes") or "").strip() or None,
        }
        chain = [row["FromAreas"], row["IntermediateAreas"], row["IntermediateAreas2"], row["ToAreas"]]
        chain = [[a for a in group.split("|") if a.strip()] for group in chain if (group or "").strip()]
        for group_from, group_to in zip(chain, chain[1:]):
            for a in group_from:
                for b in group_to:
                    add(a, b, "lift", lift)

    toilets = {}
    for row in tables["Toilets.csv"]:
        if not _true(row.get("IsAccessible")):
            continue
        toilets.setdefault((row.get("StationUniqueId") or "").strip(), []).append({
            "location": (row.get("Location") or "").strip(),
            # None when TfL leaves it blank: unknown, not "outside".
            "inside_gates": _true_or_none(row.get("IsInsideGateLine")),
            "radar_key": _true(row.get("RadarKey")),
            "fee": _true(row.get("IsFeeCharged")),
            "baby_changing": _true(row.get("HasBabyChanging")),
            "hours": (row.get("OpeningHours") or "").strip(),
        })

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

        # Platforms take the level and longitude of the place they open onto:
        # a walkway neighbour first, then any linked place, then a linked
        # platform. TfL's file has step-free links only, so a platform with
        # no step-free access has no links at all: it gets no level (None)
        # and the diagram shows it apart, not on the street row.
        unplaced = []
        for pid in platforms[sid]:
            linked = [(kind != "walkway", station_places[n]) for n, kind in neighbours.get(pid, [])
                      if station_places[n]["kind"] == "place"]
            linked.sort(key=lambda item: item[0])
            if linked:
                station_places[pid]["level"] = linked[0][1]["level"]
                station_places[pid]["lon"] = linked[0][1]["lon"]
            else:
                unplaced.append(pid)
        for pid in unplaced:
            near = [station_places[n] for n, kind in neighbours.get(pid, [])
                    if station_places[n]["kind"] == "platform" and station_places[n]["level"] is not None]
            station_places[pid]["level"] = near[0]["level"] if near else None
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
        # Different lines can share a platform name ("Westbound Platform 1"
        # at Heathrow Terminals 2 & 3): add the line where names repeat.
        counts = {}
        for pid in platforms[sid]:
            counts[station_places[pid]["name"]] = counts.get(station_places[pid]["name"], 0) + 1
        for pid, platform in platforms[sid].items():
            if counts[station_places[pid]["name"]] > 1:
                station_places[pid]["name"] = f"{platform['name']} ({', '.join(platform['lines'])})"
        _number_repeats(station_places)
        merged = _merge_platform_areas(station_places, station_links, platforms[sid])
        stations[sid] = {
            "format": DATA_FORMAT,
            "tfl_id": sid,
            "tfl_ids": [sid],
            "name": station_rows[sid]["Name"].strip(),
            "places": station_places,
            "links": station_links,
            "platforms": platforms[sid],
            "starts": starts,
            # platform-area place id -> the platform it is drawn and named as
            "merged": merged,
            "facilities": _facilities(station_rows[sid], toilets.get(sid, [])),
            "layout": compute_layout(station_places, hidden=merged),
        }

    return Feed(published=published, stations=stations,
                codes={sid: codes[sid] for sid in stations})


def merge_routes(routes_list):
    """One station's route data from several TfL stations.

    TfL sometimes lists one interchange as separate stations sharing a hub
    code (West Hampstead's Tube and Overground). We count it as one station,
    so their places, links, platforms and starts are combined. They share no
    links, so each part keeps its own street node and routes.

    Names are unique within each part, not across them, so where parts
    repeat a place or lift name ("Lift 1", "Ticket hall") the name gets the
    part's lines added. The parts are copied, never changed in place.
    """
    parts = sorted((copy.deepcopy(routes) for routes in routes_list),
                   key=lambda routes: routes["tfl_id"])

    def part_label(routes):
        lines = sorted({line for p in routes["platforms"].values() for line in p["lines"]})
        return ", ".join(lines) or routes["tfl_id"]

    place_names, lift_names = {}, {}
    for routes in parts:
        for name in {p["name"] for p in routes["places"].values() if p["kind"] != "street"}:
            place_names[name] = place_names.get(name, 0) + 1
        for name in {link["lift_name"] for link in routes["links"] if link["kind"] == "lift"}:
            lift_names[name] = lift_names.get(name, 0) + 1

    places, links, platforms, starts, merged = {}, [], {}, [], {}
    facilities = {"blue_badge_spaces": None, "taxi_rank": False, "toilets": []}
    for routes in parts:
        label = part_label(routes)
        for place in routes["places"].values():
            if place["kind"] != "street" and place_names[place["name"]] > 1:
                place["name"] = f"{place['name']} – {label}"
        for link in routes["links"]:
            if link["kind"] == "lift" and lift_names[link["lift_name"]] > 1:
                link["lift_name"] = f"{link['lift_name']} – {label}"
        places.update(routes["places"])
        links.extend(routes["links"])
        platforms.update(routes["platforms"])
        starts.extend(routes["starts"])
        merged.update(routes.get("merged", {}))
        part = routes.get("facilities") or {}
        if part.get("blue_badge_spaces") is not None:
            facilities["blue_badge_spaces"] = (facilities["blue_badge_spaces"] or 0) + part["blue_badge_spaces"]
        facilities["taxi_rank"] = facilities["taxi_rank"] or part.get("taxi_rank", False)
        facilities["toilets"] += part.get("toilets", [])
    return {
        "format": DATA_FORMAT,
        "tfl_id": parts[0]["tfl_id"],
        "tfl_ids": [routes["tfl_id"] for routes in parts],
        "name": parts[0]["name"],
        "places": places,
        "links": links,
        "platforms": platforms,
        "starts": starts,
        "merged": merged,
        "facilities": facilities,
        "layout": compute_layout(places, hidden=merged),
    }


# TfL gives places codes, not names ("NPN", "SPW"). Places of the same kind
# are told apart by where they are in the station, as a compass direction
# from its middle; anything still ambiguous is numbered west to east.
_COMPASS = ("North", "North-east", "East", "South-east", "South", "South-west", "West", "North-west")
# Closer than this (degrees, about 15 m) to the middle has no direction.
_CENTRE_RADIUS = 0.00013


def _compass(place, centre):
    if centre is None or place.get("lat") is None or place.get("lon") is None:
        return None
    lat0, lon0 = centre
    dx = (place["lon"] - lon0) * math.cos(math.radians(lat0))
    dy = place["lat"] - lat0
    if math.hypot(dx, dy) < _CENTRE_RADIUS:
        return None
    bearing = (math.degrees(math.atan2(dx, dy)) + 360) % 360
    return _COMPASS[int((bearing + 22.5) // 45) % 8]


def _with_direction(direction, label):
    # "North-west ticket hall", but "East Central line platform" keeps the
    # line's capital.
    first = label.split()[0]
    proper = first in ("Central", "Elizabeth", "Northern", "Bakerloo", "Circle", "District",
                       "Hammersmith", "Jubilee", "Metropolitan", "Piccadilly", "Victoria", "Waterloo")
    return f"{direction} {label if proper else label[0].lower() + label[1:]}"


def _name_places(station_places):
    """Give every non-platform place a readable, unique name."""
    located = [p for p in station_places.values()
               if p["kind"] != "street" and p.get("lat") is not None and p.get("lon") is not None]
    centre = ((sum(p["lat"] for p in located) / len(located),
               sum(p["lon"] for p in located) / len(located)) if located else None)
    groups = {}
    for pid, place in station_places.items():
        if place["kind"] in ("street", "platform"):
            continue
        label = place_label(place["code"])
        if label is None:
            # No rule for the code: say what it is, never show the code.
            label = "Entrance" if place["kind"] == "entrance" else "Passageway"
        place["name"] = label
        groups.setdefault(label, []).append(pid)
    for label, ids in groups.items():
        if len(ids) < 2:
            continue
        by_direction = {}
        for pid in ids:
            by_direction.setdefault(_compass(station_places[pid], centre), []).append(pid)
        for direction, members in by_direction.items():
            members.sort(key=lambda pid: (station_places[pid].get("lon") or 0.0, pid))
            base = _with_direction(direction, label) if direction else label
            for number, pid in enumerate(members, 1):
                station_places[pid]["name"] = f"{base} {number}" if len(members) > 1 else base


def _facilities(row, toilets):
    spaces = (row.get("BlueBadgeCarParkSpaces") or "").strip()
    has_parking = _true(row.get("BlueBadgeCarParking"))
    return {
        # None: no Blue Badge parking; 0: parking, number of spaces not given.
        "blue_badge_spaces": (int(spaces) if spaces.isdigit() else 0) if has_parking else None,
        "taxi_rank": _true(row.get("TaxiRanksOutsideStation")),
        "toilets": toilets,
    }


def _merge_platform_areas(station_places, station_links, station_platforms):
    """{area id: platform id} for platform areas that serve one platform.

    TfL lists most platforms twice: a platform-area point ("NorSB", the
    Northern line southbound platform) joined by a walkway to the platform
    itself ("Southbound Platform 8"). Where the area serves only that
    platform and the platform opens only onto that area, they are the same
    place: draw and name them once, as the platform. An area that serves
    several platforms (an island, a shared concourse) stays a place.
    """
    walkways, stepped = {}, set()
    for link in station_links:
        if link["kind"] == "walkway":
            walkways.setdefault(link["a"], set()).add(link["b"])
            walkways.setdefault(link["b"], set()).add(link["a"])
        else:
            # A lift or ramp between an area and a platform is a step worth
            # showing; merging the two would hide it.
            stepped.add(frozenset((link["a"], link["b"])))
    merged = {}
    for pid in station_platforms:
        areas = [n for n in walkways.get(pid, ()) if station_places[n]["kind"] == "place"]
        if len(areas) != 1 or frozenset((areas[0], pid)) in stepped:
            continue
        served = [n for n in walkways.get(areas[0], ()) if station_places[n]["kind"] == "platform"]
        if served == [pid]:
            merged[areas[0]] = pid

    # Some stations link each area to both platforms and to each other (King's
    # Cross's NorNB and NorSB both reach platforms 7 and 8). There the area's
    # code names its platform: line and direction, as in the platform's id.
    claimed = set(merged.values())
    for area, place in station_places.items():
        if place["kind"] != "place" or area in merged:
            continue
        match = _AREA_LINE_DIRECTION.match(place["code"].strip().lower())
        if not match:
            continue
        line, direction = _AREA_LINE_SLUGS[match[1].replace(" ", "")], match[2].upper()
        candidates = [
            n for n in walkways.get(area, ())
            if n in station_platforms and n not in claimed
            and _platform_direction(n) == direction and line in _platform_modes(n)
            and frozenset((area, n)) not in stepped
        ]
        if len(candidates) == 1:
            merged[area] = candidates[0]
            claimed.add(candidates[0])

    for pid in merged.values():
        # The platform name alone ("Southbound Platform 8") doesn't say
        # which line at a big interchange; the label does.
        station_places[pid]["name"] = station_platforms[pid]["label"]
    return merged


_AREA_LINE_DIRECTION = re.compile(r"^(bak|cen|cir|dis|ham|jub|met|nor|pic|vic|wat|el ?)(eb|wb|nb|sb)$")
_AREA_LINE_SLUGS = {
    "bak": "bakerloo", "cen": "central", "cir": "circle", "dis": "district",
    "ham": "hammersmith-city", "jub": "jubilee", "met": "metropolitan", "nor": "northern",
    "pic": "piccadilly", "vic": "victoria", "wat": "waterloo-city", "el": "elizabeth",
}


def _platform_direction(platform_id):
    """"NB" from "HUBKGX-Plat07-NB-northern"."""
    parts = platform_id.split("-", 3)
    return parts[2].upper() if len(parts) == 4 else ""


def _number_repeats(station_places):
    """Last resort: TfL occasionally repeats a code and area id in one
    station, so number whatever names still repeat."""
    seen = {}
    for place in station_places.values():
        if place["kind"] == "street":
            continue
        seen[place["name"]] = seen.get(place["name"], 0) + 1
        if seen[place["name"]] > 1:
            place["name"] = f"{place['name']} #{seen[place['name']]}"
