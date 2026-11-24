"""The chosen platform's step-free route, drawn level by level.

Rather than a map of the whole station (TfL gives each place as one rough
point, and a big station has more than a hundred walkways, so a full map
is a tangle), this shows only the way to one platform: each level the
route passes through as a block, top to bottom, with the places it goes
through, and the lift or ramp that takes you to the next level between
the blocks. It follows the same steps as the written directions.

This only does string formatting, so it is cheap per request.
"""

import hashlib

from .style import line_colour


def level_name(level):
    if level is None:
        return "No step-free route"
    if level == 0:
        return "Street level"
    count = abs(level)
    plural = "" if count == 1 else "s"
    side = "below" if level < 0 else "above"
    return f"{count} level{plural} {side} street"


def _panel_id(prefix, key, taken):
    """A stable id for a details panel: a short hash of the place or lift id,
    so links to a panel survive re-imports and no TfL id reaches the page."""
    digest = hashlib.sha1(key.encode()).hexdigest()
    for size in range(10, len(digest) + 1):
        panel_id = f"{prefix}-{digest[:size]}"
        if panel_id not in taken:
            taken.add(panel_id)
            return panel_id
    raise ValueError(f"no free panel id for {key}")


def _levels_text(start, end):
    if start is None or end is None or start == end:
        return ""
    count = abs(end - start)
    return f"{'up' if end > start else 'down'} {count} level{'' if count == 1 else 's'}"


def _route_without_merged_areas(routes, places, links):
    """The route with each platform area merged into its platform, as the
    written directions name it, dropping the walkway between the two."""
    merged = routes.get("merged", {})
    kept_places, kept_links = [merged.get(places[0], places[0])], []
    for link, place in zip(links, places[1:]):
        place = merged.get(place, place)
        if place != kept_places[-1]:
            kept_places.append(place)
            kept_links.append(link)
    return kept_places, kept_links


def build_journey(routes, route, lift_status_known=True):
    """{"levels": [...], "place_panels": {pid: id}, "lift_panels": {lift id: id}}.

    Each level is {"name", "items", "then"}: ``items`` alternate stops and
    the walk or same-level ramp between them; ``then`` is the lift or ramp
    to the next level, or None after the last. A stop is {"stop": True,
    "name", "kind", "panel", "colour", "notes"}; a move is {"kind", "text",
    "panel", "status"}.
    """
    empty = {"levels": [], "place_panels": {}, "lift_panels": {}}
    if not route or not route.places or not route.links:
        return empty
    info = routes["places"]
    places, links = _route_without_merged_areas(routes, route.places, route.links)
    # Entering by a walkway: start at the entrance, as the directions do.
    if info[places[0]]["kind"] == "street" and links and links[0]["kind"] == "walkway":
        places, links = places[1:], links[1:]
    if not links:
        return empty

    panel_ids, place_panels, lift_panels = set(), {}, {}

    def stop(pid):
        place = info[pid]
        if place["kind"] != "street":
            place_panels.setdefault(pid, _panel_id("place", pid, panel_ids))
        platform = routes["platforms"].get(pid) if place["kind"] == "platform" else None
        notes = []
        if platform:
            if platform.get("step"):
                notes.append(f"Step to train {platform['step']}")
            if platform.get("gap"):
                notes.append(f"Gap {platform['gap']}")
            if platform.get("manual_ramp"):
                notes.append("Staff ramp available")
        return {
            "stop": True, "kind": place["kind"],
            "name": platform["label"] if platform else ("The street" if place["kind"] == "street" else place["name"]),
            "panel": place_panels.get(pid, ""),
            "colour": line_colour(platform["lines"][0]) if platform and platform.get("lines") else "",
            "notes": notes,
        }

    # Moves between stops, each lift ride once however many stops it passes.
    stops, moves = [places[0]], []
    i = 0
    while i < len(links):
        link = links[i]
        j = i
        if link["kind"] == "lift":
            while j + 1 < len(links) and links[j + 1]["kind"] == "lift" \
                    and links[j + 1]["lift_id"] == link["lift_id"]:
                j += 1
        elif link["kind"] == "walkway":
            while j + 1 < len(links) and links[j + 1]["kind"] == "walkway":
                j += 1
        start, end = places[i], places[j + 1]
        levels = _levels_text(info[start]["level"], info[end]["level"])
        if link["kind"] == "lift":
            lift_panels.setdefault(link["lift_id"], _panel_id("lift", link["lift_id"], panel_ids))
            small = " (small lift)" if link.get("small") else ""
            moves.append({
                "kind": "lift", "text": f"{link['lift_name']}{small}", "detail": levels,
                "panel": lift_panels[link["lift_id"]],
                "status": "Working" if lift_status_known else "",
            })
        elif link["kind"] == "ramp":
            moves.append({"kind": "ramp", "text": "Ramp", "detail": levels, "panel": "", "status": ""})
        else:
            moves.append({"kind": "walk", "text": "Walk", "detail": levels, "panel": "", "status": ""})
        stops.append(end)
        i = j + 1

    blocks = [{"name": level_name(info[stops[0]]["level"]), "items": [stop(stops[0])], "then": None}]
    for move, before, pid in zip(moves, stops, stops[1:]):
        if info[pid]["level"] == info[before]["level"]:
            blocks[-1]["items"] += [move, stop(pid)]
        else:
            blocks[-1]["then"] = move
            blocks.append({"name": level_name(info[pid]["level"]), "items": [stop(pid)], "then": None})
    return {"levels": blocks, "place_panels": place_panels, "lift_panels": lift_panels}
