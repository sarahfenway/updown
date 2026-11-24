"""Text for the station page's tap-for-details panels and summary.

Built per request from the stored route data and live lift status; the
page shows it as plain HTML (no JSON, no TfL ids).
"""

from .diagram import level_name
from .routing import BLOCKED, NO_INFO, OK, find_route


UNLINKED_CAVEAT = ("Step-free only through an entrance TfL's data doesn't link to the street, "
                   "which may have steps")


def station_summary(routes):
    """Platform labels grouped by whether TfL's data has a step-free route
    from the street (ignoring live faults, which the page shows separately).

    ``step_free_unlinked``: only through an entrance TfL's data doesn't link
    to the street, which may have steps."""
    summary = {"step_free": [], "step_free_unlinked": [], "not_step_free": [], "no_information": []}
    for pid, platform in sorted(routes["platforms"].items(), key=lambda item: item[1]["label"]):
        result = find_route(routes, pid)
        if result.status == OK:
            key = "step_free_unlinked" if result.unlinked_entrance else "step_free"
        elif result.status == NO_INFO:
            key = "no_information"
        else:
            key = "not_step_free"
        summary[key].append(platform["label"])
    return summary


def facility_lines(facilities):
    """Readable lines for the station's facilities, or []."""
    if not facilities:
        return []
    lines = []
    spaces = facilities.get("blue_badge_spaces")
    if spaces:
        lines.append(f"Blue Badge parking ({spaces} space{'' if spaces == 1 else 's'})")
    elif spaces == 0:
        lines.append("Blue Badge parking")
    if facilities.get("taxi_rank"):
        lines.append("Taxi rank outside")
    for toilet in facilities.get("toilets", []):
        parts = [toilet["location"] or "location not given"]
        if toilet["inside_gates"] is not None:
            parts.append("inside the ticket gates" if toilet["inside_gates"] else "outside the ticket gates")
        if toilet["radar_key"]:
            parts.append("RADAR key")
        if toilet["baby_changing"]:
            parts.append("baby changing")
        if toilet["fee"]:
            parts.append("charge to use")
        if toilet["hours"]:
            parts.append(toilet["hours"])
        lines.append(f"Accessible toilet: {', '.join(parts)}")
    return lines


def build_panels(routes, diagram, broken_lift_ids=(), lift_status_known=True):
    """[{"id", "title", "items": [str]}] for every box and lift label."""
    broken = set(broken_lift_ids)
    places = routes["places"]
    merged = routes.get("merged", {})

    def drawn_as(pid):
        return merged.get(pid, pid)

    def lift_status(lift_id):
        if not lift_status_known:
            return "Live lift status unavailable"
        return "Out of service now" if lift_id in broken else "Working, as far as TfL reports"

    # Lift and ramp links as seen from each drawn place.
    lifts_by_id, from_place = {}, {}
    for link in routes["links"]:
        a, b = drawn_as(link["a"]), drawn_as(link["b"])
        if a == b:
            continue
        if link["kind"] == "lift":
            lifts_by_id.setdefault(link["lift_id"], []).append(link)
        if link["kind"] in ("lift", "ramp"):
            from_place.setdefault(a, []).append((link, b))
            from_place.setdefault(b, []).append((link, a))

    # Entrances the street node links to. The route starts also hold every
    # street-level ENT place, linked or not, so they can't tell us this.
    street = {pid for pid, place in places.items() if place["kind"] == "street"}
    linked = {link["b"] if link["a"] in street else link["a"]
              for link in routes["links"] if (link["a"] in street) != (link["b"] in street)}

    def connection_lines(pid):
        """Lifts and ramps from a drawn place, with live lift status."""
        here = places[pid]["level"]
        lines, seen = [], set()
        for link, other in from_place.get(pid, []):
            key = (link.get("lift_id"), other)
            if key in seen:
                continue
            seen.add(key)
            if link["kind"] == "ramp":
                lines.append(f"Ramp to {places[other]['name']}")
                continue
            there = places[other]["level"]
            way = " up" if (there or 0) > (here or 0) else " down" if (there or 0) < (here or 0) else ""
            line = f"{link['lift_name']}{way} to {places[other]['name']}"
            if lift_status_known and link["lift_id"] in broken:
                line += " – out of service"
            lines.append(line)
        return lines

    panels = []
    for pid, panel_id in diagram["place_panels"].items():
        place = places[pid]
        if place["kind"] == "platform":
            items = _platform_items(routes, pid, broken if lift_status_known else ())
            # A merged platform stands for its platform area too, where the
            # lifts arrive.
            items += connection_lines(pid)
            # A platform's full label says which line ("Northern line –
            # Northbound Platform 3"), which a box may not have room for.
            title = routes["platforms"][pid]["label"]
        else:
            items = [level_name(place["level"])]
            if place["kind"] == "entrance":
                items.append("Linked to the street in TfL's data" if pid in linked
                             else "Not linked to the street in TfL's data, so it may have steps")
            items += connection_lines(pid)
            title = place["name"]
        panels.append({"id": panel_id, "title": title, "items": items})

    for lift_id, panel_id in diagram["lift_panels"].items():
        links = lifts_by_id.get(lift_id, [])
        if not links:
            continue
        stops = {drawn_as(p) for link in links for p in (link["a"], link["b"])}
        ordered = sorted(stops, key=lambda p: -(places[p]["level"] if places[p]["level"] is not None else -99))
        items = [f"Stops: {', '.join(places[p]['name'] for p in ordered)}"]
        if links[0].get("small"):
            items.append("Small lift")
        if links[0].get("notes"):
            items.append(links[0]["notes"])
        items.append(lift_status(lift_id))
        panels.append({"id": panel_id, "title": links[0]["lift_name"], "items": items})
    return panels


def _platform_items(routes, pid, broken):
    platform = routes["platforms"][pid]
    usual = find_route(routes, pid)
    if usual.status == OK:
        now = find_route(routes, pid, broken) if broken else usual
        if now.status == BLOCKED:
            items = [f"No step-free route right now: {', '.join(now.blocking_lifts)} out of service"]
        elif usual.unlinked_entrance:
            items = [UNLINKED_CAVEAT]
        else:
            items = ["Step-free from the street"]
    elif usual.status == NO_INFO:
        items = ["TfL hasn't surveyed this platform for step-free access"]
    else:
        items = ["No step-free route from the street in TfL's data"]
    if platform.get("step"):
        items.append(f"Step to the train: {platform['step']}")
    if platform.get("gap"):
        items.append(f"Gap to the train: {platform['gap']}")
    if platform.get("boarding"):
        items.append(f"Board at: {platform['boarding']}")
    if platform.get("manual_ramp"):
        items.append("Staff ramp available")
    if platform.get("entrance"):
        items.append(f"Accessible entrance: {platform['entrance']}")
    items.extend(platform.get("notes", []))
    return items
