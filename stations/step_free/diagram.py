"""Turn a station's stored layout and live state into SVG drawing data.

All positions come from the layout computed at import; this only does
arithmetic and string formatting, so it is cheap per request.
"""

import hashlib

COLUMN_WIDTH = 150
ROW_HEIGHT = 110
BOX_WIDTH = 128
BOX_HEIGHT = 44
LEFT = 130   # room for the level labels
RIGHT = 150  # room for lift labels that start in the last column
TOP = 24
LINE_CHARS = 20
# Lift lanes in a column sit this far apart, alternating either side of the
# box centre: 0, -14, +14, -28, +28 ... They never wrap round, so every
# lift in a column keeps a line of its own.
LANE_STEP = 14
LABEL_STAGGER = 13


def _lane_offset(lane):
    side = -1 if lane % 2 else 1
    return side * ((lane + 1) // 2) * LANE_STEP


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


def level_name(level):
    if level is None:
        return "No step-free route"
    if level == 0:
        return "Street level"
    count = abs(level)
    plural = "" if count == 1 else "s"
    side = "below" if level < 0 else "above"
    return f"{count} level{plural} {side} street"


def _wrap(name):
    """Up to two lines of about LINE_CHARS characters each."""
    words, lines, current = name.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > LINE_CHARS:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    lines.append(current)
    if len(lines) > 2:
        lines = [lines[0], " ".join(lines[1:])]
    if len(lines[-1]) > LINE_CHARS + 4:
        lines[-1] = lines[-1][: LINE_CHARS + 3] + "…"
    return lines


def _link_key(link):
    return (min(link["a"], link["b"]), max(link["a"], link["b"]), link["kind"], link.get("lift_id"))


def build_diagram(routes, broken_lift_ids=(), route=None):
    layout = routes["layout"]
    positions = layout["positions"]
    merged = routes.get("merged", {})

    def drawn_as(pid):
        return merged.get(pid, pid)

    broken = set(broken_lift_ids)
    route_places = {drawn_as(pid) for pid in route.places} if route and route.places else set()
    route_links = {_link_key(link) for link in route.links} if route and route.links else set()

    def top_left(pid):
        column, row = positions[pid]
        return LEFT + column * COLUMN_WIDTH, TOP + row * ROW_HEIGHT

    def centre(pid):
        x, y = top_left(pid)
        return x + BOX_WIDTH / 2, y + BOX_HEIGHT / 2

    bands = []
    for row, level in enumerate(layout["levels"]):
        y = TOP + row * ROW_HEIGHT
        bands.append({"y": y - 14, "label_y": y + BOX_HEIGHT / 2 + 4, "label": level_name(level)})

    places = routes["places"]
    lift_panels = {}  # lift id -> id of its details panel
    panel_ids = set()
    drawn_pairs = {}  # (box, box, kind, lift id) -> index in lines
    lanes = {}       # column -> lift ids, in the order first drawn
    labelled = set()  # lifts already labelled (a lift may have several links)
    labels_below = {}  # upper place -> lift labels already placed under it
    lines = []
    for link in routes["links"]:
        a, b = drawn_as(link["a"]), drawn_as(link["b"])
        if a not in positions or b not in positions or a == b:
            continue  # the street node, and an area's walkway to its own platform
        pair = (min(a, b), max(a, b), link["kind"], link.get("lift_id"))
        if pair in drawn_pairs:
            # Merged areas can map several walkways onto the same two boxes:
            # draw the line once, highlighted if any of them is on the route.
            if _link_key(link) in route_links:
                drawn = lines[drawn_pairs[pair]]
                if "on-route" not in drawn["cls"].split():
                    drawn["cls"] += " on-route"
            continue
        drawn_pairs[pair] = len(lines)
        classes = [link["kind"]]
        label = ""
        lift_id = link.get("lift_id")
        if link["kind"] == "lift":
            # Lifts read as vertical: down from the upper place, then across
            # to the lower one. Each lift in a column gets its own lane, so
            # long lifts don't run hidden along another lift's line.
            top, bottom = sorted(
                (a, b),
                key=lambda pid: -(places[pid]["level"] if places[pid]["level"] is not None else -99),
            )
            (x1, y1), (x2, y2) = centre(top), centre(bottom)
            column = lanes.setdefault(positions[top][0], [])
            if lift_id not in column:
                column.append(lift_id)
            lane = column.index(lift_id)
            x = x1 + _lane_offset(lane)
            points = f"{x:.0f},{y1:.0f} {x:.0f},{y2:.0f} {x2:.0f},{y2:.0f}"
            if lift_id in broken:
                classes.append("broken")
            label_x = label_y = 0
            if lift_id not in labelled:
                labelled.add(lift_id)
                lift_panels[lift_id] = _panel_id("lift", lift_id, panel_ids)
                label = link["lift_name"]
                if lift_id in broken:
                    label = f"{label} – out of service"
                # Labels go in the gap just below the upper box (a long
                # lift's midpoint can fall behind a box on an in-between
                # level), each a line below the last one under that box.
                stacked = labels_below.get(top, 0)
                labels_below[top] = stacked + 1
                label_x = x + 4
                label_y = y1 + BOX_HEIGHT / 2 + 14 + stacked * LABEL_STAGGER
        else:
            (x1, y1), (x2, y2) = centre(a), centre(b)
            points = f"{x1:.0f},{y1:.0f} {x2:.0f},{y2:.0f}"
            label_x, label_y = 0, 0
        if _link_key(link) in route_links:
            classes.append("on-route")
        # The label's own classes: the line's "lift" class sets a stroke,
        # which would outline the text.
        label_cls = "link-label broken" if "broken" in classes else "link-label"
        lines.append({"points": points, "cls": " ".join(classes), "label": label,
                      "label_cls": label_cls, "lift": lift_id,
                      "panel": lift_panels.get(lift_id, "") if label else "",
                      "label_x": f"{label_x:.0f}", "label_y": f"{label_y:.0f}"})

    boxes = []
    place_panels = {}  # place id -> id of its details panel
    for pid, place in routes["places"].items():
        if pid not in positions:
            continue
        place_panels[pid] = _panel_id("place", pid, panel_ids)
        x, y = top_left(pid)
        classes = ["place", place["kind"]]
        if pid in route_places:
            classes.append("on-route")
        text_lines = _wrap(place["name"])
        first_y = y + BOX_HEIGHT / 2 + 4 - (6 if len(text_lines) > 1 else 0)
        boxes.append({
            "x": f"{x:.0f}", "y": f"{y:.0f}", "w": BOX_WIDTH, "h": BOX_HEIGHT,
            "cx": f"{x + BOX_WIDTH / 2:.0f}", "ty": f"{first_y:.0f}",
            "lines": text_lines, "cls": " ".join(classes), "panel": place_panels[pid],
            "label": place["name"],
            # The ⓘ sits on the box's top right corner, like a badge, clear
            # of the name inside.
            "icon_x": f"{x + BOX_WIDTH - 2:.0f}", "icon_y": f"{y + 2:.0f}",
            "icon_ty": f"{y + 5:.0f}",
        })

    width = LEFT + max(layout["columns"], 1) * COLUMN_WIDTH + RIGHT
    height = TOP + max(len(layout["levels"]), 1) * ROW_HEIGHT
    return {"width": width, "height": height, "bands": bands, "lines": lines, "boxes": boxes,
            "place_panels": place_panels, "lift_panels": lift_panels}
