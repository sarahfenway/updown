"""Turn a station's stored layout and live state into SVG drawing data.

All positions come from the layout computed at import; this only does
arithmetic and string formatting, so it is cheap per request.
"""

COLUMN_WIDTH = 150
ROW_HEIGHT = 110
BOX_WIDTH = 128
BOX_HEIGHT = 44
LEFT = 130   # room for the level labels
TOP = 24
LINE_CHARS = 20


def level_name(level):
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
    broken = set(broken_lift_ids)
    route_places = set(route.places) if route and route.places else set()
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

    lines = []
    for link in routes["links"]:
        if link["a"] not in positions or link["b"] not in positions:
            continue  # links to the street node are not drawn
        (x1, y1), (x2, y2) = centre(link["a"]), centre(link["b"])
        classes = [link["kind"]]
        label = ""
        if link["kind"] == "lift":
            # Lifts read as vertical: down from the first place, then across.
            points = f"{x1:.0f},{y1:.0f} {x1:.0f},{y2:.0f} {x2:.0f},{y2:.0f}"
            label = link["lift_name"]
            if link["lift_id"] in broken:
                classes.append("broken")
                label = f"{label} – out of service"
            label_x, label_y = x1 + 6, (y1 + y2) / 2
        else:
            points = f"{x1:.0f},{y1:.0f} {x2:.0f},{y2:.0f}"
            label_x, label_y = 0, 0
        if _link_key(link) in route_links:
            classes.append("on-route")
        lines.append({"points": points, "cls": " ".join(classes), "label": label,
                      "label_x": f"{label_x:.0f}", "label_y": f"{label_y:.0f}"})

    boxes = []
    for pid, place in routes["places"].items():
        if pid not in positions:
            continue
        x, y = top_left(pid)
        classes = ["place", place["kind"]]
        if pid in route_places:
            classes.append("on-route")
        text_lines = _wrap(place["name"])
        first_y = y + BOX_HEIGHT / 2 + 4 - (6 if len(text_lines) > 1 else 0)
        boxes.append({
            "x": f"{x:.0f}", "y": f"{y:.0f}", "w": BOX_WIDTH, "h": BOX_HEIGHT,
            "cx": f"{x + BOX_WIDTH / 2:.0f}", "ty": f"{first_y:.0f}",
            "lines": text_lines, "cls": " ".join(classes),
        })

    width = LEFT + max(layout["columns"], 1) * COLUMN_WIDTH
    height = TOP + max(len(layout["levels"]), 1) * ROW_HEIGHT
    return {"width": width, "height": height, "bands": bands, "lines": lines, "boxes": boxes}
