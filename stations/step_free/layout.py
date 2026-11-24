"""Grid layout for a station's step-free diagram, computed once at import."""


def compute_layout(places, hidden=()):
    """Place every drawn place on a grid of levels (rows) and columns.

    ``places`` maps place id to a dict with at least ``level`` (int),
    ``kind`` and ``lon`` (float or None). The street node (kind "street")
    is not drawn. Rows run from the highest level to the lowest, so the
    street sits above anything below ground and below any footbridge.
    Within a row, places run west to east by longitude. Places whose level
    is None (platforms with no step-free link) go in a last row. Places in
    ``hidden`` (platform areas merged into their platform) aren't drawn.

    Returns ``{"levels": [int, ...], "columns": int,
    "positions": {place_id: [column, row]}}``.
    """
    drawn = {pid: p for pid, p in places.items() if p["kind"] != "street" and pid not in hidden}
    levels = sorted({p["level"] for p in drawn.values() if p["level"] is not None}, reverse=True)
    # Platforms with no step-free link have no level: a last row of their own.
    if any(p["level"] is None for p in drawn.values()):
        levels.append(None)
    row_of = {level: row for row, level in enumerate(levels)}

    by_level = {}
    for pid, place in drawn.items():
        by_level.setdefault(place["level"], []).append(pid)

    positions = {}
    columns = 0
    for level, ids in by_level.items():
        ids.sort(key=lambda pid: (drawn[pid]["lon"] is None, drawn[pid]["lon"] or 0.0, pid))
        for column, pid in enumerate(ids):
            positions[pid] = [column, row_of[level]]
        columns = max(columns, len(ids))

    return {"levels": levels, "columns": columns, "positions": positions}
