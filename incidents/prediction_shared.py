"""Prediction logic shared by the app (incidents/ml.py) and the training
script (ml/train.py).

Standard library only: no Django, numpy or pandas. Both sides import this
exact code, so the rules for what gets predicted, how history is summarised
and how a probability vector becomes a "fixed by" bound cannot drift apart.
"""

import math
import statistics
from datetime import timedelta

from incidents.text_features import normalise_incident_text

MAX_OFFSET_CLASS = 8
BLOCKS_PER_DAY = 4
BLOCK_OVERNIGHT = 3

# A prediction is the earliest block by which the fault is this likely to be
# fixed. It was 0.95 for eleven days in 2026-11: right 96% of the time, but
# about one fault in ten lasts two days or more, so nearly every date had to
# be days away (a median 62 hours promised for faults that took 4.5). 0.80
# trades some of that certainty for dates people can plan around.
PREDICTION_COVERAGE_TARGET = 0.80

# The tail table covers block offsets MAX_OFFSET_CLASS..TAIL_HORIZON_BLOCKS
# (2 to 30 days). A category needs TAIL_MIN_COUNT long faults to have its own
# table; otherwise the all-categories table is used.
TAIL_HORIZON_BLOCKS = 120
TAIL_MIN_COUNT = 30

HISTORY_WINDOW = 50
HISTORY_OFFSET_CAP = 60
HISTORY_STATS = (
    "count",
    "mean_off",
    "median_off",
    "p90_off",
    "max_off",
    "long_share",
    "last_off",
    "last3_mean_off",
    "days_since_last_end",
)
HISTORY_FEATURE_NAMES = tuple(
    f"{group}_{stat}" for group in ("st", "sc", "sf") for stat in HISTORY_STATS
) + (
    "st_started_7d",
    "sf_started_7d",
    "st_started_30d",
    "sf_started_30d",
    "st_started_90d",
    "sf_started_90d",
    "st_open_now",
)


def is_planned_work(text, information=False):
    """True for work that is never predicted.

    Information notices and anything that states its own end ("until ...")
    need no prediction. Planned maintenance is left out because it cannot be
    predicted well: in the 2026-11 retrain design even dates eleven days out
    were right first time only 88% of the time.
    """
    text = (text or "").lower()
    return bool(information) or "planned" in text or "until " in text


def fault_category(text):
    text = (text or "").lower()
    if "faulty lift" in text:
        return "faulty_lift"
    if "planned maintenance" in text:
        return "planned_maintenance"
    if "staff" in text:
        return "staff_issue"
    return "other"


def fault_key(text):
    """Identifies "the same fault" at a station: its cleaned, lower-cased text."""
    return normalise_incident_text(text).lower()


# ---------------------------------------------------------------------------
# History from previous faults
# ---------------------------------------------------------------------------


def _group_stats(rows, start):
    if not rows:
        return (0,) + (-1.0,) * (len(HISTORY_STATS) - 1)
    offsets = [float(min(row[2], HISTORY_OFFSET_CAP)) for row in rows[-HISTORY_WINDOW:]]
    ordered = sorted(offsets)
    last_three = offsets[-3:]
    return (
        len(rows),
        sum(offsets) / len(offsets),
        float(statistics.median(ordered)),
        ordered[math.ceil(0.9 * len(ordered)) - 1],
        ordered[-1],
        sum(1 for offset in offsets if offset >= MAX_OFFSET_CLASS) / len(offsets),
        offsets[-1],
        sum(last_three) / len(last_three),
        (start - rows[-1][1]).total_seconds() / 86400,
    )


def history_features(past, start, category, key):
    """Summarise a station's earlier faults as they stood at ``start``.

    ``past`` holds one ``(start, end, offset, category, key)`` tuple per other
    fault at the same station family. ``end`` and ``offset`` are None while a
    fault is unresolved. Only what was knowable at ``start`` is used: a fault
    counts towards the duration statistics only if it had already ended.

    Groups: ``st`` the station, ``sc`` the station and this category, ``sf``
    the same fault (station and same cleaned text).
    """
    past = [row for row in past if row[0] < start]
    ended = sorted(
        (row for row in past if row[1] is not None and row[1] <= start),
        key=lambda row: row[1],
    )
    groups = {
        "st": ended,
        "sc": [row for row in ended if row[3] == category],
        "sf": [row for row in ended if row[4] == key],
    }
    features = {}
    for group, rows in groups.items():
        for stat, value in zip(HISTORY_STATS, _group_stats(rows, start)):
            features[f"{group}_{stat}"] = value
    for days in (7, 30, 90):
        since = start - timedelta(days=days)
        recent = [row for row in past if row[0] >= since]
        features[f"st_started_{days}d"] = len(recent)
        features[f"sf_started_{days}d"] = sum(1 for row in recent if row[4] == key)
    features["st_open_now"] = sum(
        1 for row in past if row[1] is None or row[1] > start
    )
    return features


# ---------------------------------------------------------------------------
# From class probabilities to a "fixed by" bound
# ---------------------------------------------------------------------------


def _tail_counts(offsets):
    counts = [0] * (TAIL_HORIZON_BLOCKS - MAX_OFFSET_CLASS + 1)
    total = 0
    for offset in offsets:
        if offset < MAX_OFFSET_CLASS:
            continue
        total += 1
        if offset <= TAIL_HORIZON_BLOCKS:
            counts[offset - MAX_OFFSET_CLASS] += 1
    return {"total": total, "counts": counts}


def build_tail_distribution(offsets_by_category):
    """How long faults lasting MAX_OFFSET_CLASS blocks or more went on to last.

    ``offsets_by_category`` maps a category to raw (uncapped) block offsets.
    Faults beyond the horizon count in ``total`` but in no bucket, so their
    share of the tail is never assigned to a date.
    """
    everything = [
        offset for offsets in offsets_by_category.values() for offset in offsets
    ]
    return {
        "by_category": {
            category: _tail_counts(offsets)
            for category, offsets in offsets_by_category.items()
        },
        "all": _tail_counts(everything),
    }


def _tail_entry(tail, category):
    if not tail:
        return None
    for entry in ((tail.get("by_category") or {}).get(category), tail.get("all")):
        if entry and entry.get("total", 0) >= TAIL_MIN_COUNT:
            return entry
    return None


def extended_cdf(class_probs, tail=None, category=None):
    """P(fixed by the end of block offset k), for k = 0, 1, 2, ...

    ``class_probs[k]`` is the model's probability for offset ``k``; the last
    entry (index MAX_OFFSET_CLASS) is the open-ended "this many or more". The
    tail table spreads that last probability over offsets up to the horizon.
    Without a usable table the curve stops before MAX_OFFSET_CLASS.
    """
    cdf = []
    running = 0.0
    for probability in list(class_probs)[:MAX_OFFSET_CLASS]:
        running += float(probability)
        cdf.append(running)
    entry = _tail_entry(tail, category)
    if entry is None or len(class_probs) <= MAX_OFFSET_CLASS:
        return cdf
    tail_probability = float(class_probs[MAX_OFFSET_CLASS])
    seen = 0
    for count in entry["counts"]:
        seen += count
        cdf.append(running + tail_probability * seen / entry["total"])
    return cdf


def bound_offset(cdf, target=PREDICTION_COVERAGE_TARGET):
    """First block offset by which the curve reaches ``target``, else None."""
    for offset, value in enumerate(cdf):
        if value >= target - 1e-9:
            return offset
    return None


def round_up_to_service_day(start_block_index, offset):
    """Bounds two or more days out are day-level, not block-level.

    Push such a bound to the last block of its service day (overnight, which
    ends at 05:00) so the label can honestly name a day.
    """
    if offset < MAX_OFFSET_CLASS:
        return offset
    return offset + BLOCK_OVERNIGHT - (start_block_index + offset) % BLOCKS_PER_DAY
