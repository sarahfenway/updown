# Right-First-Time Predictions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give nearly every new unplanned fault a "fixed by" date that is right first time about 96% of the time, admit misses instead of revising them, and stop showing anything that cannot clear 90%.

**Architecture:** The 9-class block-offset classifier stays. A new standard-library-only module, `incidents/prediction_shared.py`, is imported by both the app (`incidents/ml.py`) and the training script (`ml/train.py`), so the stated-end rule, the history features and the decode from probabilities to a bound are one piece of code on both sides. The decode spreads the model's open-ended "2+ days" probability along a tail table shipped in the model bundle, and walks to the 95% point. The display layer shows a prediction only while its recent record is 90%+, and shows "Taking longer than we predicted" once a predicted time has passed.

**Tech Stack:** Python 3.14, Django 6.0.8, scikit-learn 1.9.0 (`HistGradientBoostingClassifier`), pandas 3.0.3 (training only), SQLite, pytest.

**Spec:** `docs/superpowers/specs/2026-11-14-prediction-accuracy-coverage-design.md`

## Global Constraints

- Python 3.14, Django 6.0.8. scikit-learn is 1.9.0 in both `requirements.txt` and `ml/requirements.txt` and must stay identical in the two files.
- No new runtime dependencies. The app must not import pandas.
- Production is one 512 MB machine: the model keeps 9 classes, and nothing adds a per-request scan of history.
- No database migration. The `revised_estimated_duration` and `revised_prediction_confidence` columns stay in the model; code stops reading and writing them.
- A model bundle with `metadata.feature_version` 3 (what production runs today) must keep working until a version 4 bundle is uploaded.
- Exact numbers: aim `PREDICTION_COVERAGE_TARGET = 0.95`; shown only with at least 30 recent scored predictions of which at least 90% were right; retrain gates 0.93 overall and 0.90 for any category with at least 100 cross-validation samples; tail horizon 120 blocks; a category needs 30 long faults for its own tail table; grace stays 30 minutes.
- Exact copy: `Taking longer than we predicted`; `Right first time`; day-level labels `the end of today`, `the end of tomorrow`, `the end of Tuesday`, `the end of Sat 21 Nov`.
- App tests: `.venv/bin/python -m pytest -q` (baseline before this plan: 152 passed). Training tests: `.venv-ml/bin/python -m pytest ml/test_train.py -q -p no:cacheprovider`.
- Work on branch `prediction-accuracy-coverage`. Commit after every task. Never push: pushing `master` deploys to production.
- End every commit message with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `incidents/prediction_shared.py` (new) | Stated-end rule, fault category and key, history features, tail table, decode. Standard library only. | 1 |
| `tests/test_prediction_shared.py` (new) | Unit tests for the shared module. | 1 |
| `incidents/ml.py` | Feature building against the database, model loading, decode to a duration, labels. | 2, 3, 4 |
| `incidents/management/commands/update_incidents.py` | When a prediction is made. | 2 |
| `incidents/management/commands/predict_durations.py` | Backfill. | 8 |
| `incidents/views.py` | Display rule, overdue state, API, stats, upload validation. | 2, 3, 5, 6, 7 |
| `templates/home.html`, `templates/beta.html`, `templates/stats.html` | Wording. | 2, 3, 6, 7 |
| `ml/train.py`, `ml/test_train.py` | Training, cross-validation, gates, bundle. | 9 |
| `.github/workflows/retrain.yml` | Gate defaults. | 10 |
| `tests/test_incidents.py` | App tests. | 2 to 8 |

---

### Task 1: Shared prediction module

**Files:**
- Create: `incidents/prediction_shared.py`
- Create: `tests/test_prediction_shared.py`

**Interfaces:**
- Consumes: `incidents.text_features.normalise_incident_text(text) -> str` (exists).
- Produces, all in `incidents.prediction_shared`:
  - Constants `MAX_OFFSET_CLASS = 8`, `PREDICTION_COVERAGE_TARGET = 0.95`, `TAIL_HORIZON_BLOCKS = 120`, `TAIL_MIN_COUNT = 30`, `HISTORY_FEATURE_NAMES` (tuple of 34 strings).
  - `has_stated_end(text, information=False) -> bool`
  - `fault_category(text) -> str` returning `"faulty_lift"`, `"planned_maintenance"`, `"staff_issue"` or `"other"`
  - `fault_key(text) -> str`
  - `history_features(past, start, category, key) -> dict[str, float]` where `past` is an iterable of `(start, end, offset, category, key)` tuples, with `end` and `offset` None for an unresolved fault
  - `build_tail_distribution(offsets_by_category: dict[str, list[int]]) -> dict` shaped `{"by_category": {cat: {"total": int, "counts": list[int]}}, "all": {"total": int, "counts": list[int]}}`
  - `extended_cdf(class_probs, tail=None, category=None) -> list[float]`
  - `bound_offset(cdf, target=PREDICTION_COVERAGE_TARGET) -> int | None`
  - `round_up_to_service_day(start_block_index: int, offset: int) -> int`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_prediction_shared.py`:

```python
from datetime import datetime, timedelta, timezone

from incidents import prediction_shared as shared

T0 = datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc)


def fault(started_hours_ago, lasted_hours=None, offset=None,
          category="faulty_lift", key="lift a"):
    start = T0 - timedelta(hours=started_hours_ago)
    end = None if lasted_hours is None else start + timedelta(hours=lasted_hours)
    return (start, end, offset, category, key)


def test_stated_end_rule():
    assert shared.has_stated_end("No step-free access until Sunday 22 November")
    assert shared.has_stated_end("Lift works", information=True)
    assert not shared.has_stated_end("Not available due to planned maintenance.")
    assert not shared.has_stated_end(None)


def test_fault_key_ignores_phone_number_case_and_spacing():
    a = "Faulty lift.  Call us on 0343 222 1234 for help"
    b = "faulty lift. call us on for help"
    assert shared.fault_key(a) == shared.fault_key(b)


def test_history_with_no_past_faults():
    features = shared.history_features([], T0, "faulty_lift", "lift a")
    assert set(features) == set(shared.HISTORY_FEATURE_NAMES)
    assert features["st_count"] == 0
    assert features["st_last_off"] == -1.0
    assert features["st_open_now"] == 0


def test_history_only_uses_what_was_known_at_the_start():
    past = [
        fault(100, lasted_hours=10, offset=2),   # ended 90h ago
        fault(48, lasted_hours=24, offset=5),    # ended 24h ago
        fault(20, lasted_hours=30, offset=6),    # still open at T0
        fault(5),                                # unresolved
        fault(-1, lasted_hours=1, offset=0),     # starts after T0
    ]
    features = shared.history_features(past, T0, "faulty_lift", "lift a")
    assert features["st_count"] == 2
    assert features["st_mean_off"] == 3.5
    assert features["st_last_off"] == 5.0
    assert features["st_days_since_last_end"] == 1.0
    assert features["st_open_now"] == 2
    assert features["st_started_7d"] == 4
    assert features["st_started_30d"] == 4


def test_history_groups_by_category_and_same_fault():
    past = [
        fault(100, 10, 2, category="staff_issue", key="staff"),
        fault(80, 10, 9, category="faulty_lift", key="lift b"),
        fault(60, 10, 1, category="faulty_lift", key="lift a"),
    ]
    features = shared.history_features(past, T0, "faulty_lift", "lift a")
    assert features["st_count"] == 3
    assert features["sc_count"] == 2
    assert features["sc_long_share"] == 0.5
    assert features["sf_count"] == 1
    assert features["sf_last_off"] == 1.0
    assert features["sf_started_7d"] == 1


def test_history_statistics_use_last_fifty_capped_offsets():
    past = [fault(1000 - i, 1, 200) for i in range(10)] + [
        fault(900 - i, 1, i % 10) for i in range(50)
    ]
    features = shared.history_features(past, T0, "faulty_lift", "lift a")
    assert features["st_count"] == 60
    assert features["st_max_off"] == 9.0   # the 200s fell out of the window
    assert features["st_median_off"] == 4.5
    assert features["st_p90_off"] == 8.0
    capped = shared.history_features(past[:10], T0, "faulty_lift", "lift a")
    assert capped["st_max_off"] == 60.0


def test_tail_distribution_counts_and_overflow():
    table = shared.build_tail_distribution(
        {"faulty_lift": [3, 8, 8, 12, 120, 121, 500]}
    )
    entry = table["by_category"]["faulty_lift"]
    assert entry["total"] == 6          # 3 is not a tail fault
    assert entry["counts"][0] == 2      # offset 8
    assert entry["counts"][4] == 1      # offset 12
    assert entry["counts"][-1] == 1     # offset 120
    assert sum(entry["counts"]) == 4    # 121 and 500 overflow
    assert table["all"]["total"] == 6


def test_cdf_without_tail_stops_before_the_open_ended_class():
    probs = [0.5, 0.2, 0.1, 0, 0, 0, 0, 0, 0.2]
    cdf = shared.extended_cdf(probs)
    assert len(cdf) == 8
    assert shared.bound_offset(cdf, 0.75) == 2
    assert shared.bound_offset(cdf, 0.95) is None


def test_cdf_spreads_tail_probability_along_the_table():
    table = shared.build_tail_distribution({"faulty_lift": [8] * 20 + [12] * 20})
    probs = [0.5, 0.2, 0.1, 0, 0, 0, 0, 0, 0.2]
    cdf = shared.extended_cdf(probs, table, "faulty_lift")
    assert len(cdf) == 121
    assert abs(cdf[8] - 0.9) < 1e-9
    assert abs(cdf[12] - 1.0) < 1e-9
    assert shared.bound_offset(cdf, 0.95) == 12


def test_overflow_mass_is_never_assigned_to_a_date():
    table = shared.build_tail_distribution({"faulty_lift": [8] * 20 + [500] * 20})
    cdf = shared.extended_cdf([0, 0, 0, 0, 0, 0, 0, 0, 1.0], table, "faulty_lift")
    assert abs(cdf[-1] - 0.5) < 1e-9
    assert shared.bound_offset(cdf, 0.95) is None


def test_sparse_category_falls_back_to_all_categories():
    table = shared.build_tail_distribution(
        {"staff_issue": [9] * 5, "faulty_lift": [10] * 40}
    )
    cdf = shared.extended_cdf([0] * 8 + [1.0], table, "staff_issue")
    assert shared.bound_offset(cdf, 0.95) == 10
    nothing = shared.build_tail_distribution({"staff_issue": [9] * 5})
    assert len(shared.extended_cdf([0] * 8 + [1.0], nothing, "staff_issue")) == 8


def test_round_up_to_service_day():
    assert shared.round_up_to_service_day(start_block_index=1, offset=3) == 3
    # daytime (1) + 8 blocks is daytime two days on: push to that day's overnight
    assert shared.round_up_to_service_day(start_block_index=1, offset=8) == 10
    assert shared.round_up_to_service_day(start_block_index=1, offset=10) == 10
    assert shared.round_up_to_service_day(start_block_index=3, offset=9) == 12
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_prediction_shared.py -q`
Expected: collection error, `ImportError: cannot import name 'prediction_shared' from 'incidents'`.

- [ ] **Step 3: Write the module**

Create `incidents/prediction_shared.py`:

```python
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
# fixed. 0.95 measured about 96% right first time in the 2026-11 backtest.
PREDICTION_COVERAGE_TARGET = 0.95

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


def has_stated_end(text, information=False):
    """True when a fault says when it will end, so it is never predicted."""
    return bool(information) or "until " in (text or "").lower()


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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_prediction_shared.py -q`
Expected: `12 passed`.

Run: `.venv/bin/python -m pytest -q`
Expected: `164 passed`.

- [ ] **Step 5: Commit**

```bash
git add incidents/prediction_shared.py tests/test_prediction_shared.py
git commit -m "Add prediction logic shared by the app and the training script

Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>"
```

---

### Task 2: A first prediction is final

Remove revised predictions and stop replacing a prediction when TfL edits the fault text.

**Files:**
- Modify: `incidents/management/commands/update_incidents.py` (imports, `refresh_prediction`, `refresh_revised_predictions`, the text-changed branch in `consolidate_incidents`, the trailing `refresh_revised_predictions()` call)
- Modify: `incidents/ml.py` (delete `predict_remaining_duration`)
- Modify: `incidents/views.py` (`_prepare_incidents`, `_api_prediction`, `_prediction_stats`)
- Modify: `templates/home.html:48`
- Test: `tests/test_incidents.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `refresh_prediction(incident)` writes only `estimated_duration` and `prediction_confidence`. `incidents.ml.predict_remaining_duration` and `update_incidents.refresh_revised_predictions` no longer exist. Prepared incidents no longer have `prediction_is_revised`. The API's `prediction["revised"]` is always `False`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_incidents.py`, class `ConsolidationTests`, replace the whole of `test_consolidate_refreshes_prediction_after_attaching_report` with:

```python
    def _incident_with_reworded_report(self, **incident_overrides):
        parent = self.create_parent_station("Green Park")
        first_report = self.create_report(
            parent,
            text="Lift unavailable on the northbound platform",
            start_time=timezone.now() - timedelta(hours=4),
        )
        incident = self.create_incident(
            parent,
            reports=[first_report],
            text=first_report.text,
            start_time=first_report.start_time,
            **incident_overrides,
        )
        self.create_report(
            parent,
            text="Lift unavailable on the northbound platform.",
            source=Report.SOURCE_TFLAPI_V2,
            start_time=timezone.now() - timedelta(hours=1),
        )
        return incident

    def test_consolidate_keeps_first_prediction_when_text_changes(self):
        incident = self._incident_with_reworded_report(
            estimated_duration=timedelta(hours=9),
            prediction_confidence=0.95,
        )

        with patch(
            "incidents.management.commands.update_incidents.predict_duration",
            return_value=(timedelta(hours=30), 0.95),
        ) as predict_mock, patch(
            "incidents.management.commands.update_incidents.send_tweet"
        ), patch(
            "incidents.management.commands.update_incidents.send_bluesky_messages"
        ):
            consolidate_incidents()

        incident.refresh_from_db()
        predict_mock.assert_not_called()
        self.assertEqual(incident.estimated_duration, timedelta(hours=9))
        self.assertEqual(incident.reports.count(), 2)

    def test_consolidate_predicts_on_text_change_when_there_is_no_prediction(self):
        incident = self._incident_with_reworded_report()
        observed = {}

        def fake_predict(updated_incident):
            observed["reports_count"] = updated_incident.reports.count()
            return timedelta(hours=4), 0.95

        with patch(
            "incidents.management.commands.update_incidents.predict_duration",
            side_effect=fake_predict,
        ), patch(
            "incidents.management.commands.update_incidents.send_tweet"
        ) as send_tweet_mock, patch(
            "incidents.management.commands.update_incidents.send_bluesky_messages"
        ) as send_bluesky_messages_mock:
            consolidate_incidents()

        incident.refresh_from_db()
        self.assertEqual(observed["reports_count"], 2)
        self.assertEqual(incident.estimated_duration, timedelta(hours=4))
        self.assertEqual(incident.prediction_confidence, 0.95)
        send_tweet_mock.assert_not_called()
        send_bluesky_messages_mock.assert_not_called()
```

In class `ViewAndCommandTests`, replace the whole of `test_api_incidents_flags_revised_prediction` with:

```python
    def test_api_incidents_ignores_stale_revised_columns(self):
        parent = self.create_parent_station("Holborn", naptan_id="HOL")
        self.create_incident(
            parent,
            text="Faulty lift",
            start_time=timezone.now() - timedelta(hours=1),
            estimated_duration=timedelta(hours=5),
            prediction_confidence=0.8,
            revised_estimated_duration=timedelta(hours=9),
            revised_prediction_confidence=0.7,
        )

        [issue] = self._api_payload_with_open_policy()["issues"]

        prediction = issue["prediction"]
        self.assertIs(prediction["revised"], False)
        self.assertEqual(prediction["confidence_pct"], 80)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_incidents.py -q -k "keeps_first_prediction or ignores_stale_revised"`
Expected: 2 failed. The first because `predict_duration` was called; the second because `confidence_pct` is 70.

- [ ] **Step 3: Stop making and showing revisions**

In `incidents/management/commands/update_incidents.py`:

Change the import on line 13 to:

```python
from incidents.ml import predict_duration
```

Replace `refresh_prediction` and delete `refresh_revised_predictions` entirely, leaving:

```python
def refresh_prediction(incident):
    """Store a fault's first prediction.

    Called once per fault. The first date shown is the one we are scored on,
    so it is never recomputed afterwards.
    """
    duration, confidence = predict_duration(incident)
    incident.estimated_duration = duration
    incident.prediction_confidence = confidence
    incident.save(update_fields=["estimated_duration", "prediction_confidence"])
```

In `consolidate_incidents`, replace the comment block that begins `# Only (re)compute the ML prediction when it can actually` with:

```python
            # A fault is predicted once, when it first appears. The first
            # date shown is the one we are scored on, so a later text edit
            # never replaces it; it only fills in a fault that has none.
```

and replace

```python
                if text_changed:
                    should_predict = True
```

with

```python
                if text_changed and incident.estimated_duration is None:
                    should_predict = True
```

Delete the last line of `consolidate_incidents`, `refresh_revised_predictions()`, and the blank line before it.

In `incidents/ml.py`, delete the whole `predict_remaining_duration` function.

In `incidents/views.py`, in `_prepare_incidents`, replace everything from the comment `# For an unresolved incident that has overrun its original bound, a` down to and including the `else:` branch that sets `display_confidence` with:

```python
        display_duration = incident.estimated_duration
        display_confidence = incident.prediction_confidence
```

In `_api_prediction`, change `"revised": incident.prediction_is_revised,` to:

```python
        # Always False: a first prediction is final. Kept so existing API
        # clients that read the field keep working.
        "revised": False,
```

In `_prediction_stats`, delete these lines from the `.only(...)` call:

```python
            # _prepare_incidents reads these for every open incident; leaving
            # them deferred costs one or two queries per row.
            "revised_estimated_duration",
            "revised_prediction_confidence",
```

In `templates/home.html`, replace line 48 with:

```html
                                                <span title="BETA: {{ issue.prediction_label|capfirst }} to be fixed {{ issue.expected_block }}{% if issue.prediction_accuracy_pct %} (recently similar predictions were right {{ issue.prediction_accuracy_pct }}% of the time){% endif %}">{{ issue.start_time }}</span>
```

- [ ] **Step 4: Remove the tests of removed behaviour**

In `tests/test_incidents.py` delete these tests entirely:

- `RevisedPredictionTests.test_refresh_revised_predictions_sets_overdue_and_clears_stale`
- `RevisedPredictionTests.test_prepare_incidents_displays_revised_bound_instead_of_hiding`
- `ReviewFixRegressionTests.test_refresh_prediction_clears_a_stale_revision`
- `ReviewFixRegressionTests.test_revision_without_original_prediction_is_not_shown`
- `LongRunningDisplayTests.test_shown_revised_bound_wins_over_some_time_status`

In `ReviewFixRegressionTests.test_stats_query_count_does_not_grow_with_revised_open_incidents`, rename the test to `test_stats_query_count_does_not_grow_with_overdue_open_incidents` and delete its two `revised_...` keyword arguments.

- [ ] **Step 5: Run the suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (160 passed).

Run: `grep -rn "predict_remaining_duration\|refresh_revised_predictions\|prediction_is_revised" incidents templates tests`
Expected: no output.

Run: `grep -n "revised_" tests/test_incidents.py`
Expected: only the two keyword arguments in `test_api_incidents_ignores_stale_revised_columns`.

- [ ] **Step 6: Commit**

```bash
git add -A incidents templates tests
git commit -m "Make a fault's first prediction final and drop revised predictions

Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>"
```

---

### Task 3: Decode to a 95% bound with the tail table, and name the day

**Files:**
- Modify: `incidents/ml.py` (constants near the top, `format_block_slot` area, `_predict_duration`, `_decode_offset_bound`, the end of `_incident_offset_probs`)
- Modify: `incidents/views.py` (imports, `BETA_LONG_RANGE_DAYS`, `_beta_current_status_text`, `_prepare_incidents`)
- Modify: `templates/home.html:48` and `:90`
- Test: `tests/test_incidents.py`

**Interfaces:**
- Consumes from Task 1: `MAX_OFFSET_CLASS`, `PREDICTION_COVERAGE_TARGET`, `bound_offset`, `extended_cdf`, `fault_category`, `round_up_to_service_day`.
- Produces:
  - `incidents.ml._incident_offset_probs(incident)` returns `None` or a 4-tuple `(model, probs, tail, category)`.
  - `incidents.ml._decode_offset_bound(model, probs, incident, tail=None, category=None) -> (timedelta, float) | (None, None)`.
  - `incidents.ml.format_bound_label(start_time, expected_end_time, now) -> str`.
  - `incidents.ml.MAX_OFFSET_CLASS` and `incidents.ml.PREDICTION_COVERAGE_TARGET` remain importable from `incidents.ml`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_incidents.py`, delete these four tests from `RevisedPredictionTests`: `test_decode_conditions_on_elapsed_and_returns_later_bound`, `test_decode_returns_none_when_only_open_ended_tail_survives`, `test_decode_original_prediction_on_tail_class_is_not_shown`, `test_decode_returns_none_when_nothing_survives_the_floor`. Delete its `_fake_model` and `_unsaved_incident_at` helpers too, and rename the class to `OverdueStatsTests` (its two remaining tests are about the stats page).

Add this class directly above `OverdueStatsTests`:

```python
class ExtendedDecodeTests(StationFactoryMixin, TestCase):
    # 1 July 2026 is BST. 09:00 UTC is 10:00 local: the daytime block.
    START = datetime(2026, 7, 1, 9, 0, tzinfo=dt_timezone.utc)

    def _decode(self, probs_by_offset, tail=None, category="faulty_lift"):
        class _Model:
            classes_ = np.array(sorted(probs_by_offset))

        incident = Incident(
            station=self.create_parent_station("Decode"),
            text="Faulty lift",
            information=False,
            start_time=self.START,
            resolved=False,
        )
        probs = np.array([probs_by_offset[k] for k in sorted(probs_by_offset)])
        return incidents_ml._decode_offset_bound(
            _Model(), probs, incident, tail, category
        )

    def _tail(self, offsets):
        from incidents.prediction_shared import build_tail_distribution

        return build_tail_distribution({"faulty_lift": offsets})

    def test_bound_inside_the_classifier_range(self):
        duration, confidence = self._decode({0: 0.96, 5: 0.04})
        # End of the start block: 15:00 local, one second inside.
        self.assertEqual(duration, timedelta(hours=4, minutes=59, seconds=59))
        self.assertEqual(confidence, 0.95)

    def test_aim_is_ninety_five_percent(self):
        duration, _ = self._decode({0: 0.80, 1: 0.14, 2: 0.06})
        # 0.94 at offset 1 is not enough; offset 2 is the overnight block,
        # ending 05:00 local on 2 July.
        self.assertEqual(duration, timedelta(hours=18, minutes=59, seconds=59))

    def test_open_ended_class_without_a_tail_table_gives_no_prediction(self):
        self.assertEqual(self._decode({8: 1.0}), (None, None))

    def test_tail_table_turns_the_open_ended_class_into_a_day(self):
        duration, confidence = self._decode({8: 1.0}, self._tail([10] * 40))
        # Offset 10 from 1 July daytime is the overnight block of 3 July,
        # which ends 05:00 local on 4 July.
        self.assertEqual(
            duration, timedelta(days=2, hours=18, minutes=59, seconds=59)
        )
        self.assertEqual(confidence, 0.95)

    def test_day_level_bound_is_rounded_up_to_the_end_of_that_day(self):
        duration, _ = self._decode({8: 1.0}, self._tail([9] * 40))
        # Offset 9 is 3 July evening; the promise is the end of 3 July.
        self.assertEqual(
            duration, timedelta(days=2, hours=18, minutes=59, seconds=59)
        )

    def test_no_prediction_when_the_aim_is_beyond_the_horizon(self):
        self.assertEqual(self._decode({8: 1.0}, self._tail([500] * 40)), (None, None))


class BoundLabelTests(SimpleTestCase):
    def test_short_bounds_keep_block_wording_and_long_ones_name_the_day(self):
        from zoneinfo import ZoneInfo

        from incidents.ml import format_bound_label

        london = ZoneInfo("Europe/London")
        start = datetime(2026, 11, 10, 12, 0, tzinfo=london)       # Tuesday
        short = datetime(2026, 11, 11, 9, 59, 59, tzinfo=london)   # +3 blocks
        by_thursday = datetime(2026, 11, 13, 4, 59, 59, tzinfo=london)
        far = datetime(2026, 11, 21, 4, 59, 59, tzinfo=london)

        def label(end, day, hour=13):
            now = datetime(2026, 11, day, hour, 0, tzinfo=london)
            return format_bound_label(start, end, now)

        self.assertEqual(label(short, 10), "tomorrow morning")
        self.assertEqual(label(by_thursday, 10), "the end of Thursday")
        self.assertEqual(label(by_thursday, 11), "the end of tomorrow")
        self.assertEqual(label(by_thursday, 12), "the end of today")
        self.assertEqual(label(far, 10), "the end of Fri 20 Nov")
        self.assertEqual(label(by_thursday, 14), "the end of Thu 12 Nov")
```

In `ViewAndCommandTests`, replace the last line of `test_prepare_incidents_uses_more_than_a_week_for_long_range_prediction`, and rename the test to `test_prepare_incidents_names_the_day_for_long_range_prediction`:

```python
        self.assertTrue(incidents[0].show_prediction)
        self.assertTrue(
            incidents[0].beta_current_status.startswith("Very likely by the end of "),
            incidents[0].beta_current_status,
        )
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_incidents.py -q -k "ExtendedDecodeTests or BoundLabelTests or names_the_day"`
Expected: failures. `ExtendedDecodeTests` fail with `TypeError` (the fifth argument) or a wrong duration, `BoundLabelTests` with `ImportError`, and the long-range test with `More than a week`.

- [ ] **Step 3: Rewrite the decode in `incidents/ml.py`**

Add to the imports at the top, below `from incidents.text_features import normalise_incident_text`:

```python
from incidents.prediction_shared import (
    MAX_OFFSET_CLASS,
    PREDICTION_COVERAGE_TARGET,
    bound_offset,
    extended_cdf,
    fault_category,
    round_up_to_service_day,
)
```

Delete the local `MAX_OFFSET_CLASS = 8` assignment with the comment block above it, and the local `PREDICTION_COVERAGE_TARGET = 0.75` assignment with the comment block above it. Put this comment where they were:

```python
# MAX_OFFSET_CLASS and PREDICTION_COVERAGE_TARGET live in
# incidents/prediction_shared.py so the app and ml/train.py cannot disagree.
# Everything at or beyond MAX_OFFSET_CLASS blocks is one open-ended class;
# the model bundle's tail table says how long such faults went on to last.
```

Directly below `format_block_slot`, add:

```python
def format_bound_label(start_time, expected_end_time, now):
    """Human label for a "fixed by" bound, read after the word "by".

    A bound two or more days after the fault began is only day-level, so it
    names the day ("the end of Tuesday") instead of a five-hour block.
    """
    if block_index(expected_end_time) - block_index(start_time) < MAX_OFFSET_CLASS:
        return format_block_slot(time_block_slot(expected_end_time), now)
    day, _ = time_block_slot(expected_end_time)
    delta = (day - _to_local(now).date()).days
    if delta == 0:
        return "the end of today"
    if delta == 1:
        return "the end of tomorrow"
    if 1 < delta < 7:
        return f"the end of {day.strftime('%A')}"
    return f"the end of {day.strftime('%a %-d %b')}"
```

Replace `_predict_duration` and `_decode_offset_bound` with:

```python
def _predict_duration(incident):
    """Returns (timedelta, confidence) or (None, None).

    The model is a classifier over "block offset": how many coarse time
    blocks after the incident's start block it resolves in. The prediction
    is the earliest block by which there is a PREDICTION_COVERAGE_TARGET
    chance it is resolved, so the duration is a one-sided "fixed by" bound
    and ``confidence`` is the cumulative probability under that bound.
    """
    result = _incident_offset_probs(incident)
    if result is None:
        return None, None
    model, probs, tail, category = result
    return _decode_offset_bound(model, probs, incident, tail, category)


def _decode_offset_bound(model, probs, incident, tail=None, category=None):
    """Turn a block-offset probability vector into a (duration, confidence).

    Returns (None, None) when there is no honest bound: the target is only
    reached in the open-ended class and there is no tail table to say how
    long such faults last, or it is not reached within the table's horizon.
    ml/train.py's decode_bound mirrors this for cross-validation.
    """
    class_probs = [0.0] * (MAX_OFFSET_CLASS + 1)
    for cls, probability in zip(model.classes_, probs):
        cls = int(cls)
        if 0 <= cls <= MAX_OFFSET_CLASS:
            class_probs[cls] = float(probability)

    cdf = extended_cdf(class_probs, tail, category)
    offset = bound_offset(cdf)
    if offset is None:
        return None, None

    start_idx = block_index(incident.start_time)
    offset = round_up_to_service_day(start_idx, offset)
    confidence = cdf[min(offset, len(cdf) - 1)]
    confidence = max(0.05, min(0.95, confidence))
    return _offset_to_duration(start_idx, offset, incident.start_time, confidence)
```

In `_incident_offset_probs`:

- change the `except ImportError:` branch near the end from `return None, None` to `return None`;
- replace the final comment block and `return model, probs` with:

```python
    # The blended probability vector over block-offset classes, plus what
    # the decode needs to extend it past the open-ended class.
    return model, probs, data.get("tail_distribution"), fault_category(feature_text)
```

- update its docstring's first line to `Build features for ``incident`` and return (model, probs, tail, category).`

Delete `_prediction_category_from_text` and change its one caller in `_historical_baseline_probs` from `_prediction_category_from_text(text)` to `fault_category(text)`.

- [ ] **Step 4: Use the label in `incidents/views.py`**

In the `from incidents.ml import (...)` block add `format_bound_label,` after `format_block_slot,`.

Delete the line `BETA_LONG_RANGE_DAYS = 7`.

In `_beta_current_status_text`, delete:

```python
    if expected_end_time > now + timedelta(days=BETA_LONG_RANGE_DAYS):
        return "More than a week"
```

In `_prepare_incidents`, replace

```python
            incident.expected_block = format_block_slot(
                time_block_slot(incident.expected_end_time), now
            )
```

with

```python
            incident.expected_block = format_bound_label(
                incident.start_time, incident.expected_end_time, now
            )
```

In `templates/home.html` line 48, change `to be fixed {{ issue.expected_block }}` to `to be fixed by {{ issue.expected_block }}`.

In `templates/home.html` line 90, replace the `title="..."` attribute's three branches so the whole line reads:

```html
                                            <span title="BETA: {% if issue.prediction_outcome == 'exact' %}We correctly thought it was {{ issue.prediction_label }} to be fixed by {{ issue.expected_block }}{% elif issue.prediction_outcome == 'near' %}We were nearly right: fixed {{ issue.actual_block }}, just after our predicted time ({{ issue.expected_block }}){% else %}Fixed {{ issue.actual_block }}, we had thought it was {{ issue.prediction_label }} to be fixed by {{ issue.expected_block }}{% endif %}{% if issue.prediction_accuracy_pct %} (recently similar predictions were right {{ issue.prediction_accuracy_pct }}% of the time){% endif %}">{{ issue.end_time }}</span>
```

- [ ] **Step 5: Update the comment in an existing test**

In `PredictionFeatureTests.test_predict_duration_blends_in_historical_station_baseline`, change `The 75%` in the trailing comment to `The 95%`. The asserted values do not change.

- [ ] **Step 6: Run the suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (163 passed).

- [ ] **Step 7: Commit**

```bash
git add -A incidents templates tests
git commit -m "Decode predictions to a 95% bound using the tail table, name the day

Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>"
```

---

### Task 4: Version 4 features at prediction time

**Files:**
- Modify: `incidents/ml.py` (`_incident_offset_probs`, new `_history_features`)
- Test: `tests/test_incidents.py` (`PredictionFeatureTests`)

**Interfaces:**
- Consumes from Task 1: `has_stated_end`, `fault_category`, `fault_key`, `history_features`. From Task 3: the 4-tuple return of `_incident_offset_probs`.
- Produces: with a bundle whose `metadata.feature_version >= 4`, the feature row offered to the model contains every name in `HISTORY_FEATURE_NAMES` plus `minute_of_day` and `network_started_24h`, and only faults with a stated end are skipped. Task 9 trains on exactly these names.

- [ ] **Step 1: Write the failing tests**

Add to `PredictionFeatureTests`:

```python
    def _capturing_model(self, feature_names, captured):
        class FakeModel:
            classes_ = np.array([0])
            feature_names_in_ = np.array(feature_names, dtype=object)

            def predict_proba(self, x):
                captured["features"] = dict(zip(self.feature_names_in_, x[0]))
                return np.array([[1.0]])

        return FakeModel()

    def test_version_4_model_gets_history_from_previous_faults(self):
        if np is None:
            self.skipTest("numpy not installed")

        def utc(day, hour, month=7):
            return datetime(2026, month, day, hour, 0, tzinfo=dt_timezone.utc)

        lift_text = (
            "Step free access is not available due to a faulty lift. "
            "Call us on 0343 222 1234"
        )
        parent = self.create_parent_station("Canary Wharf")
        other = self.create_parent_station("Elsewhere")
        # Same fault, 28 Jun 08:00 local (morning) to 30 Jun 10:00 local
        # (daytime): 2 days and one block, so 9 blocks.
        self.create_incident(
            parent, text=lift_text, resolved=True,
            start_time=utc(28, 7, month=6), end_time=utc(30, 9, month=6),
        )
        self.create_incident(
            parent, text="Unavailability of station staff", resolved=True,
            start_time=utc(1, 6), end_time=utc(1, 7),
        )
        self.create_incident(
            parent, text="Another fault", resolved=False, start_time=utc(1, 8),
        )
        self.create_incident(
            other, text="Faulty lift", resolved=False, start_time=utc(1, 5),
        )
        incident = Incident(
            station=parent,
            text=lift_text.replace("0343 222 1234", "").upper(),
            information=False,
            start_time=datetime(2026, 7, 1, 9, 30, tzinfo=dt_timezone.utc),
            resolved=False,
        )

        captured = {}
        model = self._capturing_model(
            [
                "st_count", "st_open_now", "st_started_7d", "sc_count",
                "sf_count", "sf_last_off", "sf_days_since_last_end",
                "minute_of_day", "network_started_24h",
            ],
            captured,
        )
        with patch(
            "incidents.ml._load_model",
            return_value={"model": model, "metadata": {"feature_version": 4}},
        ):
            duration, _ = predict_duration(incident)

        features = captured["features"]
        self.assertIsNotNone(duration)
        self.assertEqual(features["st_count"], 2)
        self.assertEqual(features["st_open_now"], 1)
        self.assertEqual(features["st_started_7d"], 3)
        self.assertEqual(features["sc_count"], 1)
        self.assertEqual(features["sf_count"], 1)
        self.assertEqual(features["sf_last_off"], 9.0)
        self.assertAlmostEqual(features["sf_days_since_last_end"], 24.5 / 24)
        self.assertEqual(features["minute_of_day"], 10 * 60 + 30)
        self.assertEqual(features["network_started_24h"], 3)

    def test_what_counts_as_planned_work_depends_on_the_model_version(self):
        if np is None:
            self.skipTest("numpy not installed")
        parent = self.create_parent_station("Canada Water")

        def predicted(text, version, information=False):
            incident = Incident(
                station=parent, text=text, information=information,
                start_time=datetime(2026, 7, 1, 9, 30, tzinfo=dt_timezone.utc),
                resolved=False,
            )
            model = self._capturing_model(["station_id"], {})
            with patch(
                "incidents.ml._load_model",
                return_value={"model": model, "metadata": {"feature_version": version}},
            ):
                return predict_duration(incident)[0] is not None

        maintenance = "Step free access is not available due to planned maintenance."
        self.assertTrue(predicted(maintenance, 4))
        self.assertFalse(predicted(maintenance, 3))
        self.assertFalse(predicted("No step-free access until Sunday 22 November", 4))
        self.assertFalse(predicted("Lift works", 4, information=True))
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_incidents.py -q -k "history_from_previous_faults or depends_on_the_model_version"`
Expected: 2 failed. The first with `st_count` 0.0 (the feature is missing, so it defaults to 0); the second because planned maintenance is not predicted.

- [ ] **Step 3: Implement**

In `incidents/ml.py`, extend the `incidents.prediction_shared` import to:

```python
from incidents.prediction_shared import (
    MAX_OFFSET_CLASS,
    PREDICTION_COVERAGE_TARGET,
    bound_offset,
    extended_cdf,
    fault_category,
    fault_key,
    has_stated_end,
    history_features,
    round_up_to_service_day,
)
```

In `_incident_offset_probs`, replace

```python
    if _is_planned_work(incident):
        return None

    model = data["model"]
    vectorizer = data.get("vectorizer")
    historical_baselines = data.get("historical_baselines")
    metadata = data.get("metadata") or {}
    feature_version = metadata.get("feature_version", 1)
    use_v2_features = feature_version >= 2
```

with

```python
    model = data["model"]
    vectorizer = data.get("vectorizer")
    historical_baselines = data.get("historical_baselines")
    metadata = data.get("metadata") or {}
    feature_version = metadata.get("feature_version", 1)
    use_v2_features = feature_version >= 2

    # A version 4 model was trained on undated planned maintenance, so only
    # faults that state their own end are skipped. Older models never saw
    # planned work of any kind.
    if feature_version >= 4:
        if has_stated_end(incident.text, incident.information):
            return None
    elif _is_planned_work(incident):
        return None
```

Replace the block that starts with the comment `# Per-station historical stats from resolved incidents.` and ends with the `else:` branch setting `mean_dur = 0` and `mean_offset = 0` with:

```python
    # Versions 1 to 3: three station averages. Version 4 replaces them with
    # the richer history built in _history_features.
    count = 0
    mean_dur = 0
    mean_offset = 0
    if feature_version < 4:
        past_qs = Incident.objects.filter(
            station_filter,
            resolved=True,
            end_time__isnull=False,
        )
        if use_v2_features:
            past_qs = past_qs.filter(end_time__lte=incident.start_time)
        past = list(past_qs.values_list("start_time", "end_time"))
        count = len(past)
        if count:
            mean_dur = sum((e - s).total_seconds() for s, e in past) / count / 60
            offsets = [
                max(0, min(MAX_OFFSET_CLASS, block_index(e) - block_index(s)))
                for s, e in past
            ]
            mean_offset = sum(offsets) / count
```

Directly after the `features = {...}` dict literal, add:

```python
    if feature_version >= 4:
        features.update(
            _history_features(incident, station, feature_text, start_time_local)
        )
```

Add this function directly above `_incident_offset_probs`:

```python
def _history_features(incident, station, feature_text, start_time_local):
    """Version 4 history features for ``incident``.

    One query loads the station family's earlier faults; the summary itself
    is prediction_shared.history_features, the same code ml/train.py runs.
    """
    from incidents.models import Incident

    earlier = Incident.objects.filter(
        station_id__in=_station_family_ids(station.pk),
        start_time__lt=incident.start_time,
    )
    recent = Incident.objects.filter(
        start_time__gte=incident.start_time - timedelta(days=1),
        start_time__lt=incident.start_time,
    )
    if incident.pk:
        earlier = earlier.exclude(pk=incident.pk)
        recent = recent.exclude(pk=incident.pk)

    past = []
    for start, end, resolved, text in earlier.values_list(
        "start_time", "end_time", "resolved", "text"
    ):
        if resolved:
            if end is None or end <= start:
                continue  # bad data; the training export drops these too
            offset = block_index(end) - block_index(start)
        else:
            # An open fault's end_time, if set, is TfL's stated end, not a fix.
            end = offset = None
        past.append((start, end, offset, fault_category(text), fault_key(text)))

    features = history_features(
        past,
        incident.start_time,
        fault_category(feature_text),
        fault_key(feature_text),
    )
    features["minute_of_day"] = start_time_local.hour * 60 + start_time_local.minute
    features["network_started_24h"] = recent.count()
    return features
```

- [ ] **Step 4: Run the suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (165 passed).

- [ ] **Step 5: Commit**

```bash
git add incidents/ml.py tests/test_incidents.py
git commit -m "Feed version 4 models history from previous faults

Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>"
```

---

### Task 5: Accept version 4 model uploads

**Files:**
- Modify: `incidents/views.py` (`SUPPORTED_FEATURE_VERSIONS`, `_validate_model_payload`)
- Test: `tests/test_incidents.py` (`SecurityTests`)

**Interfaces:**
- Consumes: the bundle shape Task 9 produces: `{"model", "vectorizer", "historical_baselines", "tail_distribution": {"by_category": {...}, "all": {...}}, "metadata": {"feature_version": 4}}`.
- Produces: `/functions/upload_model` accepts that bundle and rejects a version 4 bundle without a tail table.

- [ ] **Step 1: Write the failing tests**

In `SecurityTests`, change the signature and payload of `_model_bytes`:

```python
    def _model_bytes(self, feature_version=3, as_dict=True, with_model=True, tail=None):
```

and add after the `payload = {...}` literal:

```python
        if tail is not None:
            payload["tail_distribution"] = tail
```

Add:

```python
    @override_settings(FUNCTIONS_SECRET_KEY="secret")
    def test_upload_model_accepts_version_4_with_a_tail_table(self):
        tail = {"by_category": {}, "all": {"total": 0, "counts": [0] * 113}}
        with patch("incidents.views.predict_duration") as pred:
            pred.cache_clear = MagicMock()
            response = self._upload(self._model_bytes(feature_version=4, tail=tail))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(MLModel.objects.count(), 1)

    @override_settings(FUNCTIONS_SECRET_KEY="secret")
    def test_upload_model_rejects_version_4_without_a_tail_table(self):
        response = self._upload(self._model_bytes(feature_version=4))
        self.assertEqual(response.status_code, 400)
        self.assertIn("tail_distribution", response.json()["error"])
        self.assertEqual(MLModel.objects.count(), 0)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_incidents.py -q -k "version_4_with or version_4_without"`
Expected: 2 failed. The first with status 400 (unsupported version); the second because the error does not mention `tail_distribution`.

- [ ] **Step 3: Implement**

In `incidents/views.py` change:

```python
SUPPORTED_FEATURE_VERSIONS = frozenset({1, 2, 3, 4})
```

In `_validate_model_payload`, directly after the `feature_version` check, add:

```python
    if metadata["feature_version"] >= 4:
        tail = payload.get("tail_distribution")
        if not isinstance(tail, dict) or "all" not in tail or "by_category" not in tail:
            return "Version 4 model payload is missing its tail_distribution"
```

- [ ] **Step 4: Run the suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (167 passed).

- [ ] **Step 5: Commit**

```bash
git add incidents/views.py tests/test_incidents.py
git commit -m "Accept version 4 model bundles, requiring their tail table

Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>"
```

---

### Task 6: Admit a miss; remove "likely to be some time"

**Files:**
- Modify: `incidents/views.py` (long-running constants and functions, `_prepare_incidents`, `_status_page_context`, `_api_prediction`, `_api_incident_rows`, `api_incidents`, `_prediction_stats`, imports)
- Modify: `templates/home.html:47-53`, `templates/beta.html:45-51`, `templates/stats.html:97-123` and `:145-160`
- Test: `tests/test_incidents.py`

**Interfaces:**
- Consumes: `_prepare_incidents` as left by Tasks 2 and 3.
- Produces:
  - `_prepare_incidents(queryset, prediction_policy=None, station_prediction_overrides=None, now=None)`: the `long_running_policy` parameter is gone. Every prepared incident has `prediction_overdue: bool`. Prepared incidents no longer have `long_running`.
  - `OVERDUE_STATUS_TEXT = "Taking longer than we predicted"` in `incidents.views`.
  - Every row in `/api/incidents/` has a boolean `prediction_overdue`. `kind` is always `"fixed_by"`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_incidents.py` delete `LONG_RUNNING_TEST_POLICY`, the whole `LongRunningPolicyTests` class and the whole `LongRunningDisplayTests` class. Keep `OPEN_PREDICTION_POLICY`. In their place add:

```python
class OverduePredictionTests(StationFactoryMixin, TestCase):
    def setUp(self):
        self.parent = self.create_parent_station("Overdue Park")

    def _incident(self, **overrides):
        values = {
            "text": "Faulty lift",
            "start_time": timezone.now() - timedelta(hours=6),
            "estimated_duration": timedelta(hours=1),
            "prediction_confidence": 0.95,
        }
        values.update(overrides)
        return self.create_incident(self.parent, **values)

    def _prepared(self, incident, policy=OPEN_PREDICTION_POLICY):
        [prepared] = _prepare_incidents([incident], prediction_policy=policy)
        return prepared

    def test_a_missed_prediction_is_admitted_without_a_new_date(self):
        prepared = self._prepared(self._incident())

        self.assertTrue(prepared.prediction_overdue)
        self.assertFalse(prepared.show_prediction)
        self.assertEqual(prepared.beta_current_status, "Taking longer than we predicted")
        self.assertIn("We predicted this would be fixed by", prepared.beta_current_status_title)
        self.assertIn(
            "90% of our predictions were right first time",
            prepared.beta_current_status_title,
        )

    def test_no_admission_inside_the_grace_period(self):
        prepared = self._prepared(
            self._incident(start_time=timezone.now() - timedelta(minutes=70))
        )
        self.assertFalse(prepared.prediction_overdue)
        self.assertIsNone(prepared.beta_current_status)

    def test_no_admission_for_a_prediction_the_site_would_not_show(self):
        hidden = {b: {"show_prediction": False, "count": 999} for b in range(0, 100, 10)}
        self.assertFalse(self._prepared(self._incident(), hidden).prediction_overdue)

    def test_no_admission_without_a_prediction_or_once_resolved(self):
        none = self._incident(estimated_duration=None, prediction_confidence=None)
        self.assertFalse(self._prepared(none).prediction_overdue)
        fixed = self._incident(resolved=True, end_time=timezone.now())
        self.assertFalse(self._prepared(fixed).prediction_overdue)

    def test_a_live_prediction_is_not_overdue(self):
        prepared = self._prepared(self._incident(estimated_duration=timedelta(hours=9)))
        self.assertFalse(prepared.prediction_overdue)
        self.assertTrue(prepared.show_prediction)

    def _get(self, path):
        with patch(
            "incidents.views._prediction_display_policy",
            return_value=OPEN_PREDICTION_POLICY,
        ), patch("incidents.views.get_last_updated", return_value="09:15 26 Mar"):
            return self.client.get(path)

    def test_api_flags_the_miss_and_gives_no_prediction(self):
        self._incident()
        self._incident(text="Fresh fault", estimated_duration=timedelta(hours=9))

        issues = {i["text"]: i for i in self._get("/api/incidents/").json()["issues"]}

        self.assertIsNone(issues["Faulty lift"]["prediction"])
        self.assertIs(issues["Faulty lift"]["prediction_overdue"], True)
        self.assertEqual(issues["Fresh fault"]["prediction"]["kind"], "fixed_by")
        self.assertIs(issues["Fresh fault"]["prediction_overdue"], False)

    def test_home_and_beta_pages_admit_the_miss(self):
        self._incident()
        for path in ("/", "/beta/"):
            response = self._get(path)
            self.assertContains(response, "Taking longer than we predicted")

    def test_stats_page_no_longer_has_a_long_running_section(self):
        self._incident()
        response = self._get("/stats/")
        self.assertNotContains(response, "Long-Running Faults")
        self.assertNotContains(response, "likely to be some time")
```

In `ViewAndCommandTests.test_stats_uses_small_constant_number_of_queries`, replace the last three lines (the two-line comment and the assertion) with:

```python
        self.assertLessEqual(len(queries), 3)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_incidents.py -q -k "OverduePredictionTests or small_constant_number"`
Expected: failures with `AttributeError: 'Incident' object has no attribute 'prediction_overdue'`, a missing `prediction_overdue` key in the API, and 4 queries on the stats page.

- [ ] **Step 3: Remove "some time" from `incidents/views.py`**

Delete:

- the comment block that begins `# "Likely to be some time". The model's classes stop at ~2 days`, and the constants `LONG_RUNNING_AGE_BANDS`, `LONG_RUNNING_HORIZON_BLOCKS`, `LONG_RUNNING_WINDOW_DAYS`;
- the functions `_long_running_band`, `_long_running_policy_from_rows`, `_long_running_policy`, `_long_running_age_label`, `_long_running_status_text`, `_long_running_status_title`, `_long_running_status`.

In `_prepare_incidents`, remove the `long_running_policy=None,` parameter and replace

```python
        incident.beta_meter_pct = incident.prediction_accuracy_pct
        incident.long_running = _long_running_status(
            incident, long_running_policy, now
        )
        if incident.long_running:
            incident.beta_current_status = _long_running_status_text(
                incident.long_running["label"]
            )
            incident.beta_current_status_title = _long_running_status_title(
                incident.beta_current_status, incident.long_running["accuracy_pct"]
            )
            incident.beta_meter_pct = incident.long_running["accuracy_pct"]
```

with

```python
        incident.beta_meter_pct = incident.prediction_accuracy_pct
        # A missed prediction is admitted, never replaced: once the promised
        # time (plus the scoring grace) has passed on a prediction we were
        # showing, say so and offer no second date.
        incident.prediction_overdue = bool(
            not incident.resolved
            and not incident.information
            and incident.expected_end_time is not None
            and incident.prediction_bucket_shown
            and incident.expected_end_time + PREDICTION_BOUNDARY_GRACE < now
        )
        if incident.prediction_overdue:
            incident.beta_current_status = OVERDUE_STATUS_TEXT
            incident.beta_current_status_title = _overdue_status_title(incident)
```

Directly above `_prediction_hidden_reason`, add:

```python
OVERDUE_STATUS_TEXT = "Taking longer than we predicted"


def _overdue_status_title(incident):
    title = f"We predicted this would be fixed by {incident.expected_block}."
    if incident.prediction_accuracy_pct is not None:
        title += (
            f" Recently {incident.prediction_accuracy_pct}% of our predictions "
            "were right first time."
        )
    return title
```

In `_status_page_context`, `api_incidents` and `_prediction_stats`, delete the `long_running_policy=...` argument from each `_prepare_incidents(...)` call. In `_prediction_stats` also delete the line `long_running_policy = _long_running_policy(now)`.

In `_api_prediction`, delete the `if getattr(incident, "long_running", None):` block and replace the docstring with:

```python
    """The beta prediction exactly as the site shows it, or None.

    Mirrors the homepage: information notices never carry one, and a
    prediction the display policy hides (sparse or inaccurate bucket, time
    already passed, no usable prediction) is omitted rather than exposed, so
    API clients can't show what the site deliberately hides.

    ``include_less_reliable`` (``?predictions=all``) is the exception: current
    predictions hidden only by a sparse or inaccurate bucket are included with
    ``reliable: false`` and likelihood "maybe". Past-due and missing
    predictions are never exposed; a past-due one is reported by the row's
    ``prediction_overdue`` flag instead.
    """
```

Replace `_api_incident_rows` with:

```python
def _api_incident_rows(incidents, include_less_reliable=False, now=None):
    return [
        {
            **{field: getattr(incident, field) for field in API_INCIDENT_FIELDS},
            "prediction": _api_prediction(incident, include_less_reliable, now),
            "prediction_overdue": incident.prediction_overdue,
        }
        for incident in incidents
    ]
```

In `_prediction_stats`:

- delete the `current_prediction_long_running_count = sum(...)` statement;
- replace the comment and list comprehension that build `hidden_reasons` with

```python
    hidden_reasons = [issue.prediction_hidden_reason for issue in current_issues]
```

- delete the `long_running_rows = [...]` statement;
- delete these three entries from the returned dict: `"current_prediction_long_running_count"`, `"long_running_rows"`, `"long_running_window_days"`.

In the `from incidents.ml import (...)` block, remove `BLOCKS_PER_DAY,`, `block_index,` and `is_planned_work_text,`. Confirm they are unused:

Run: `grep -n "BLOCKS_PER_DAY\|block_index\|is_planned_work_text\|long_running" incidents/views.py`
Expected: no output.

- [ ] **Step 4: Update the templates**

`templates/home.html`: replace lines 47 to 53 (the `{% if issue.show_prediction %}` block through its `{% endif %}`) with:

```html
                                            {% if issue.show_prediction %}
                                                <span title="BETA: {{ issue.prediction_label|capfirst }} to be fixed by {{ issue.expected_block }}{% if issue.prediction_accuracy_pct %} (recently similar predictions were right {{ issue.prediction_accuracy_pct }}% of the time){% endif %}">{{ issue.start_time }}</span>
                                            {% elif issue.prediction_overdue %}
                                                <span title="BETA: {{ issue.beta_current_status }}. {{ issue.beta_current_status_title }}">{{ issue.start_time }}</span>
                                            {% else %}
                                                {{ issue.start_time }}
                                            {% endif %}
```

`templates/beta.html`: on line 45 change `{% if issue.show_prediction or issue.long_running %}` to `{% if issue.show_prediction or issue.prediction_overdue %}`. On line 51 replace

```html
{% if issue.long_running %}Estimate from recent long-running faults{% else %}AI prediction{% endif %}
```

with `AI prediction`.

`templates/stats.html`:

- delete the whole `{% if long_running_rows %} ... {% endif %}` block (the "Long-Running Faults" heading, paragraph and table);
- delete the table row whose first cell is `Shown in beta: likely to be some time`;
- change the cell text `Hidden: predicted time has passed` to `Taking longer than predicted`;
- in that table's `<caption>`, change `hidden because the predicted time has already passed` to `taking longer than predicted`.

- [ ] **Step 5: Run the suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add -A incidents templates tests
git commit -m "Admit a missed prediction on the fault and remove 'some time'

Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>"
```

---

### Task 7: Show a prediction only while it is right nine times in ten

**Files:**
- Modify: `incidents/views.py` (constants, `_wilson_lower_bound`, `_prediction_window_metrics_from_incidents`, `_prediction_display_policy_from_window_metrics`, `_prediction_station_overrides`, `_prepare_incidents`, `_status_page_context`, `api_incidents`, `_prediction_stats`, imports)
- Modify: `templates/stats.html:54-60`
- Test: `tests/test_incidents.py`

**Interfaces:**
- Consumes: `_prepare_incidents` as left by Task 6.
- Produces:
  - `PREDICTION_BUCKET_MIN_SAMPLES = 30`, `PREDICTION_MIN_ACCURACY = 0.90` in `incidents.views`.
  - `_prepare_incidents(queryset, prediction_policy=None, now=None)`: the `station_prediction_overrides` parameter is gone.
  - Bucket entries no longer have `lower_bound` or `lower_bound_pct`.
  - Test helper `StationFactoryMixin.create_track_record(station, right=30, wrong=0, confidence=0.95)`.

- [ ] **Step 1: Add the test helper and write the failing tests**

In `tests/test_incidents.py`, add `from types import SimpleNamespace` to the imports, and in the `from incidents.views import (...)` block remove `_prediction_station_overrides,` and add `_prediction_window_metrics_from_incidents,`.

Add to `StationFactoryMixin`:

```python
    def create_track_record(self, station, right=30, wrong=0, confidence=0.95):
        """Resolved predictions from the last fortnight.

        ``right`` of them were fixed inside the predicted time and ``wrong``
        overran it. Six hours apart, so 60 of them still fit in 30 days.
        """
        now = timezone.now()
        for i in range(right + wrong):
            start = now - timedelta(days=1, hours=6 * i)
            held = i < right
            self.create_incident(
                station,
                text=f"Track record {confidence} {i}",
                resolved=True,
                start_time=start,
                end_time=start + timedelta(hours=1 if held else 12),
                estimated_duration=timedelta(hours=2),
                prediction_confidence=confidence,
            )
```

Add to `ViewAndCommandTests`:

```python
    def _bucket(self, right, wrong):
        now = timezone.now()
        start = now - timedelta(days=2)
        rows = [
            SimpleNamespace(
                start_time=start,
                end_time=start + timedelta(hours=1 if i < right else 12),
                estimated_duration=timedelta(hours=2),
                prediction_confidence=0.95,
            )
            for i in range(right + wrong)
        ]
        return _prediction_window_metrics_from_incidents(rows, now, (30,))[30][
            "buckets"
        ][90]

    def test_display_rule_needs_thirty_predictions_and_ninety_percent_right(self):
        self.assertFalse(self._bucket(right=29, wrong=0)["show_prediction"])
        self.assertTrue(self._bucket(right=30, wrong=0)["show_prediction"])
        self.assertTrue(self._bucket(right=27, wrong=3)["show_prediction"])
        self.assertFalse(self._bucket(right=26, wrong=4)["show_prediction"])
        self.assertEqual(self._bucket(right=27, wrong=3)["label"], "very likely")
        self.assertNotIn("lower_bound", self._bucket(right=27, wrong=3))

    def test_stats_explains_the_right_first_time_rule(self):
        parent = self.create_parent_station("Green Park")
        self.create_track_record(parent, right=29, wrong=1)

        with patch("incidents.views.get_last_updated", return_value="09:15 26 Mar"):
            response = self.client.get("/stats/")

        self.assertContains(response, "Right first time: 29 / 30 (97%)")
        self.assertContains(response, "at least 30 recent predictions")
        self.assertContains(response, "at least 90% of them")
        self.assertNotContains(response, "narrow exception")
```

- [ ] **Step 2: Rewrite the existing tests that encode the old rule**

Delete `ViewAndCommandTests.test_prepare_incidents_allows_40_bucket_for_station_with_strong_history`.

Replace the whole of `test_prepare_incidents_only_shows_predictions_from_stronger_buckets` with:

```python
    def test_prepare_incidents_only_shows_buckets_right_nine_times_in_ten(self):
        parent = self.create_parent_station("Waterloo")
        self.create_track_record(parent, right=28, wrong=2, confidence=0.95)  # 93%
        self.create_track_record(parent, right=26, wrong=4, confidence=0.85)  # 87%
        weak = self.create_incident(
            parent,
            text="Weak bucket outage",
            estimated_duration=timedelta(hours=2),
            prediction_confidence=0.85,
        )
        strong = self.create_incident(
            parent,
            text="Strong bucket outage",
            estimated_duration=timedelta(hours=2),
            prediction_confidence=0.95,
        )
        policy = _prediction_bucket_metrics(
            timezone.now() - timedelta(days=365)
        )["buckets"]

        incidents = _prepare_incidents(
            Incident.objects.filter(id__in=[weak.id, strong.id]).order_by("id"),
            prediction_policy=policy,
        )

        self.assertFalse(incidents[0].show_prediction)
        self.assertIsNone(incidents[0].prediction_label)
        self.assertTrue(incidents[1].show_prediction)
        self.assertEqual(incidents[1].prediction_label, "very likely")
        self.assertEqual(incidents[1].prediction_accuracy_pct, 93)
```

In `test_prediction_display_policy_falls_back_when_recent_bucket_is_sparse`, replace the first `for i in range(10):` loop (the one creating `Recent good` incidents) with:

```python
        self.create_track_record(parent, right=20, confidence=0.65)
```

and change the second loop's `range(5)` to `range(10)`. The assertions stay.

In `test_prediction_display_policy_keeps_recent_bad_bucket_hidden`, replace the first loop (`Recent miss`, with its comment) with:

```python
        self.create_track_record(parent, right=0, wrong=30, confidence=0.65)
```

and change the second loop's `range(20)` to `range(40)`. The assertions stay.

In each of these four tests, replace the whole `base = ...` assignment (with its comment) and the `for i in range(15):` loop that creates `Resolved good` incidents with the single line `self.create_track_record(parent, confidence=0.65)`:

- `test_prepare_incidents_hides_overdue_current_prediction`
- `test_prepare_incidents_names_the_day_for_long_range_prediction`
- `test_prepare_incidents_marks_resolved_miss_as_wrong`
- `test_beta_renders_inline_status_next_to_event_time`

In `test_stats_includes_current_prediction_visibility_breakdown`, replace the `base = ...` line and the `for i in range(15):` loop with:

```python
        self.create_track_record(parent, confidence=0.65)
        self.create_track_record(parent, right=0, wrong=30, confidence=0.25)
```

In `test_stats_includes_hidden_low_accuracy_diagnostics`, replace the `base = ...` line and the `for i in range(15):` loop with:

```python
        self.create_track_record(tube_station, right=0, wrong=30, confidence=0.25)
        self.create_track_record(dlr_station, right=0, wrong=30, confidence=0.35)
```

and change both `recent_prediction_count` assertions from `15` to `30`.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_incidents.py -q`
Expected: failures in `test_display_rule_needs_thirty_predictions_and_ninety_percent_right` (29 predictions are shown; 26 of 30 is shown) and `test_stats_explains_the_right_first_time_rule` (old headline), and an `ImportError` is not expected because `_prediction_window_metrics_from_incidents` already exists.

- [ ] **Step 4: Change the rule in `incidents/views.py`**

Replace the constants:

```python
PREDICTION_BUCKET_MIN_SAMPLES = 30
# Shown only while at least this share of recent predictions in the bucket
# were right first time. The model aims higher (about 96%); this is the floor
# below which we stop showing predictions at all.
PREDICTION_MIN_ACCURACY = 0.90
```

deleting `PREDICTION_BUCKET_MIN_LOWER_BOUND`, `PREDICTION_STATION_OVERRIDE_BUCKET`, `PREDICTION_STATION_OVERRIDE_WINDOW_DAYS`, `PREDICTION_STATION_OVERRIDE_MIN_SAMPLES` and `PREDICTION_STATION_OVERRIDE_MIN_ACCURACY`.

Delete the functions `_wilson_lower_bound` and `_prediction_station_overrides`, and the import `from math import sqrt`.

In `_prediction_window_metrics_from_incidents`, replace the body of the `for bucket in range(0, 100, 10):` loop with:

```python
            entry = buckets.setdefault(bucket, {"count": 0, "correct": 0})
            accuracy = entry["correct"] / entry["count"] if entry["count"] else None
            show_prediction = (
                entry["count"] >= PREDICTION_BUCKET_MIN_SAMPLES
                and accuracy >= PREDICTION_MIN_ACCURACY
            )
            entry.update(
                {
                    "bucket": bucket,
                    "accuracy": accuracy,
                    "accuracy_pct": (
                        round(accuracy * 100) if accuracy is not None else None
                    ),
                    "show_prediction": show_prediction,
                    "label": (
                        _prediction_confidence_label(accuracy)
                        if show_prediction
                        else None
                    ),
                }
            )
```

In `_prediction_display_policy_from_window_metrics`, replace the `show_prediction = (...)` statement with:

```python
        show_prediction = (
            selected_entry["count"] >= PREDICTION_BUCKET_MIN_SAMPLES
            and selected_entry["accuracy"] >= PREDICTION_MIN_ACCURACY
        )
```

In `_prepare_incidents`, remove the `station_prediction_overrides=None,` parameter and replace

```python
        bucket_policy = prediction_policy.get(bucket, {}) if bucket is not None else {}
        incident.used_station_prediction_override = False
        if (
            station_prediction_overrides
            and bucket == PREDICTION_STATION_OVERRIDE_BUCKET
        ):
            station_override = station_prediction_overrides.get(
                _effective_station(incident.station).id
            )
            if station_override:
                bucket_policy = {**bucket_policy, **station_override}
                incident.used_station_prediction_override = True
```

with

```python
        bucket_policy = prediction_policy.get(bucket, {}) if bucket is not None else {}
```

In `_status_page_context`, delete the `station_prediction_overrides = _prediction_station_overrides(...)` statement and the `station_prediction_overrides=...` argument. `raw_issues` is then only used once, so pass `issue_queryset` straight to `_prepare_incidents` and delete the `raw_issues = list(issue_queryset)` line.

In `api_incidents`, replace the `issues = _prepare_incidents(...)` call with:

```python
    issues = _prepare_incidents(
        base_qs.filter(resolved=False, information=False),
        prediction_policy=prediction_policy,
        now=now,
    )
```

and delete the `raw_issues = list(...)` line above it.

In `_prediction_stats`:

- delete the `station_prediction_overrides = _prediction_station_overrides(...)` statement and the `station_prediction_overrides=...` argument;
- replace the `"prediction_display_min_accuracy_pct"` entry's value with `round(PREDICTION_MIN_ACCURACY * 100)`;
- delete the four `"prediction_station_override_..."` entries from the returned dict.

Run: `grep -n "station_override\|station_prediction_overrides\|lower_bound\|_wilson\|sqrt" incidents/views.py templates/*.html`
Expected: only the line in `templates/stats.html` that Step 5 removes.

- [ ] **Step 5: Update the stats page wording**

In `templates/stats.html` replace the paragraph `<p>Fixed by the predicted time: ...</p>` with:

```html
                <p>Right first time: {{ prediction_correct_count }} / {{ prediction_count }} ({{ prediction_accuracy_pct }}%)</p>
```

Replace the three explanatory paragraphs that follow the `{% if current_prediction_overdue_count %} ... {% endif %}` block (they begin `Each prediction is a "fixed by" bound`, `User-facing predictions are only shown when` and `A narrow exception applies`) with:

```html
                <p><i>Each prediction is a "fixed by" time, made once when the fault is first reported and never changed. It is right first time if the fault was fixed at or before that time, allowing {{ prediction_boundary_grace_minutes }} minutes of grace. Being fixed earlier still counts. We aim to be right about 96 times in 100, so the times we give are deliberately cautious.</i></p>
                <p><i>Predictions are only shown while they have earned it: at least {{ prediction_display_min_samples }} recent predictions of the same confidence must have resolved, and at least {{ prediction_display_min_accuracy_pct }}% of them must have been right first time. We use the last {{ prediction_display_primary_window_days }} days when possible, and fall back to the last {{ prediction_display_fallback_window_days }} days if there are too few. When a prediction is missed we say so on the fault and do not offer a second one.</i></p>
```

- [ ] **Step 6: Run the suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (164 passed). No other test builds a track record or quotes the old headline; a `grep -n "range(15)" tests/test_incidents.py` should now show only the three rate-limit tests.

- [ ] **Step 7: Commit**

```bash
git add -A incidents templates tests
git commit -m "Show predictions only while 90% are right first time

Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>"
```

---

### Task 8: Backfill never stores a prediction that has already passed

**Files:**
- Modify: `incidents/management/commands/predict_durations.py`
- Test: `tests/test_incidents.py` (`ViewAndCommandTests`)

**Interfaces:**
- Consumes: `predict_duration(incident) -> (timedelta | None, float | None)`.
- Produces: `manage.py predict_durations` leaves `estimated_duration` unset when `start_time + duration` is not in the future, and reports how many it skipped for that reason.

- [ ] **Step 1: Write the failing test**

Add to `ViewAndCommandTests`:

```python
    def test_predict_durations_never_stores_a_time_that_has_already_passed(self):
        parent = self.create_parent_station("Bank")
        old = self.create_incident(
            parent, text="Old outage", start_time=timezone.now() - timedelta(days=3)
        )
        fresh = self.create_incident(parent, text="Fresh outage")
        stdout = StringIO()

        with patch(
            "incidents.management.commands.predict_durations._load_model",
            return_value=object(),
        ), patch(
            "incidents.management.commands.predict_durations.predict_duration",
            return_value=(timedelta(hours=5), 0.95),
        ):
            call_command("predict_durations", stdout=stdout)

        old.refresh_from_db()
        fresh.refresh_from_db()
        self.assertIsNone(old.estimated_duration)
        self.assertEqual(fresh.estimated_duration, timedelta(hours=5))
        self.assertIn("1 skipped as already past", stdout.getvalue())
```

In `test_predict_durations_command_can_resume_in_small_batches`, the mocked prediction is 30 minutes for a fault that started an hour ago, which is now skipped. Change both `timedelta(minutes=30)` in that test to `timedelta(hours=5)`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_incidents.py -q -k "predict_durations"`
Expected: `test_predict_durations_never_stores_a_time_that_has_already_passed` fails because `old.estimated_duration` is 5 hours.

- [ ] **Step 3: Implement**

In `incidents/management/commands/predict_durations.py`, in `handle`, add `skipped_past = 0` beside `skipped_planned = 0`, and replace

```python
            if duration is not None:
                incident.estimated_duration = duration
                incident.prediction_confidence = confidence
                incident.save(update_fields=["estimated_duration", "prediction_confidence"])
                updated += 1
            else:
                skipped_planned += 1
```

with

```python
            if duration is None:
                skipped_planned += 1
            elif incident.start_time + duration <= timezone.now():
                # A prediction is a promise shown to users. One whose time
                # has already passed was never shown, so storing it would
                # have the site admit to (or take credit for) a prediction
                # it never made.
                skipped_past += 1
            else:
                incident.estimated_duration = duration
                incident.prediction_confidence = confidence
                incident.save(update_fields=["estimated_duration", "prediction_confidence"])
                updated += 1
```

Replace the final `self.stdout.write(self.style.SUCCESS(...))` with:

```python
        self.stdout.write(self.style.SUCCESS(
            f"Updated {updated} / {total} incident(s) "
            f"({skipped_planned} skipped with no prediction, "
            f"{skipped_past} skipped as already past)"
        ))
```

and change the progress line's `skipped={skipped_planned}` to `skipped={skipped_planned + skipped_past}`.

- [ ] **Step 4: Run the suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add incidents/management/commands/predict_durations.py tests/test_incidents.py
git commit -m "Backfill never stores a prediction whose time has passed

Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>"
```

---

### Task 9: Train the version 4 model

**Files:**
- Modify: `ml/train.py`
- Modify: `ml/test_train.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes from Task 1: everything in `incidents.prediction_shared`. The feature names Task 4 serves: `HISTORY_FEATURE_NAMES`, `minute_of_day`, `network_started_24h`.
- Produces:
  - `train.prepare_features(df) -> (df, feature_cols, vectorizer)`; `df` keeps `block_offset_raw`, `start_block_idx`, `text`.
  - `train.train_model(df, feature_cols) -> (model, historical_baselines, tail_distribution, cv_metrics)` where `cv_metrics` has `mean_bound_given`, `mean_resolved_by_bound`, `mean_resolved_by_bound_grace`, `by_category` (`{category: {"count": int, "resolved_by_bound_grace": float}}`) and `n_splits`.
  - `train.quality_gate_failure(cv_metrics, min_boundary_accuracy, min_category_accuracy=0.0, min_category_samples=100) -> str | None`.
  - `train.upload_model(model, vectorizer, historical_baselines, tail_distribution, upload_url, key)` writing a bundle with `tail_distribution` and `metadata.feature_version == 4`.
  - CLI flags `--min-category-accuracy` and `--min-category-samples`.

- [ ] **Step 1: Create the training environment**

`ml/test_train.py` needs pandas, which the app's `.venv` deliberately lacks.

```bash
.venv/bin/python -m venv .venv-ml
.venv-ml/bin/python -m pip install --quiet -r ml/requirements.txt pytest
printf '.venv-ml\n' >> .gitignore
.venv-ml/bin/python -m pytest ml/test_train.py -q -p no:cacheprovider
```

Expected: `3 passed`.

- [ ] **Step 2: Write the failing tests**

In `ml/test_train.py`, add `import pytest` to the imports and, below `import train`, add:

```python
from incidents import prediction_shared as shared
```

(`train` puts the repository root on `sys.path`, so this import must come after it.)

In the two existing tests that call `train.train_model`, change `model, _, _ = train.train_model(...)` to `model, _, _, _ = train.train_model(...)`.

Add:

```python
def _row(station_id, began, hours, text="The lift is out of order.", information=False):
    lasted = timedelta(hours=hours)
    return {
        "station_id": station_id,
        "information": information,
        "text": text,
        "start_time": began.isoformat(),
        "end_time": (began + lasted).isoformat(),
        "duration_minutes": lasted.total_seconds() / 60,
        "is_planned_work": False,
        "has_faulty_lift": True,
        "has_planned_maintenance": False,
        "has_staff_issue": False,
        "tube": True,
        "dlr": False,
        "national_rail": False,
        "crossrail": False,
        "overground": False,
        "access_via_lift": True,
        "num_reports": 1,
        "days_since_last_incident": -1.0,
        "concurrent_incidents": 0,
    }


def test_only_work_with_a_stated_end_is_left_out_of_training():
    raw = _raw_incidents(30)
    maintenance = "Step free access is not available due to planned maintenance."
    raw.loc[0, "text"] = maintenance
    raw.loc[0, "is_planned_work"] = True
    raw.loc[1, "text"] = "No step-free access until Sunday 22 November."
    raw.loc[2, "information"] = True

    df, _, _ = train.prepare_features(raw)

    assert maintenance in set(df["text"])
    assert not any("until" in text for text in df["text"])
    assert len(df) == len(raw) - 2


def test_history_features_are_the_shared_ones_and_only_use_the_past():
    # 3 March 2025 is GMT, so local time is UTC. 09:00 is the morning block.
    start = datetime(2025, 3, 3, 9, 0, tzinfo=timezone.utc)
    raw = pd.DataFrame([
        _row(1, start, 60),                          # ends 5 March 21:00
        _row(1, start + timedelta(hours=10), 2),     # ends 3 March 21:00
        _row(1, start + timedelta(hours=80), 1),     # starts 6 March 17:00
        _row(2, start + timedelta(hours=81), 1),
    ])

    df, feature_cols, _ = train.prepare_features(raw)
    df = df.sort_values("start_time").reset_index(drop=True)
    first, second, third, other = (df.iloc[i] for i in range(4))

    for name in shared.HISTORY_FEATURE_NAMES + ("minute_of_day", "network_started_24h"):
        assert name in feature_cols
    assert "station_mean_duration" not in feature_cols
    assert first["st_count"] == 0
    # The second fault began while the first was still open.
    assert second["st_count"] == 0
    assert second["st_open_now"] == 1
    # By the third, both have ended; the 60-hour one ended last.
    assert third["st_count"] == 2
    assert third["sf_count"] == 2
    assert third["st_last_off"] == 11.0   # morning 3 March to overnight 5 March
    assert third["st_days_since_last_end"] == pytest.approx(20 / 24)
    assert third["minute_of_day"] == 17 * 60
    assert third["network_started_24h"] == 0
    assert other["st_count"] == 0
    assert other["network_started_24h"] == 1


def test_training_returns_a_tail_table_and_scores_the_real_decode():
    df, feature_cols, _ = train.prepare_features(_raw_incidents(300))

    _, _, tail, cv = train.train_model(df, feature_cols)

    assert set(tail) == {"by_category", "all"}
    assert tail["all"]["total"] == int(
        (df["block_offset_raw"] >= train.MAX_OFFSET_CLASS).sum()
    )
    assert 0 < cv["mean_bound_given"] <= 1
    assert 0 <= cv["mean_resolved_by_bound_grace"] <= 1
    assert sum(entry["count"] for entry in cv["by_category"].values()) > 0


def test_decode_bound_matches_the_shared_decode():
    tail = shared.build_tail_distribution({"faulty_lift": [9] * 40})
    # All mass in the open-ended class; start block index 1 is a daytime block.
    assert train.decode_bound([0, 8], [0.0, 1.0], tail, "faulty_lift", 1) == 10
    assert train.decode_bound([0, 8], [0.0, 1.0], None, "faulty_lift", 1) is None
    assert train.decode_bound([0, 1], [0.96, 0.04], None, "faulty_lift", 1) == 0


def test_quality_gate_checks_overall_and_each_large_category():
    metrics = {
        "mean_resolved_by_bound_grace": 0.96,
        "by_category": {
            "faulty_lift": {"count": 500, "resolved_by_bound_grace": 0.92},
            "other": {"count": 20, "resolved_by_bound_grace": 0.50},
        },
    }
    assert train.quality_gate_failure(metrics, 0.93, 0.90, 100) is None

    low_overall = {**metrics, "mean_resolved_by_bound_grace": 0.92}
    assert "overall" in train.quality_gate_failure(low_overall, 0.93, 0.90, 100)

    weak_lifts = {
        "mean_resolved_by_bound_grace": 0.96,
        "by_category": {"faulty_lift": {"count": 500, "resolved_by_bound_grace": 0.89}},
    }
    assert "faulty_lift" in train.quality_gate_failure(weak_lifts, 0.93, 0.90, 100)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv-ml/bin/python -m pytest ml/test_train.py -q -p no:cacheprovider`
Expected: 7 failed, 1 passed. The two existing training tests fail with `ValueError: not enough values to unpack (expected 4, got 3)`, and the new ones with errors such as `AttributeError: module 'train' has no attribute 'decode_bound'`.

- [ ] **Step 4: Import the shared module in `ml/train.py`**

Replace the import block at the top (from `import argparse` to the `sklearn` imports) with:

```python
import argparse
import bisect
import os
import re
import sys
from datetime import date as date_cls
from datetime import datetime, time, timedelta

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import TimeSeriesSplit

# The rules shared with the app live in incidents/prediction_shared.py, which
# is standard library only. Make the repository root importable so this
# script runs as ``python ml/train.py`` without installing anything else.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from incidents.prediction_shared import (  # noqa: E402
    HISTORY_FEATURE_NAMES,
    MAX_OFFSET_CLASS,
    PREDICTION_COVERAGE_TARGET,
    bound_offset,
    build_tail_distribution,
    extended_cdf,
    fault_category,
    fault_key,
    has_stated_end,
    history_features,
    round_up_to_service_day,
)
```

(`heapq` and `accuracy_score` are no longer used.)

Delete the local `MAX_OFFSET_CLASS = 8` assignment with its comment block, and the local `PREDICTION_COVERAGE_TARGET = 0.75` assignment with its comment. Keep `PREDICTION_BOUNDARY_GRACE`.

- [ ] **Step 5: Replace the history features**

Delete `add_causal_station_history` and add in its place:

```python
def add_history_features(df):
    """History from previous faults, computed by the code the app runs.

    ``df`` must still hold every exported fault, including work with a stated
    end, because the app's history query sees those too. Needs ``text``
    already cleaned and ``block_offset_raw`` already computed.
    """
    df = df.sort_values(["start_time", "end_time", "station_id"]).reset_index(drop=True)
    starts = df["start_time"].tolist()
    ends = df["end_time"].tolist()
    offsets = [int(offset) for offset in df["block_offset_raw"]]
    categories = [fault_category(text) for text in df["text"]]
    keys = [fault_key(text) for text in df["text"]]

    positions_by_station = {}
    for position, station_id in enumerate(df["station_id"].tolist()):
        positions_by_station.setdefault(station_id, []).append(position)

    rows = [None] * len(df)
    for positions in positions_by_station.values():
        faults = [
            (starts[p], ends[p], offsets[p], categories[p], keys[p])
            for p in positions
        ]
        for i, p in enumerate(positions):
            # faults[:i] began no later than this one; history_features
            # itself drops ties and anything that had not yet ended.
            rows[p] = history_features(faults[:i], starts[p], categories[p], keys[p])

    history = pd.DataFrame(rows, index=df.index, columns=list(HISTORY_FEATURE_NAMES))
    df = pd.concat([df, history], axis=1)

    df["minute_of_day"] = (
        df["start_time"].dt.hour * 60 + df["start_time"].dt.minute
    ).astype(int)
    ordered_starts = sorted(starts)
    df["network_started_24h"] = [
        bisect.bisect_left(ordered_starts, start)
        - bisect.bisect_left(ordered_starts, start - timedelta(days=1))
        for start in starts
    ]
    return df
```

Replace the whole of `prepare_features` with:

```python
def prepare_features(df):
    # Parse datetime columns and convert to local time. Blocks are defined
    # in London time, not UTC — otherwise the boundaries drift by an hour
    # during BST.
    df["start_time"] = pd.to_datetime(
        df["start_time"], utc=True, format="ISO8601"
    ).dt.tz_convert(LOCAL_TZ)
    df["end_time"] = pd.to_datetime(
        df["end_time"], utc=True, format="ISO8601"
    ).dt.tz_convert(LOCAL_TZ)

    # Keep text noise down so the TF-IDF branch does not spend capacity on
    # TfL boilerplate like the phone number.
    df["text"] = df["text"].fillna("").map(normalise_incident_text)

    # Compute block indices and the raw offset
    df["start_block_idx"] = df["start_time"].apply(block_index)
    df["end_block_idx"] = df["end_time"].apply(block_index)
    df["block_offset_raw"] = df["end_block_idx"] - df["start_block_idx"]

    # Drop obviously bad rows
    before = len(df)
    df = df[df["block_offset_raw"] >= 0].copy()
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with negative block offset (bad data).")

    # History comes first, while every exported fault is still present.
    df = add_history_features(df)

    # Leave out only work that states its own end: there is nothing to
    # predict. Undated planned maintenance behaves like any other fault and
    # is trained on.
    stated_end = np.array(
        [
            has_stated_end(text, information)
            for text, information in zip(df["text"], df["information"])
        ],
        dtype=bool,
    )
    df = df[~stated_end].copy()
    print(
        f"Excluded {int(stated_end.sum())} faults with a stated end, "
        f"{len(df)} remaining."
    )

    # Cap into a finite class set
    df[TARGET_COLUMN] = df["block_offset_raw"].clip(upper=MAX_OFFSET_CLASS).astype(int)

    # Derived features are all based on local time so they line up with the
    # block target we are predicting.
    df["hour_of_day"] = df["start_time"].dt.hour.astype(int)
    df["day_of_week"] = df["start_time"].dt.weekday.astype(int)
    df["month"] = df["start_time"].dt.month.astype(int)
    df["start_block"] = (df["start_block_idx"] % BLOCKS_PER_DAY).astype(int)
    df["is_weekend"] = (df["start_time"].dt.weekday >= 5).astype(int)

    bool_columns = [
        "information",
        "has_faulty_lift",
        "has_planned_maintenance",
        "has_staff_issue",
        "tube",
        "dlr",
        "national_rail",
        "crossrail",
        "overground",
        "access_via_lift",
    ]
    for col in bool_columns:
        df[col] = df[col].astype(int)

    # TF-IDF on incident text
    vectorizer = TfidfVectorizer(max_features=50, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["text"])
    tfidf_cols = [f"tfidf_{name}" for name in vectorizer.get_feature_names_out()]
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(), columns=tfidf_cols, index=df.index
    )
    df = pd.concat([df, tfidf_df], axis=1)

    feature_cols = (
        FEATURE_COLUMNS
        + list(HISTORY_FEATURE_NAMES)
        + ["minute_of_day", "network_started_24h"]
        + tfidf_cols
    )

    return df, feature_cols, vectorizer
```

- [ ] **Step 6: Decode in cross-validation exactly as the app does**

Add directly above `train_model`:

```python
def build_tail(df):
    """Tail table from the faults in ``df``, keyed the way the app keys it."""
    offsets_by_category = {}
    for text, offset in zip(df["text"], df["block_offset_raw"]):
        offsets_by_category.setdefault(fault_category(text), []).append(int(offset))
    return build_tail_distribution(offsets_by_category)


def decode_bound(classes, probs, tail, category, start_block_idx):
    """The block offset the app would promise, or None for no prediction.

    Mirrors incidents.ml._decode_offset_bound.
    """
    class_probs = [0.0] * (MAX_OFFSET_CLASS + 1)
    for cls, probability in zip(classes, probs):
        class_probs[int(cls)] = float(probability)
    offset = bound_offset(extended_cdf(class_probs, tail, category))
    if offset is None:
        return None
    return round_up_to_service_day(int(start_block_idx), offset)


def quality_gate_failure(
    cv_metrics,
    min_boundary_accuracy,
    min_category_accuracy=0.0,
    min_category_samples=100,
):
    """Why this model must not ship, or None when it may."""
    overall = cv_metrics["mean_resolved_by_bound_grace"]
    if overall < min_boundary_accuracy:
        return (
            f"overall right-first-time {overall * 100:.1f}% is below the "
            f"required {min_boundary_accuracy * 100:.1f}%"
        )
    for category, entry in sorted(cv_metrics.get("by_category", {}).items()):
        if entry["count"] < min_category_samples:
            continue
        rate = entry["resolved_by_bound_grace"]
        if rate < min_category_accuracy:
            return (
                f"{category} right-first-time {rate * 100:.1f}% "
                f"(n={entry['count']}) is below the required "
                f"{min_category_accuracy * 100:.1f}%"
            )
    return None
```

In `train_model`, replace everything from the comment `# Time-series cross-validation on block accuracy.` to the end of the function with:

```python
    # Time-series cross-validation, decoded and scored the way the app
    # decodes and scores: a bound is right first time when the fault was
    # fixed at or before it.
    tscv = TimeSeriesSplit(n_splits=3)
    fold_scores = []
    category_tally = {}
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        sw_train = sample_weights[train_idx]
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        baselines = build_historical_baselines(train_df)
        tail = build_tail(train_df)

        cv_model = _fresh_model()
        cv_model.fit(X_train, y_train, sample_weight=sw_train)
        model_probs = cv_model.predict_proba(X_test)

        given = held = held_with_grace = 0
        for row, row_model_probs in zip(
            test_df.itertuples(index=False), model_probs
        ):
            baseline_full_probs, baseline_evidence = historical_baseline_for_row(
                row, baselines
            )
            baseline_probs = align_baseline_probs(
                baseline_full_probs,
                cv_model.classes_,
            )
            blended_probs = blend_prediction_probs(
                row_model_probs,
                baseline_probs,
                baseline_evidence,
                baselines["config"],
            )
            category = fault_category(row.text)
            offset = decode_bound(
                cv_model.classes_, blended_probs, tail, category, row.start_block_idx
            )
            if offset is None:
                continue
            close = prediction_is_close_enough(
                int(row.start_block_idx) + offset, row.end_time
            )
            given += 1
            held += int(row.block_offset_raw <= offset)
            held_with_grace += int(close)
            tally = category_tally.setdefault(category, [0, 0])
            tally[0] += 1
            tally[1] += int(close)

        fold_scores.append(
            (
                given / len(test_df),
                held / given if given else 0.0,
                held_with_grace / given if given else 0.0,
            )
        )

    print(
        f"\nCross-validation results ({tscv.n_splits} splits, decoded as the app "
        f"decodes, aim {PREDICTION_COVERAGE_TARGET:.2f}):"
    )
    grace_m = int(PREDICTION_BOUNDARY_GRACE.total_seconds() / 60)
    for i, (given_rate, held_rate, grace_rate) in enumerate(fold_scores):
        print(
            f"  fold {i + 1}: given a date {given_rate * 100:.1f}%, "
            f"right first time {held_rate * 100:.1f}%, "
            f"with +{grace_m}m grace {grace_rate * 100:.1f}%"
        )
    mean_given = sum(s[0] for s in fold_scores) / len(fold_scores)
    mean_exact = sum(s[1] for s in fold_scores) / len(fold_scores)
    mean_boundary = sum(s[2] for s in fold_scores) / len(fold_scores)
    print(
        f"  mean:   given a date {mean_given * 100:.1f}%, "
        f"right first time {mean_exact * 100:.1f}%, "
        f"with +{grace_m}m grace {mean_boundary * 100:.1f}%"
    )
    by_category = {
        category: {"count": count, "resolved_by_bound_grace": hits / count}
        for category, (count, hits) in sorted(category_tally.items())
    }
    for category, entry in by_category.items():
        print(
            f"  {category}: {entry['resolved_by_bound_grace'] * 100:.1f}% "
            f"(n={entry['count']})"
        )

    # Train final model on all data
    model = _fresh_model()
    model.fit(X, y, sample_weight=sample_weights)

    historical_baselines = build_historical_baselines(df)
    tail_distribution = build_tail(df)
    cv_metrics = {
        "mean_bound_given": mean_given,
        "mean_resolved_by_bound": mean_exact,
        "mean_resolved_by_bound_grace": mean_boundary,
        "by_category": by_category,
        "n_splits": tscv.n_splits,
    }
    return model, historical_baselines, tail_distribution, cv_metrics
```

- [ ] **Step 7: Ship the tail table and apply the gates**

Replace `upload_model`'s signature and `joblib.dump` call with:

```python
def upload_model(
    model, vectorizer, historical_baselines, tail_distribution, upload_url, key
):
    model_path = "ml_model.joblib"
    joblib.dump({
        "model": model,
        "vectorizer": vectorizer,
        "historical_baselines": historical_baselines,
        "tail_distribution": tail_distribution,
        "metadata": {
            "feature_version": 4,
        },
    }, model_path)
```

In `main`, add after the `--min-boundary-accuracy` argument:

```python
    parser.add_argument(
        "--min-category-accuracy",
        type=float,
        default=0.0,
        help=(
            "Refuse to upload if any fault category with enough "
            "cross-validation samples is right first time less often than "
            "this fraction (0-1). Default 0 (no gate)."
        ),
    )
    parser.add_argument(
        "--min-category-samples",
        type=int,
        default=100,
        help=(
            "Only apply --min-category-accuracy to categories with at least "
            "this many cross-validation samples. Default 100."
        ),
    )
```

Replace from `model, historical_baselines, cv_metrics = train_model(df, feature_cols)` through the end of the first `if boundary_accuracy < args.min_boundary_accuracy:` block with:

```python
    model, historical_baselines, tail_distribution, cv_metrics = train_model(
        df, feature_cols
    )

    boundary_accuracy = cv_metrics["mean_resolved_by_bound_grace"]
    failure = quality_gate_failure(
        cv_metrics,
        args.min_boundary_accuracy,
        args.min_category_accuracy,
        args.min_category_samples,
    )
    if failure:
        print(
            f"\nABORTING UPLOAD: {failure}. The current model is left in place."
        )
        sys.exit(2)
```

Replace the final `upload_model(...)` call with:

```python
    upload_model(
        model,
        vectorizer,
        historical_baselines,
        tail_distribution,
        args.upload_url,
        args.key,
    )
```

- [ ] **Step 8: Run both suites**

Run: `.venv-ml/bin/python -m pytest ml/test_train.py -q -p no:cacheprovider`
Expected: `8 passed`.

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (the app suite does not import `ml/train.py`).

- [ ] **Step 9: Commit**

```bash
git add ml/train.py ml/test_train.py .gitignore
git commit -m "Train the version 4 model: history features, tail table, stricter gates

Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>"
```

---

### Task 10: Retrain workflow defaults and final check

**Files:**
- Modify: `.github/workflows/retrain.yml`

**Interfaces:**
- Consumes: `ml/train.py` flags `--min-boundary-accuracy` and `--min-category-accuracy` from Task 9.
- Produces: scheduled and manual retrains refuse to upload below 0.93 overall or 0.90 for a large category.

- [ ] **Step 1: Update the workflow**

In `.github/workflows/retrain.yml`:

Replace the header comment's bullet list with:

```yaml
#   * overall right-first-time floor      (--min-boundary-accuracy)
#   * per-category right-first-time floor (--min-category-accuracy)
#   * no regression vs the incumbent's recent live accuracy (--max-regression)
# If any gate fails, train.py exits non-zero and uploads nothing, so a
```

(the line that follows, `# regressed model can never auto-ship — the run just goes red.`, stays).

Under `workflow_dispatch.inputs`, change `min_boundary_accuracy`'s `default` to `"0.93"` and its description to `"Floor for overall right-first-time accuracy (0-1)"`, and add after it:

```yaml
      min_category_accuracy:
        description: "Floor for each large fault category (0-1)"
        required: false
        default: "0.90"
```

In the final step's `run:` block, replace the `--min-boundary-accuracy` line with:

```yaml
            --min-boundary-accuracy "${{ github.event.inputs.min_boundary_accuracy || '0.93' }}" \
            --min-category-accuracy "${{ github.event.inputs.min_category_accuracy || '0.90' }}" \
```

- [ ] **Step 2: Check the whole change**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

Run: `.venv-ml/bin/python -m pytest ml/test_train.py -q -p no:cacheprovider`
Expected: `8 passed`.

Run: `grep -rn "some_time\|long_running\|predict_remaining_duration\|station_override\|_wilson_lower_bound" incidents ml templates .github`
Expected: no output.

Run: `grep -rn "PREDICTION_COVERAGE_TARGET = \|MAX_OFFSET_CLASS = " incidents ml`
Expected: exactly two lines, both in `incidents/prediction_shared.py`.

The app suite's `MigrationMemoryTests.test_models_have_no_unmigrated_changes` passing confirms this change needs no migration.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/retrain.yml
git commit -m "Retrain workflow: gate on 93% overall and 90% per category

Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>"
```

---

## After merge (manual; each step needs Sarah's go-ahead)

Nothing here is automated by the tasks above. Merging to `master` deploys.

1. **Merge and deploy.** The production model is version 3, so it keeps working. At the 95% aim and without a tail table it gives a date to fewer faults until step 3.
2. **Dry-run the retrain.** GitHub → Actions → "Retrain model" → Run workflow with `dry_run` ticked. Read the "Cross-validation results" block in the log. Expect about 96% overall with every large category at 90% or more. If a category is below 90%, raise `PREDICTION_COVERAGE_TARGET` in `incidents/prediction_shared.py` by 0.01, redeploy, and repeat.
3. **Retrain for real.** Run the workflow again without `dry_run`. The log ends with `Model uploaded successfully.`
4. **Backfill open faults.** On the Fly machine: `python updown/manage.py predict_durations`. It fills only faults with no prediction and skips any whose time has already passed.
5. **Expect a gap of two to three days** with few predictions on the site, until 30 new predictions have resolved and the 90-100% confidence bucket has a record of its own.
6. **Watch `/stats/`.** "Right first time" should settle near 96%. If it falls below 90% the site hides predictions by itself.

## Amendments during execution (2026-11-15)

The tasks above are as written before execution. What changed while doing them:

1. **Planned maintenance is not predicted.** Running the rewritten
   `ml/train.py` on the public export put undated planned maintenance at 88.5%
   right first time, below the 90% bar, and the per-category gate refused the
   model. `has_stated_end` became `is_planned_work` (information, "until ",
   or "planned"), used for every model version by both the app and the
   training script. The version-dependent rule in Task 4, its test, and Task
   9's exclusion test were rewritten to match. With the exclusion restored the
   gates pass: 96.3% overall, lifts 95.0%, staff 98.4%, other 96.3%.
2. **Task 6** also had to update `test_prepare_incidents_hides_overdue_current_prediction`
   (renamed `..._admits_overdue_current_prediction`): it asserted the old
   silent hiding of a missed prediction.
3. **Task 9's** tail-table test uses the 300-station fixture. Forty stations
   left a class with a single example, which scikit-learn's early-stopping
   split rejects.
4. **Task 3:** `PREDICTION_COVERAGE_TARGET` is not re-exported from
   `incidents.ml`; nothing imports it from there.
