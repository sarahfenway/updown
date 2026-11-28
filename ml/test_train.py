"""Tests for ml/train.py.

These need the training dependencies (ml/requirements.txt), not the app's,
so they live here rather than in tests/. Run from the repo root with:

    python -m pytest ml/test_train.py
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import train

from incidents import prediction_shared as shared


def _raw_incidents(n_stations, per_station=8):
    """Resolved incidents shaped like /api/training-data/ rows, with the
    first stations much busier than the rest."""
    rng = np.random.default_rng(0)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    for station_id in range(1, n_stations + 1):
        count = per_station * 4 if station_id <= 20 else per_station
        for _ in range(count):
            began = start + timedelta(minutes=int(rng.integers(0, 60 * 24 * 300)))
            lasted = timedelta(minutes=int(rng.integers(20, 60 * 30)))
            rows.append({
                "station_id": station_id,
                "information": False,
                "text": "The lift is out of order. Engineers are working to fix it.",
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
                "num_reports": int(rng.integers(1, 5)),
                "days_since_last_incident": float(rng.integers(0, 60)),
                "concurrent_incidents": int(rng.integers(0, 10)),
            })
    return pd.DataFrame(rows)


def test_station_categories_keep_the_busiest_and_group_the_rest():
    station_ids = pd.Series([1] * 5 + [2] * 4 + [3] * 3 + [4] * 2 + [5])

    capped = train.cap_station_categories(station_ids, max_categories=3)

    assert capped.tolist()[:12] == [1.0] * 5 + [2.0] * 4 + [3.0] * 3
    assert capped.iloc[12:].isna().all()


def test_trains_with_more_stations_than_the_model_has_categories_for():
    # Production outage, 6 Nov 2026: 262 stations in the training data, but
    # HistGradientBoostingClassifier allows at most max_bins (255)
    # categories, so the final fit raised and no model was uploaded.
    n_stations = 300
    df, feature_cols, _ = train.prepare_features(_raw_incidents(n_stations))
    assert df["station_id"].nunique() == n_stations

    model, _, _, _ = train.train_model(df, feature_cols)

    assert model.predict_proba(df[feature_cols].head(1)).shape[1] == len(model.classes_)


def test_raw_station_ids_at_prediction_time_match_the_training_groups():
    # The app passes raw station IDs to the model. A station that was
    # grouped with the quiet ones in training, and one never seen at all,
    # must both be scored exactly like the quiet group.
    df, feature_cols, _ = train.prepare_features(_raw_incidents(300))
    model, _, _, _ = train.train_model(df, feature_cols)

    row = df[feature_cols].head(1).copy()
    capped = train.cap_station_categories(df["station_id"])
    quiet_station = int(df.loc[capped.isna(), "station_id"].iloc[0])
    busiest_station = int(df.loc[capped.notna(), "station_id"].iloc[0])

    def probs(station_id):
        candidate = row.copy()
        candidate["station_id"] = station_id
        return model.predict_proba(candidate)

    np.testing.assert_array_equal(probs(quiet_station), probs(np.nan))
    np.testing.assert_array_equal(probs(999_999), probs(np.nan))
    assert not np.array_equal(probs(busiest_station), probs(np.nan))


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


def test_planned_and_dated_work_is_left_out_of_training_but_kept_as_history():
    start = datetime(2025, 3, 3, 9, 0, tzinfo=timezone.utc)
    maintenance = "Step free access is not available due to planned maintenance."
    raw = pd.DataFrame([
        _row(1, start, 2, text=maintenance),
        _row(1, start + timedelta(hours=5), 2, text="No access until Sunday 9 March."),
        _row(1, start + timedelta(hours=10), 2, information=True),
        _row(1, start + timedelta(hours=20), 2),
        _row(1, start + timedelta(hours=30), 2),
    ])

    df, _, _ = train.prepare_features(raw)
    df = df.sort_values("start_time").reset_index(drop=True)

    # Only the two ordinary faults are trained on...
    assert list(df["text"]) == ["The lift is out of order."] * 2
    # ...but the three excluded ones still count as the station's history,
    # because the app's history query sees every fault.
    assert df.iloc[0]["st_count"] == 3


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
    assert 0 <= cv["long_bound_share"] <= 1
    assert cv["median_bound_offset"] >= 0


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


def test_bound_length_summary_counts_dates_two_or_more_days_out():
    summary = train.bound_length_summary(
        bound_offsets=[0, 1, 1, 10], actual_offsets=[0, 0, 1, 2]
    )
    assert summary == {
        "long_bound_share": 0.25,
        "median_bound_offset": 1.0,
        "long_actual_share": 0.0,
        "median_actual_offset": 0.5,
    }


def test_quality_gate_rejects_a_model_whose_dates_are_mostly_days_away():
    metrics = {
        "mean_resolved_by_bound_grace": 0.99,
        "long_bound_share": 0.60,
        "by_category": {},
    }
    assert train.quality_gate_failure(metrics, 0.78, 0.75, 100) is None
    failure = train.quality_gate_failure(
        metrics, 0.78, 0.75, 100, max_long_bound_share=0.35
    )
    assert "two or more days" in failure
