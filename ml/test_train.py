"""Tests for ml/train.py.

These need the training dependencies (ml/requirements.txt), not the app's,
so they live here rather than in tests/. Run from the repo root with:

    python -m pytest ml/test_train.py
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import train


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

    model, _, _ = train.train_model(df, feature_cols)

    assert model.predict_proba(df[feature_cols].head(1)).shape[1] == len(model.classes_)


def test_raw_station_ids_at_prediction_time_match_the_training_groups():
    # The app passes raw station IDs to the model. A station that was
    # grouped with the quiet ones in training, and one never seen at all,
    # must both be scored exactly like the quiet group.
    df, feature_cols, _ = train.prepare_features(_raw_incidents(300))
    model, _, _ = train.train_model(df, feature_cols)

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
