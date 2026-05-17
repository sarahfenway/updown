#!/usr/bin/env python3
"""
Standalone training script for the incident duration prediction model.

Usage:
    python train.py \
        --api-url https://updownlondon.com/api/training-data/ \
        --upload-url https://updownlondon.com/functions/upload_model \
        --key YOUR_FUNCTIONS_SECRET_KEY

This script:
1. Pulls training data from the Up Down London API.
2. Trains a classifier that predicts which coarse time block an incident
   will be resolved in ("block offset" from the start block).
3. Uploads the trained model back to the server.

The classifier directly optimises for the metric we care about (did the
prediction land in the right block?), instead of regressing on minutes and
hoping the boundary doesn't get crossed.
"""

import argparse
import heapq
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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit


# ---------------------------------------------------------------------------
# Block helpers — kept in sync with incidents/ml.py
# ---------------------------------------------------------------------------

BLOCK_MORNING = 0
BLOCK_DAYTIME = 1
BLOCK_EVENING = 2
BLOCK_OVERNIGHT = 3
BLOCKS_PER_DAY = 4
BLOCK_NAMES = ["morning", "daytime", "evening", "overnight"]
MAX_OFFSET_CLASS = 20
PREDICTION_BOUNDARY_GRACE = timedelta(minutes=30)


def block_index(dt):
    hour = dt.hour
    if hour < 5:
        return (dt.date() - timedelta(days=1)).toordinal() * BLOCKS_PER_DAY + BLOCK_OVERNIGHT
    if hour < 10:
        return dt.date().toordinal() * BLOCKS_PER_DAY + BLOCK_MORNING
    if hour < 15:
        return dt.date().toordinal() * BLOCKS_PER_DAY + BLOCK_DAYTIME
    if hour < 20:
        return dt.date().toordinal() * BLOCKS_PER_DAY + BLOCK_EVENING
    return dt.date().toordinal() * BLOCKS_PER_DAY + BLOCK_OVERNIGHT


def block_start_end(index):
    date_ord, block_num = divmod(index, BLOCKS_PER_DAY)
    day = date_cls.fromordinal(date_ord)
    if block_num == BLOCK_MORNING:
        return datetime.combine(day, time(5, 0)), datetime.combine(day, time(10, 0))
    if block_num == BLOCK_DAYTIME:
        return datetime.combine(day, time(10, 0)), datetime.combine(day, time(15, 0))
    if block_num == BLOCK_EVENING:
        return datetime.combine(day, time(15, 0)), datetime.combine(day, time(20, 0))
    start = datetime.combine(day, time(20, 0))
    return start, start + timedelta(hours=9)


def block_distance_from_index(dt, index):
    # Blocks are defined in local wall-clock time. The training data has
    # already been converted into Europe/London, so strip timezone info here
    # before comparing against naive block boundaries.
    if getattr(dt, "tzinfo", None) is not None:
        if hasattr(dt, "tz_localize"):
            dt = dt.tz_localize(None)
        else:
            dt = dt.replace(tzinfo=None)

    block_start_dt, block_end_dt = block_start_end(index)

    if dt < block_start_dt:
        return block_start_dt - dt
    if dt > block_end_dt:
        return dt - block_end_dt
    return timedelta(0)


def prediction_is_close_enough(predicted_block_idx, actual_dt, grace=PREDICTION_BOUNDARY_GRACE):
    return block_distance_from_index(actual_dt, predicted_block_idx) <= grace


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "station_id",
    "information",
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "start_block",
    "has_faulty_lift",
    "has_planned_maintenance",
    "has_staff_issue",
    "tube",
    "dlr",
    "national_rail",
    "crossrail",
    "overground",
    "access_via_lift",
    "num_reports",
    "days_since_last_incident",
    "has_history",
    "log_days_since_last",
    "concurrent_incidents",
    "category_x_start_block",
    "category_x_is_weekend",
]

TARGET_COLUMN = "block_offset"
NETWORK_COLUMNS = ["tube", "dlr", "national_rail", "crossrail", "overground"]

BASELINE_CATEGORY_OTHER = "other"
BASELINE_COMPONENT_WEIGHTS = {
    "global_weight": 1.0,
    "category_weight": 1.0,
    "network_category_weight": 0.5,
    "station_weight": 0.75,
    "station_category_weight": 1.25,
}
BASELINE_COMPONENT_MIN_COUNTS = {
    "category_min_count": 8,
    "network_category_min_count": 8,
    "station_min_count": 5,
    "station_category_min_count": 3,
}
BASELINE_BLEND_MAX_WEIGHT = 0.35
BASELINE_BLEND_COUNT_SCALE = 25.0


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------


def fetch_training_data(api_url, key):
    print(f"Fetching training data from {api_url}...")
    response = requests.get(api_url, params={"key": key})
    response.raise_for_status()

    data = response.json()
    incidents = data["incidents"]

    if not incidents:
        print("No training data available.")
        sys.exit(1)

    df = pd.DataFrame(incidents)
    print(f"Fetched {len(df)} resolved incidents.")
    return df


LOCAL_TZ = "Europe/London"
PHONE_NUMBER_RE = re.compile(r"0[38]43\s*222\s*1234")


def normalise_incident_text(text):
    if not text:
        return ""

    text = PHONE_NUMBER_RE.sub(" ", text)
    return " ".join(text.split())


STATION_HISTORY_DECAY = 0.95
STATION_VELOCITY_WINDOW = 5


def add_causal_station_history(df):
    """Add station history features using only incidents already resolved.

    Uses exponential decay weighting so recent incidents matter more than
    ancient ones, and also computes a 'velocity' feature — the EMA of the
    last N resolution durations.
    """
    df = df.sort_values(["start_time", "end_time", "station_id"]).reset_index(drop=True)

    station_durations = {}
    station_offsets = {}
    pending = []

    mean_durations = []
    incident_counts = []
    mean_offsets = []
    velocities = []

    for seq, row in enumerate(df.itertuples(index=False), start=1):
        while pending and pending[0][0] <= row.start_time:
            _, _, station_id, duration_minutes, block_offset = heapq.heappop(pending)
            station_durations.setdefault(station_id, []).append(duration_minutes)
            station_offsets.setdefault(station_id, []).append(block_offset)

        durations = station_durations.get(row.station_id, [])
        offsets = station_offsets.get(row.station_id, [])
        count = len(durations)
        incident_counts.append(count)

        if count:
            weights = np.array([STATION_HISTORY_DECAY ** (count - 1 - i) for i in range(count)])
            w_sum = weights.sum()
            mean_durations.append(np.dot(weights, durations) / w_sum)
            mean_offsets.append(np.dot(weights, offsets) / w_sum)
            recent = durations[-STATION_VELOCITY_WINDOW:]
            alpha = 2.0 / (len(recent) + 1)
            ema = recent[0]
            for v in recent[1:]:
                ema = alpha * v + (1 - alpha) * ema
            velocities.append(ema)
        else:
            mean_durations.append(0.0)
            mean_offsets.append(0.0)
            velocities.append(0.0)

        heapq.heappush(
            pending,
            (
                row.end_time,
                seq,
                row.station_id,
                row.duration_minutes,
                row.block_offset,
            ),
        )

    df["station_mean_duration"] = mean_durations
    df["station_incident_count"] = incident_counts
    df["station_mean_offset"] = mean_offsets
    df["station_velocity"] = velocities
    return df


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

    # Drop planned work — duration is in the text, no point predicting
    planned_count = df["is_planned_work"].astype(bool).sum()
    df = df[~df["is_planned_work"].astype(bool)].copy()
    print(f"Excluded {planned_count} planned work incidents, {len(df)} remaining.")

    # Compute block indices and offset target
    df["start_block_idx"] = df["start_time"].apply(block_index)
    df["end_block_idx"] = df["end_time"].apply(block_index)
    df["block_offset_raw"] = df["end_block_idx"] - df["start_block_idx"]

    # Drop obviously bad rows
    before = len(df)
    df = df[df["block_offset_raw"] >= 0].copy()
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with negative block offset (bad data).")

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

    # Keep text noise down so the TF-IDF branch does not spend capacity on
    # TfL boilerplate like the phone number.
    df["text"] = df["text"].fillna("").map(normalise_incident_text)

    # Use only already-ended incidents when building station history so the
    # training features match what we can know at prediction time.
    df = add_causal_station_history(df)

    # Log-transform days_since_last_incident + binary has_history flag
    df["has_history"] = (df["days_since_last_incident"] >= 0).astype(int)
    df["log_days_since_last"] = df["days_since_last_incident"].apply(
        lambda x: np.log1p(x) if x >= 0 else 0.0
    )

    # Interaction features: category × time signals
    category_codes = df.apply(
        lambda r: _incident_category_from_flags(
            bool(r["has_faulty_lift"]),
            bool(r["has_planned_maintenance"]),
            bool(r["has_staff_issue"]),
        ),
        axis=1,
    )
    category_map = {"faulty_lift": 0, "planned_maintenance": 1, "staff_issue": 2, "other": 3}
    category_numeric = category_codes.map(category_map).astype(int)
    df["category_x_start_block"] = category_numeric * BLOCKS_PER_DAY + df["start_block"]
    df["category_x_is_weekend"] = category_numeric * 2 + df["is_weekend"]

    # TF-IDF on incident text — bigrams + more features for richer signal
    vectorizer = TfidfVectorizer(
        max_features=100, stop_words="english", ngram_range=(1, 2)
    )
    tfidf_matrix = vectorizer.fit_transform(df["text"].fillna(""))
    tfidf_cols = [f"tfidf_{name}" for name in vectorizer.get_feature_names_out()]
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(), columns=tfidf_cols, index=df.index
    )
    df = pd.concat([df, tfidf_df], axis=1)

    feature_cols = (
        FEATURE_COLUMNS
        + [
            "station_mean_duration",
            "station_incident_count",
            "station_mean_offset",
            "station_velocity",
        ]
        + tfidf_cols
    )

    return df, feature_cols, vectorizer


def _incident_category_from_flags(has_faulty_lift, has_planned_maintenance, has_staff_issue):
    if has_faulty_lift:
        return "faulty_lift"
    if has_planned_maintenance:
        return "planned_maintenance"
    if has_staff_issue:
        return "staff_issue"
    return BASELINE_CATEGORY_OTHER


def _row_networks(row):
    return [network for network in NETWORK_COLUMNS if bool(getattr(row, network))]


def _increment_counts(store, key, target):
    counts = store.setdefault(key, np.zeros(MAX_OFFSET_CLASS + 1, dtype=float))
    counts[target] += 1


def build_historical_baselines(df):
    global_counts = np.zeros(MAX_OFFSET_CLASS + 1, dtype=float)
    category_counts = {}
    network_category_counts = {}
    station_counts = {}
    station_category_counts = {}

    for row in df.itertuples(index=False):
        target = int(getattr(row, TARGET_COLUMN))
        station_id = int(row.station_id)
        category = _incident_category_from_flags(
            bool(row.has_faulty_lift),
            bool(row.has_planned_maintenance),
            bool(row.has_staff_issue),
        )

        global_counts[target] += 1
        _increment_counts(category_counts, category, target)
        _increment_counts(station_counts, station_id, target)
        _increment_counts(station_category_counts, (station_id, category), target)

        for network in _row_networks(row):
            _increment_counts(network_category_counts, (network, category), target)

    def _serialise(store):
        return {
            key: {
                "count": int(counts.sum()),
                "counts": counts.tolist(),
            }
            for key, counts in store.items()
        }

    return {
        "num_classes": MAX_OFFSET_CLASS + 1,
        "config": {
            **BASELINE_COMPONENT_WEIGHTS,
            **BASELINE_COMPONENT_MIN_COUNTS,
            "blend_max_weight": BASELINE_BLEND_MAX_WEIGHT,
            "blend_count_scale": BASELINE_BLEND_COUNT_SCALE,
        },
        "global": {
            "count": int(global_counts.sum()),
            "counts": global_counts.tolist(),
        },
        "category": _serialise(category_counts),
        "network_category": _serialise(network_category_counts),
        "station": _serialise(station_counts),
        "station_category": _serialise(station_category_counts),
    }


def _apply_baseline_component(combined_counts, entry, weight, min_count):
    if not entry or entry["count"] < min_count:
        return 0.0

    combined_counts += np.asarray(entry["counts"], dtype=float) * weight
    return entry["count"] * weight


def historical_baseline_for_row(row, baselines):
    if not baselines or not baselines.get("global"):
        return None, 0.0

    config = baselines["config"]
    combined_counts = (
        np.asarray(baselines["global"]["counts"], dtype=float)
        * config["global_weight"]
    )
    evidence = 0.0
    category = _incident_category_from_flags(
        bool(getattr(row, "has_faulty_lift")),
        bool(getattr(row, "has_planned_maintenance")),
        bool(getattr(row, "has_staff_issue")),
    )
    station_id = int(getattr(row, "station_id"))

    evidence += _apply_baseline_component(
        combined_counts,
        baselines["category"].get(category),
        config["category_weight"],
        config["category_min_count"],
    )
    evidence += _apply_baseline_component(
        combined_counts,
        baselines["station"].get(station_id),
        config["station_weight"],
        config["station_min_count"],
    )
    evidence += _apply_baseline_component(
        combined_counts,
        baselines["station_category"].get((station_id, category)),
        config["station_category_weight"],
        config["station_category_min_count"],
    )

    network_entries = [
        baselines["network_category"].get((network, category))
        for network in _row_networks(row)
    ]
    network_entries = [
        entry
        for entry in network_entries
        if entry and entry["count"] >= config["network_category_min_count"]
    ]
    if network_entries:
        avg_counts = sum(
            np.asarray(entry["counts"], dtype=float) for entry in network_entries
        ) / len(network_entries)
        combined_counts += avg_counts * config["network_category_weight"]
        evidence += (
            sum(entry["count"] for entry in network_entries) / len(network_entries)
        ) * config["network_category_weight"]

    total = combined_counts.sum()
    if total <= 0:
        return None, evidence
    return combined_counts / total, evidence


def align_baseline_probs(full_probs, classes):
    if full_probs is None:
        return None

    aligned = np.asarray(
        [
            full_probs[int(cls)] if 0 <= int(cls) < len(full_probs) else 0.0
            for cls in classes
        ],
        dtype=float,
    )
    total = aligned.sum()
    if total <= 0:
        return None
    return aligned / total


def blend_prediction_probs(model_probs, baseline_probs, baseline_evidence, config):
    if baseline_probs is None or baseline_evidence <= 0:
        return model_probs

    baseline_weight = min(
        config["blend_max_weight"],
        baseline_evidence / (baseline_evidence + config["blend_count_scale"]),
    )
    blended = (1 - baseline_weight) * model_probs + baseline_weight * baseline_probs
    total = blended.sum()
    if total <= 0:
        return model_probs
    return blended / total


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _describe_class_distribution(y):
    counts = y.value_counts().sort_index()
    total = len(y)
    print("\nBlock offset class distribution:")
    for cls, n in counts.items():
        pct = n / total * 100
        print(f"  offset {cls:>2}: {n:>5} ({pct:.1f}%)")


def train_model(df, feature_cols):
    df = df.sort_values("start_time").reset_index(drop=True)

    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    _describe_class_distribution(y)

    # Weight recent data a bit more heavily, but keep the ramp gentle so we
    # do not overfit to the latest slice at the expense of the dominant
    # short-duration classes.
    n = len(df)
    sample_weights = np.linspace(1.0, 1.5, n)

    # Treat station_id and interaction features as categorical so the tree
    # model can learn per-group behaviour directly.
    categorical_cols = {"station_id", "category_x_start_block", "category_x_is_weekend"}
    categorical_mask = [col in categorical_cols for col in feature_cols]

    def _fresh_model():
        return HistGradientBoostingClassifier(
            max_iter=500,
            max_depth=6,
            learning_rate=0.08,
            min_samples_leaf=10,
            l2_regularization=0.3,
            categorical_features=categorical_mask,
            random_state=42,
        )

    # Time-series cross-validation on block accuracy.
    tscv = TimeSeriesSplit(n_splits=5)
    fold_scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        sw_train = sample_weights[train_idx]
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        baselines = build_historical_baselines(train_df)

        cv_model = _fresh_model()
        cv_model.fit(X_train, y_train, sample_weight=sw_train)
        model_probs = cv_model.predict_proba(X_test)
        preds = []
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
            pred_pos = int(np.argmax(blended_probs))
            preds.append(int(cv_model.classes_[pred_pos]))
        preds = np.asarray(preds, dtype=int)

        exact = accuracy_score(y_test, preds)
        predicted_block_idx = test_df["start_block_idx"].values + preds
        actual_end_times = test_df["end_time"].tolist()
        boundary_grace = np.mean(
            [
                prediction_is_close_enough(pred_idx, actual_dt)
                for pred_idx, actual_dt in zip(predicted_block_idx, actual_end_times)
            ]
        )
        # "Within ±1 block" is a softer success criterion — useful sanity check
        within_one = np.mean(np.abs(preds - y_test.values) <= 1)
        fold_scores.append((exact, boundary_grace, within_one))

    print(
        f"\nCross-validation results ({tscv.n_splits} splits, model + historical baselines):"
    )
    for i, (exact, boundary_grace, within_one) in enumerate(fold_scores):
        print(
            f"  fold {i + 1}: exact block {exact * 100:.1f}%, "
            f"within {int(PREDICTION_BOUNDARY_GRACE.total_seconds() / 60)}m of block {boundary_grace * 100:.1f}%, "
            f"within ±1 block {within_one * 100:.1f}%"
        )
    mean_exact = sum(s[0] for s in fold_scores) / len(fold_scores)
    mean_boundary = sum(s[1] for s in fold_scores) / len(fold_scores)
    mean_within = sum(s[2] for s in fold_scores) / len(fold_scores)
    print(
        f"  mean:   exact block {mean_exact * 100:.1f}%, "
        f"within {int(PREDICTION_BOUNDARY_GRACE.total_seconds() / 60)}m of block {mean_boundary * 100:.1f}%, "
        f"within ±1 block {mean_within * 100:.1f}%"
    )

    # Train final model on all data
    model = _fresh_model()
    model.fit(X, y, sample_weight=sample_weights)

    historical_baselines = build_historical_baselines(df)
    return model, historical_baselines


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def upload_model(model, vectorizer, historical_baselines, upload_url, key):
    model_path = "ml_model.joblib"
    joblib.dump({
        "model": model,
        "vectorizer": vectorizer,
        "historical_baselines": historical_baselines,
        "metadata": {
            "feature_version": 4,
        },
    }, model_path)
    print(f"\nModel saved locally to {model_path}")

    print(f"Uploading model to {upload_url}...")
    with open(model_path, "rb") as f:
        response = requests.post(
            upload_url,
            data={"key": key},
            files={"model": ("ml_model.joblib", f, "application/octet-stream")},
        )
    response.raise_for_status()
    print("Model uploaded successfully.")


# ---------------------------------------------------------------------------
# Calibration — how well did the *previous* model do?
# ---------------------------------------------------------------------------


def calibration_report(df):
    """Compare stored predictions to actuals using block accuracy."""
    if "estimated_duration_minutes" not in df.columns:
        return

    df = df.copy()
    df["start_time_dt"] = pd.to_datetime(
        df["start_time"], utc=True, format="ISO8601"
    ).dt.tz_convert(LOCAL_TZ)
    df["end_time_dt"] = pd.to_datetime(
        df["end_time"], utc=True, format="ISO8601"
    ).dt.tz_convert(LOCAL_TZ)

    has_prediction = df["estimated_duration_minutes"].notna()
    n_with = int(has_prediction.sum())
    n_total = len(df)
    print(f"\nCalibration report ({n_with}/{n_total} incidents had predictions)")

    if n_with == 0:
        print("  No previous predictions to evaluate.")
        return

    sub = df[has_prediction].copy()
    sub["predicted_end"] = sub["start_time_dt"] + pd.to_timedelta(
        sub["estimated_duration_minutes"], unit="m"
    )
    sub["predicted_block_idx"] = sub["predicted_end"].apply(block_index)
    sub["actual_block_idx"] = sub["end_time_dt"].apply(block_index)
    sub["correct"] = sub["predicted_block_idx"] == sub["actual_block_idx"]
    sub["close_enough"] = [
        prediction_is_close_enough(pred_idx, actual_dt)
        for pred_idx, actual_dt in zip(
            sub["predicted_block_idx"].tolist(),
            sub["end_time_dt"].tolist(),
        )
    ]
    sub["offset_error"] = (
        sub["predicted_block_idx"] - sub["actual_block_idx"]
    ).abs()

    acc = sub["correct"].mean()
    close_enough = sub["close_enough"].mean()
    within_one = (sub["offset_error"] <= 1).mean()
    print(f"  Exact block:       {acc * 100:.1f}%")
    print(
        f"  Within {int(PREDICTION_BOUNDARY_GRACE.total_seconds() / 60)}m of block: "
        f"{close_enough * 100:.1f}%"
    )
    print(f"  Within ±1 block:   {within_one * 100:.1f}%")

    categories = {
        "Faulty lift": sub["has_faulty_lift"].astype(bool),
        "Planned maintenance": sub["has_planned_maintenance"].astype(bool),
        "Staff issue": sub["has_staff_issue"].astype(bool),
    }
    print("\n  By category:")
    for label, mask in categories.items():
        if mask.sum() == 0:
            continue
        cat_acc = sub.loc[mask, "close_enough"].mean()
        print(
            f"    {label} (n={int(mask.sum())}): "
            f"{cat_acc * 100:.1f}% within {int(PREDICTION_BOUNDARY_GRACE.total_seconds() / 60)}m"
        )

    networks = ["tube", "dlr", "national_rail", "crossrail", "overground"]
    print("\n  By network:")
    for net in networks:
        mask = sub[net].astype(bool)
        if mask.sum() == 0:
            continue
        net_acc = sub.loc[mask, "close_enough"].mean()
        print(
            f"    {net} (n={int(mask.sum())}): "
            f"{net_acc * 100:.1f}% within {int(PREDICTION_BOUNDARY_GRACE.total_seconds() / 60)}m"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train incident duration prediction model"
    )
    parser.add_argument(
        "--api-url",
        required=True,
        help="URL to fetch training data (e.g. https://updownlondon.com/api/training-data/)",
    )
    parser.add_argument(
        "--upload-url",
        required=True,
        help="URL to upload trained model (e.g. https://updownlondon.com/functions/upload_model)",
    )
    parser.add_argument(
        "--key",
        required=True,
        help="Functions secret key for API authentication",
    )
    args = parser.parse_args()

    df = fetch_training_data(args.api_url, args.key)
    calibration_report(df)
    df, feature_cols, vectorizer = prepare_features(df)

    print(f"\nTraining on {len(df)} incidents with {len(feature_cols)} features...")
    model, historical_baselines = train_model(df, feature_cols)

    upload_model(
        model,
        vectorizer,
        historical_baselines,
        args.upload_url,
        args.key,
    )


if __name__ == "__main__":
    main()
