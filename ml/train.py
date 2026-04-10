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
from sklearn.utils.class_weight import compute_sample_weight


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
    "concurrent_incidents",
]

TARGET_COLUMN = "block_offset"


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


def prepare_features(df):
    # Parse datetime columns and convert to local time. Blocks are defined
    # in London time, not UTC — otherwise the boundaries drift by an hour
    # during BST.
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True).dt.tz_convert(LOCAL_TZ)
    df["end_time"] = pd.to_datetime(df["end_time"], utc=True).dt.tz_convert(LOCAL_TZ)

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

    # Derived features
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

    # Per-station historical aggregates. The raw station_id is also fed in
    # as a categorical so the tree model can learn per-station behaviour
    # directly ("this manager is shit"). These aggregates give a smoother
    # fallback for stations with too few incidents for the tree to learn
    # anything station-specific.
    station_stats = df.groupby("station_id").agg(
        station_mean_duration=("duration_minutes", "mean"),
        station_incident_count=("duration_minutes", "count"),
        station_mean_offset=(TARGET_COLUMN, "mean"),
    )
    df = df.merge(station_stats, on="station_id", how="left")

    # TF-IDF on incident text
    vectorizer = TfidfVectorizer(max_features=50, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["text"].fillna(""))
    tfidf_cols = [f"tfidf_{name}" for name in vectorizer.get_feature_names_out()]
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(), columns=tfidf_cols, index=df.index
    )
    df = pd.concat([df, tfidf_df], axis=1)

    feature_cols = (
        FEATURE_COLUMNS
        + ["station_mean_duration", "station_incident_count", "station_mean_offset"]
        + tfidf_cols
    )

    return df, feature_cols, vectorizer


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

    # Class-balanced sample weights so the classifier doesn't just learn
    # "always say offset=X" for the most common class.
    class_weights = compute_sample_weight(class_weight="balanced", y=y)

    # Additionally weight recent data more heavily. Linear ramp from 1.0
    # (oldest) to 3.0 (newest) so the last few months matter more.
    n = len(df)
    recency_weights = np.linspace(1.0, 3.0, n)
    sample_weights = class_weights * recency_weights

    # Treat station_id as a proper categorical so the tree model can learn
    # per-station behaviour directly, instead of treating it as an ordinal
    # number that happens to sit next to other station IDs.
    categorical_mask = [col == "station_id" for col in feature_cols]

    def _fresh_model():
        return HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=6,
            learning_rate=0.08,
            min_samples_leaf=20,
            l2_regularization=0.5,
            categorical_features=categorical_mask,
            random_state=42,
        )

    # Time-series cross-validation on block accuracy.
    tscv = TimeSeriesSplit(n_splits=3)
    fold_scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        sw_train = sample_weights[train_idx]

        cv_model = _fresh_model()
        cv_model.fit(X_train, y_train, sample_weight=sw_train)
        preds = cv_model.predict(X_test)

        exact = accuracy_score(y_test, preds)
        # "Within ±1 block" is a softer success criterion — useful sanity check
        within_one = np.mean(np.abs(preds - y_test.values) <= 1)
        fold_scores.append((exact, within_one))

    print(f"\nCross-validation results ({tscv.n_splits} splits):")
    for i, (exact, within_one) in enumerate(fold_scores):
        print(
            f"  fold {i + 1}: exact block {exact * 100:.1f}%, "
            f"within ±1 block {within_one * 100:.1f}%"
        )
    mean_exact = sum(s[0] for s in fold_scores) / len(fold_scores)
    mean_within = sum(s[1] for s in fold_scores) / len(fold_scores)
    print(
        f"  mean:   exact block {mean_exact * 100:.1f}%, "
        f"within ±1 block {mean_within * 100:.1f}%"
    )

    # Train final model on all data
    model = _fresh_model()
    model.fit(X, y, sample_weight=sample_weights)

    return model


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def upload_model(model, vectorizer, upload_url, key):
    model_path = "ml_model.joblib"
    joblib.dump({
        "model": model,
        "vectorizer": vectorizer,
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
    df["start_time_dt"] = pd.to_datetime(df["start_time"], utc=True).dt.tz_convert(LOCAL_TZ)
    df["end_time_dt"] = pd.to_datetime(df["end_time"], utc=True).dt.tz_convert(LOCAL_TZ)

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
    sub["offset_error"] = (
        sub["predicted_block_idx"] - sub["actual_block_idx"]
    ).abs()

    acc = sub["correct"].mean()
    within_one = (sub["offset_error"] <= 1).mean()
    print(f"  Exact block:       {acc * 100:.1f}%")
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
        cat_acc = sub.loc[mask, "correct"].mean()
        print(f"    {label} (n={int(mask.sum())}): {cat_acc * 100:.1f}% exact")

    networks = ["tube", "dlr", "national_rail", "crossrail", "overground"]
    print("\n  By network:")
    for net in networks:
        mask = sub[net].astype(bool)
        if mask.sum() == 0:
            continue
        net_acc = sub.loc[mask, "correct"].mean()
        print(f"    {net} (n={int(mask.sum())}): {net_acc * 100:.1f}% exact")


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
    model = train_model(df, feature_cols)

    upload_model(model, vectorizer, args.upload_url, args.key)


if __name__ == "__main__":
    main()
