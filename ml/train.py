#!/usr/bin/env python3
"""
Standalone training script for the incident duration prediction model.

Usage:
    python train.py \
        --api-url https://updownlondon.com/api/training-data/ \
        --upload-url https://updownlondon.com/functions/upload_model \
        --key YOUR_FUNCTIONS_SECRET_KEY

This script:
1. Pulls training data from the Up Down London API
2. Trains a gradient-boosted regression model to predict incident duration
3. Uploads the trained model back to the server
"""

import argparse
import sys

import joblib
import pandas as pd
import requests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, median_absolute_error


FEATURE_COLUMNS = [
    "station_id",
    "information",
    "hour_of_day",
    "day_of_week",
    "month",
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

TARGET_COLUMN = "duration_minutes"


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


def prepare_features(df):
    # Convert booleans to ints
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

    # Add per-station historical features
    station_stats = df.groupby("station_id")[TARGET_COLUMN].agg(
        station_mean_duration="mean",
        station_median_duration="median",
        station_incident_count="count",
    )
    df = df.merge(station_stats, on="station_id", how="left")

    feature_cols = FEATURE_COLUMNS + [
        "station_mean_duration",
        "station_median_duration",
        "station_incident_count",
    ]

    return df, feature_cols


def train_model(df, feature_cols):
    # Sort by start_time for proper time-series splitting
    df = df.sort_values("start_time").reset_index(drop=True)

    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_leaf=5,
        random_state=42,
    )

    # Evaluate with time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    mae_scores = []
    median_ae_scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae_scores.append(mean_absolute_error(y_test, preds))
        median_ae_scores.append(median_absolute_error(y_test, preds))

    print(f"\nCross-validation results ({tscv.n_splits} splits):")
    print(f"  Mean Absolute Error:   {sum(mae_scores)/len(mae_scores):.1f} minutes")
    print(
        f"  Median Absolute Error: "
        f"{sum(median_ae_scores)/len(median_ae_scores):.1f} minutes"
    )

    # Train final model on all data
    model.fit(X, y)

    # Feature importance
    print("\nFeature importance:")
    importances = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    for name, importance in importances:
        print(f"  {name}: {importance:.4f}")

    return model


def upload_model(model, upload_url, key):
    model_path = "ml_model.joblib"
    joblib.dump(model, model_path)
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
    df, feature_cols = prepare_features(df)

    print(f"\nTraining on {len(df)} incidents with {len(feature_cols)} features...")
    model = train_model(df, feature_cols)

    upload_model(model, args.upload_url, args.key)


if __name__ == "__main__":
    main()
