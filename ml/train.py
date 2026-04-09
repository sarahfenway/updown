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
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
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
    "is_planned_work",
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
    # Exclude planned work — duration is in the text, no point predicting
    planned_count = df["is_planned_work"].sum()
    df = df[~df["is_planned_work"].astype(bool)].copy()
    print(f"Excluded {planned_count} planned work incidents, {len(df)} remaining.")

    # Convert booleans to ints
    bool_columns = [
        "information",
        "has_faulty_lift",
        "has_planned_maintenance",
        "has_staff_issue",
        "is_planned_work",
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

    # TF-IDF on incident text
    vectorizer = TfidfVectorizer(max_features=50, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["text"].fillna(""))
    tfidf_cols = [f"tfidf_{name}" for name in vectorizer.get_feature_names_out()]
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_cols, index=df.index)
    df = pd.concat([df, tfidf_df], axis=1)

    feature_cols = FEATURE_COLUMNS + [
        "station_mean_duration",
        "station_median_duration",
        "station_incident_count",
    ] + tfidf_cols

    return df, feature_cols, vectorizer


def train_model(df, feature_cols):
    # Sort by start_time for proper time-series splitting
    df = df.sort_values("start_time").reset_index(drop=True)

    X = df[feature_cols]
    # Log-transform target so the model optimises for relative error,
    # not absolute minutes — stops long planned outages dominating training
    y_log = np.log1p(df[TARGET_COLUMN])

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
        y_train, y_test = y_log.iloc[train_idx], y_log.iloc[test_idx]

        model.fit(X_train, y_train)
        preds_log = model.predict(X_test)

        # Convert back to minutes for reporting
        preds = np.expm1(preds_log)
        y_actual = np.expm1(y_test)

        mae_scores.append(mean_absolute_error(y_actual, preds))
        median_ae_scores.append(median_absolute_error(y_actual, preds))

    print(f"\nCross-validation results ({tscv.n_splits} splits):")
    print(f"  Mean Absolute Error:   {sum(mae_scores)/len(mae_scores):.1f} minutes")
    print(
        f"  Median Absolute Error: "
        f"{sum(median_ae_scores)/len(median_ae_scores):.1f} minutes"
    )

    # Train final model on all data
    model.fit(X, y_log)

    # Train quantile models for confidence intervals
    model_lower = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        min_samples_leaf=5, random_state=42,
        loss="quantile", alpha=0.25,
    )
    model_upper = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        min_samples_leaf=5, random_state=42,
        loss="quantile", alpha=0.75,
    )
    model_lower.fit(X, y_log)
    model_upper.fit(X, y_log)

    # Feature importance
    print("\nFeature importance:")
    importances = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    for name, importance in importances:
        print(f"  {name}: {importance:.4f}")

    return model, model_lower, model_upper


def upload_model(model, model_lower, model_upper, vectorizer, upload_url, key):
    model_path = "ml_model.joblib"
    joblib.dump({
        "model": model,
        "model_lower": model_lower,
        "model_upper": model_upper,
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


def calibration_report(df):
    predicted = df["estimated_duration_minutes"]
    actual = df["duration_minutes"]
    has_prediction = predicted.notna()

    n_with = has_prediction.sum()
    n_total = len(df)
    print(f"\nCalibration report ({n_with}/{n_total} incidents had predictions)")

    if n_with == 0:
        print("  No previous predictions to evaluate.")
        return

    predicted = predicted[has_prediction]
    actual = actual[has_prediction]
    error = predicted - actual  # positive = overestimate

    print(f"  Mean error:            {error.mean():+.1f} min (positive = overestimate)")
    print(f"  Mean absolute error:   {error.abs().mean():.1f} min")
    print(f"  Median absolute error: {error.abs().median():.1f} min")

    # Breakdown by incident type
    categories = {
        "Faulty lift": df.loc[has_prediction, "has_faulty_lift"].astype(bool),
        "Planned maintenance": df.loc[has_prediction, "has_planned_maintenance"].astype(bool),
        "Staff issue": df.loc[has_prediction, "has_staff_issue"].astype(bool),
    }
    print("\n  By category:")
    for label, mask in categories.items():
        if mask.sum() == 0:
            continue
        cat_error = error[mask]
        print(
            f"    {label} (n={mask.sum()}): "
            f"mean error {cat_error.mean():+.1f} min, "
            f"MAE {cat_error.abs().mean():.1f} min"
        )

    # Breakdown by station network
    networks = ["tube", "dlr", "national_rail", "crossrail", "overground"]
    print("\n  By network:")
    for net in networks:
        mask = df.loc[has_prediction, net].astype(bool)
        if mask.sum() == 0:
            continue
        net_error = error[mask]
        print(
            f"    {net} (n={mask.sum()}): "
            f"mean error {net_error.mean():+.1f} min, "
            f"MAE {net_error.abs().mean():.1f} min"
        )


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
    model, model_lower, model_upper = train_model(df, feature_cols)

    upload_model(model, model_lower, model_upper, vectorizer, args.upload_url, args.key)


if __name__ == "__main__":
    main()
