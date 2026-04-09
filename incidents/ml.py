import os
from datetime import timedelta
from functools import lru_cache

from django.conf import settings
from django.db.models import Avg, Count, F
from django.db.models.functions import Extract


@lru_cache(maxsize=1)
def _load_model():
    model_path = os.path.join(settings.BASE_DIR, "ml_model.joblib")
    if not os.path.exists(model_path):
        return None
    try:
        import joblib
    except ImportError:
        return None
    data = joblib.load(model_path)
    # Support old (bare model) format
    if not isinstance(data, dict):
        return {"model": data}
    return data


def _is_planned_work(incident):
    text = incident.text.lower()
    return "planned" in text or "until " in text or incident.information


def predict_duration(incident):
    """Returns (timedelta, confidence) or (None, None)."""
    data = _load_model()
    if data is None:
        return None, None

    # Skip planned work — duration is in the text
    if _is_planned_work(incident):
        return None, None

    model = data["model"]
    vectorizer = data.get("vectorizer")
    model_lower = data.get("model_lower")
    model_upper = data.get("model_upper")

    from incidents.models import Incident

    station = incident.station
    text = incident.text.lower()

    # Compute per-station historical stats from resolved incidents
    station_stats = (
        Incident.objects.filter(
            station=station, resolved=True, end_time__isnull=False
        )
        .annotate(dur_seconds=Extract(F("end_time") - F("start_time"), "epoch"))
        .aggregate(
            station_mean_duration=Avg("dur_seconds"),
            station_incident_count=Count("id"),
        )
    )
    mean_dur = (station_stats["station_mean_duration"] or 0) / 60
    count = station_stats["station_incident_count"] or 0

    features = {
        "station_id": station.id,
        "information": int(incident.information),
        "hour_of_day": incident.start_time.hour,
        "day_of_week": incident.start_time.weekday(),
        "month": incident.start_time.month,
        "has_faulty_lift": int("faulty lift" in text),
        "has_planned_maintenance": int("planned maintenance" in text),
        "has_staff_issue": int("staff" in text),
        "is_planned_work": 0,  # always 0 — we skip planned work above
        "tube": int(bool(station.tube)),
        "dlr": int(bool(station.dlr)),
        "national_rail": int(bool(station.national_rail)),
        "crossrail": int(bool(station.crossrail)),
        "overground": int(bool(station.overground)),
        "access_via_lift": int(bool(station.access_via_lift)),
        "station_mean_duration": mean_dur,
        "station_median_duration": mean_dur,  # approximation at prediction time
        "station_incident_count": count,
    }

    try:
        import pandas as pd
    except ImportError:
        return None, None

    df = pd.DataFrame([features])

    # Add TF-IDF features if vectorizer is available
    if vectorizer is not None:
        tfidf_matrix = vectorizer.transform([incident.text or ""])
        tfidf_cols = [f"tfidf_{name}" for name in vectorizer.get_feature_names_out()]
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_cols, index=df.index)
        df = pd.concat([df, tfidf_df], axis=1)

    import numpy as np

    # Model predicts log1p(minutes), so inverse transform
    predicted_minutes = np.expm1(model.predict(df)[0])

    # Clamp to reasonable range: 5 minutes to 30 days
    predicted_minutes = max(5, min(predicted_minutes, 60 * 24 * 30))

    # Compute confidence from quantile interval width
    confidence = None
    if model_lower is not None and model_upper is not None:
        lower = np.expm1(model_lower.predict(df)[0])
        upper = np.expm1(model_upper.predict(df)[0])
        if predicted_minutes > 0:
            # Narrow interval relative to prediction = high confidence
            spread = max(0, upper - lower)
            confidence = max(0.05, min(0.95, 1 - spread / predicted_minutes))
            confidence = round(confidence, 2)

    return timedelta(minutes=predicted_minutes), confidence


# Allow cache clearing when a new model is uploaded
predict_duration.cache_clear = _load_model.cache_clear
