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
        return None, None
    try:
        import joblib
    except ImportError:
        return None, None
    data = joblib.load(model_path)
    # Support both old (bare model) and new (dict with vectorizer) formats
    if isinstance(data, dict):
        return data["model"], data.get("vectorizer")
    return data, None


def predict_duration(incident):
    model, vectorizer = _load_model()
    if model is None:
        return None

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
        "is_planned_work": int(
            "planned" in text
            or "until " in text
            or incident.information
        ),
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
        return None

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

    return timedelta(minutes=predicted_minutes)


# Allow cache clearing when a new model is uploaded
predict_duration.cache_clear = _load_model.cache_clear
