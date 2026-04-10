import os
from datetime import date as date_cls
from datetime import datetime, time, timedelta
from functools import lru_cache

from django.conf import settings


# ---------------------------------------------------------------------------
# Time-of-day blocks
# ---------------------------------------------------------------------------
#
# Incidents are predicted into one of four coarse blocks per day:
#   morning   05:00 – 10:00
#   daytime   10:00 – 15:00
#   evening   15:00 – 20:00
#   overnight 20:00 – 05:00 (next day)
#
# Every (date, block) pair has a unique integer ``block_index`` computed as
# ``date.toordinal() * 4 + block_number``.  Overnight is anchored to the date
# the block *starts*, so 22:00 Monday and 02:00 Tuesday share the same index.

BLOCK_MORNING = 0
BLOCK_DAYTIME = 1
BLOCK_EVENING = 2
BLOCK_OVERNIGHT = 3
BLOCKS_PER_DAY = 4
BLOCK_NAMES = ["morning", "daytime", "evening", "overnight"]

# Everything at or beyond this many blocks from the start is lumped into one
# "long tail" class during training. Five days is plenty of detail; beyond
# that we barely have the data to distinguish blocks anyway.
MAX_OFFSET_CLASS = 20


def _to_local(dt):
    """Return ``dt`` expressed in the project's configured timezone.

    Blocks are defined in local (London) time, but Django stores everything
    as UTC. Without this conversion morning/daytime/etc. boundaries shift by
    an hour during BST.
    """
    from django.utils import timezone as tz_module

    if tz_module.is_aware(dt):
        return tz_module.localtime(dt)
    return dt


def block_index(dt):
    """Monotonic integer identifying the (date, block) slot containing ``dt``."""
    dt = _to_local(dt)
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
    """Return (naive start_dt, naive end_dt) for the block at ``index``."""
    date_ord, block_num = divmod(index, BLOCKS_PER_DAY)
    day = date_cls.fromordinal(date_ord)
    if block_num == BLOCK_MORNING:
        return datetime.combine(day, time(5, 0)), datetime.combine(day, time(10, 0))
    if block_num == BLOCK_DAYTIME:
        return datetime.combine(day, time(10, 0)), datetime.combine(day, time(15, 0))
    if block_num == BLOCK_EVENING:
        return datetime.combine(day, time(15, 0)), datetime.combine(day, time(20, 0))
    # overnight: 20:00 → 05:00 next day
    start = datetime.combine(day, time(20, 0))
    return start, start + timedelta(hours=9)


def time_block_slot(dt):
    """Return a (date, block_name) tuple identifying a coarse time slot."""
    idx = block_index(dt)
    date_ord, block_num = divmod(idx, BLOCKS_PER_DAY)
    return (date_cls.fromordinal(date_ord), BLOCK_NAMES[block_num])


def format_block_slot(slot, now):
    """Human-friendly label for a (date, block) slot relative to ``now``."""
    date, block = slot
    today = now.date()
    delta = (date - today).days
    if delta == 0:
        return {
            "morning": "this morning",
            "daytime": "this afternoon",
            "evening": "this evening",
            "overnight": "tonight",
        }[block]
    if delta == 1:
        if block == "overnight":
            return "tomorrow night"
        return f"tomorrow {block}"
    if delta == -1:
        return f"yesterday {block}"
    if 1 < delta < 7:
        return f"{date.strftime('%A')} {block}"
    if -7 < delta < -1:
        return f"last {date.strftime('%A')} {block}"
    return f"{date.strftime('%-d %b')} {block}"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


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
    if not isinstance(data, dict):
        return {"model": data}
    return data


def _is_planned_work(incident):
    text = incident.text.lower()
    return "planned" in text or "until " in text or incident.information


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_duration(incident):
    """Returns (timedelta, confidence) or (None, None).

    The model is a classifier over "block offset" — how many coarse time
    blocks after the incident's start block we expect it to be fixed in.
    We decode the predicted class back into a representative timedelta so
    callers keep getting a duration.
    """
    data = _load_model()
    if data is None:
        return None, None

    if _is_planned_work(incident):
        return None, None

    model = data["model"]
    vectorizer = data.get("vectorizer")

    from django.db.models import Q
    from django.utils import timezone as tz_module

    from incidents.models import Incident

    # Roll the station up to its parent — a station like "Bank" has
    # separate child records per line but is one place from the user's
    # point of view, and that's where the station-manager-level signal lives.
    station = incident.station.parent_station or incident.station
    text = incident.text.lower()

    # Historical filter: the effective station itself, or any of its
    # children. Covers both possibilities regardless of which record the
    # incident was originally filed against.
    station_filter = Q(station=station) | Q(station__parent_station=station)

    # Per-station historical stats from resolved incidents. We pull the raw
    # start/end times so we can compute both mean duration and mean block
    # offset in a single query — the training pipeline uses the same two
    # features, so they have to match.
    past = list(
        Incident.objects.filter(
            station_filter, resolved=True, end_time__isnull=False
        ).values_list("start_time", "end_time")
    )
    count = len(past)
    if count:
        mean_dur = sum((e - s).total_seconds() for s, e in past) / count / 60
        offsets = [
            max(0, min(MAX_OFFSET_CLASS, block_index(e) - block_index(s)))
            for s, e in past
        ]
        mean_offset = sum(offsets) / count
    else:
        mean_dur = 0
        mean_offset = 0

    if hasattr(incident, "prefetched_reports"):
        num_reports = len(incident.prefetched_reports)
    elif incident.pk:
        num_reports = incident.reports.count()
    else:
        num_reports = 1

    prev = (
        Incident.objects.filter(
            station_filter, start_time__lt=incident.start_time
        )
        .order_by("-start_time")
        .values_list("start_time", flat=True)
        .first()
    )
    if prev:
        days_since_last = (incident.start_time - prev).total_seconds() / 86400
    else:
        days_since_last = -1

    concurrent = Incident.objects.filter(
        resolved=False, start_time__lte=incident.start_time
    ).exclude(pk=incident.pk).count()
    if concurrent == 0:
        concurrent = Incident.objects.filter(resolved=False).exclude(
            pk=incident.pk
        ).count()

    start_idx = block_index(incident.start_time)
    start_block_num = start_idx % BLOCKS_PER_DAY

    features = {
        "station_id": station.id,
        "information": int(incident.information),
        "hour_of_day": incident.start_time.hour,
        "day_of_week": incident.start_time.weekday(),
        "month": incident.start_time.month,
        "is_weekend": int(incident.start_time.weekday() >= 5),
        "start_block": start_block_num,
        "has_faulty_lift": int("faulty lift" in text),
        "has_planned_maintenance": int("planned maintenance" in text),
        "has_staff_issue": int("staff" in text),
        "tube": int(bool(station.tube)),
        "dlr": int(bool(station.dlr)),
        "national_rail": int(bool(station.national_rail)),
        "crossrail": int(bool(station.crossrail)),
        "overground": int(bool(station.overground)),
        "access_via_lift": int(bool(station.access_via_lift)),
        "num_reports": num_reports,
        "days_since_last_incident": round(days_since_last, 2),
        "concurrent_incidents": concurrent,
        "station_mean_duration": mean_dur,
        "station_incident_count": count,
        "station_mean_offset": mean_offset,
    }

    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        return None, None

    df = pd.DataFrame([features])

    if vectorizer is not None:
        tfidf_matrix = vectorizer.transform([incident.text or ""])
        tfidf_cols = [f"tfidf_{name}" for name in vectorizer.get_feature_names_out()]
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(), columns=tfidf_cols, index=df.index
        )
        df = pd.concat([df, tfidf_df], axis=1)

    # Classifier predicts a block offset; confidence is the class probability.
    probs = model.predict_proba(df)[0]
    pred_pos = int(np.argmax(probs))
    pred_offset = int(model.classes_[pred_pos])
    confidence = float(probs[pred_pos])
    confidence = max(0.05, min(0.95, confidence))

    # Decode offset → representative end datetime. block_start_end returns
    # naive datetimes in local (London) time, so attach the project's
    # configured timezone before doing any arithmetic with the incident's
    # (UTC-stored) start_time.
    end_idx = start_idx + pred_offset
    block_start_naive, block_end_naive = block_start_end(end_idx)
    local_tz = tz_module.get_default_timezone()
    block_start_dt = block_start_naive.replace(tzinfo=local_tz)
    block_end_dt = block_end_naive.replace(tzinfo=local_tz)

    if pred_offset == 0:
        # Same block as the start — midpoint between now and block end.
        end_time = incident.start_time + (block_end_dt - incident.start_time) / 2
    else:
        end_time = block_start_dt + (block_end_dt - block_start_dt) / 2

    # Guarantee strictly positive duration of at least 5 minutes.
    min_end = incident.start_time + timedelta(minutes=5)
    if end_time < min_end:
        end_time = min_end

    duration = end_time - incident.start_time
    # Clamp to 30-day ceiling like before.
    max_duration = timedelta(days=30)
    if duration > max_duration:
        duration = max_duration

    return duration, round(confidence, 2)


# Allow cache clearing when a new model is uploaded
predict_duration.cache_clear = _load_model.cache_clear
