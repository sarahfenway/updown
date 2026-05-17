import io
import logging
import os
from datetime import date as date_cls
from datetime import datetime, time, timedelta

from django.conf import settings

from incidents.text_features import normalise_incident_text


logger = logging.getLogger(__name__)


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
PREDICTION_BOUNDARY_GRACE = timedelta(minutes=30)

# Everything at or beyond this many blocks from the start is lumped into one
# "long tail" class during training. Five days is plenty of detail; beyond
# that we barely have the data to distinguish blocks anyway.
MAX_OFFSET_CLASS = 20

# Predictions are framed as a one-sided "fixed by <time>" bound rather than
# "fixed at <block>". We pick the earliest block by which the model's
# cumulative probability of resolution reaches this target. The model is
# only ~57% accurate at the exact block but ~72% within ±1 block, so a
# cumulative bound is honest about the real uncertainty while still being
# actionable ("won't be fixed before this evening").
PREDICTION_COVERAGE_TARGET = 0.75

NETWORK_FIELDS = ("tube", "dlr", "national_rail", "crossrail", "overground")
BASELINE_CATEGORY_OTHER = "other"
HISTORICAL_BASELINE_DEFAULTS = {
    "global_weight": 1.0,
    "category_weight": 1.0,
    "network_category_weight": 0.5,
    "station_weight": 0.75,
    "station_category_weight": 1.25,
    "category_min_count": 8,
    "network_category_min_count": 8,
    "station_min_count": 5,
    "station_category_min_count": 3,
    "blend_max_weight": 0.35,
    "blend_count_scale": 25.0,
}


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


def _aware_block_bounds(index):
    from django.utils import timezone as tz_module

    start_naive, end_naive = block_start_end(index)
    local_tz = tz_module.get_default_timezone()
    return start_naive.replace(tzinfo=local_tz), end_naive.replace(tzinfo=local_tz)


def block_distance_from_index(dt, index):
    local_dt = _to_local(dt)
    block_start_dt, block_end_dt = _aware_block_bounds(index)

    if local_dt < block_start_dt:
        return block_start_dt - local_dt
    if local_dt > block_end_dt:
        return local_dt - block_end_dt
    return timedelta(0)


def prediction_is_close_enough(predicted_dt, actual_dt, grace=PREDICTION_BOUNDARY_GRACE):
    """One-sided "fixed by" check.

    ``predicted_dt`` is the time we told the user it would be fixed *by*.
    The promise held if the incident was actually resolved at or before
    that time. Resolving earlier still counts — we never claimed it
    wouldn't be sooner. A small grace absorbs cases that land just the
    wrong side of a block boundary.
    """
    return _to_local(actual_dt) <= _to_local(predicted_dt) + grace


def prediction_outcome(predicted_dt, actual_dt, grace=PREDICTION_BOUNDARY_GRACE):
    """Outcome of a one-sided "fixed by" prediction.

    The promise was that it would be resolved by ``predicted_dt``. Being
    resolved earlier is a full success — we never claimed it wouldn't be
    sooner — so "exact" means resolved at or before the promised time,
    whether early or bang on. "near" is a soft miss: it overran, but only
    within the grace window. Anything later is a "miss".
    """
    actual = _to_local(actual_dt)
    predicted = _to_local(predicted_dt)

    if actual <= predicted:
        return "exact"
    if actual <= predicted + grace:
        return "near"
    return "miss"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


# Process-local cache so we don't redeserialize on every prediction. We
# additionally remember the id of the MLModel row we deserialized, so that a
# fresh upload is picked up by long-running processes (e.g. the web dyno) on
# the next call without needing to restart.
_cached_model = None
_cached_model_row_id = None


def _load_model():
    global _cached_model, _cached_model_row_id

    try:
        import joblib
    except ImportError:
        logger.warning("joblib not installed — predictions disabled")
        return None

    from incidents.models import MLModel

    latest = MLModel.objects.order_by("-id").values("id").first()
    if latest is None:
        # Fall back to an on-disk file if one happens to exist (legacy path
        # and local dev convenience). In production this will normally miss.
        model_path = os.path.join(settings.BASE_DIR, "ml_model.joblib")
        if os.path.exists(model_path):
            if _cached_model_row_id != ("file", os.path.getmtime(model_path)):
                data = joblib.load(model_path)
                if not isinstance(data, dict):
                    data = {"model": data}
                _cached_model = data
                _cached_model_row_id = ("file", os.path.getmtime(model_path))
            return _cached_model
        _cached_model = None
        _cached_model_row_id = None
        return None

    latest_id = latest["id"]
    if latest_id == _cached_model_row_id:
        return _cached_model

    # New (or first) model — pull the bytes and deserialize.
    row = MLModel.objects.only("data").get(pk=latest_id)
    raw = bytes(row.data)
    try:
        data = joblib.load(io.BytesIO(raw))
    except Exception:
        logger.exception("Failed to deserialize MLModel row %s", latest_id)
        return None
    if not isinstance(data, dict):
        data = {"model": data}
    _cached_model = data
    _cached_model_row_id = latest_id
    return _cached_model


def _clear_model_cache():
    global _cached_model, _cached_model_row_id
    _cached_model = None
    _cached_model_row_id = None


def _is_planned_work(incident):
    text = normalise_incident_text(incident.text).lower()
    return "planned" in text or "until " in text or incident.information


def _prediction_category_from_text(text):
    text = (text or "").lower()
    if "faulty lift" in text:
        return "faulty_lift"
    if "planned maintenance" in text:
        return "planned_maintenance"
    if "staff" in text:
        return "staff_issue"
    return BASELINE_CATEGORY_OTHER


def _station_networks(station):
    return [network for network in NETWORK_FIELDS if getattr(station, network, False)]


def _apply_baseline_component(combined_counts, entry, weight, min_count):
    if not entry or entry.get("count", 0) < min_count:
        return 0.0

    import numpy as np

    combined_counts += np.asarray(entry["counts"], dtype=float) * weight
    return entry["count"] * weight


def _historical_baseline_probs(historical_baselines, station, text, classes):
    if not historical_baselines or not historical_baselines.get("global"):
        return None, 0.0

    try:
        import numpy as np
    except ImportError:
        return None, 0.0

    config = {
        **HISTORICAL_BASELINE_DEFAULTS,
        **(historical_baselines.get("config") or {}),
    }

    combined_counts = (
        np.asarray(historical_baselines["global"]["counts"], dtype=float)
        * config["global_weight"]
    )
    evidence = 0.0
    category = _prediction_category_from_text(text)

    evidence += _apply_baseline_component(
        combined_counts,
        historical_baselines.get("category", {}).get(category),
        config["category_weight"],
        config["category_min_count"],
    )
    evidence += _apply_baseline_component(
        combined_counts,
        historical_baselines.get("station", {}).get(station.id),
        config["station_weight"],
        config["station_min_count"],
    )
    evidence += _apply_baseline_component(
        combined_counts,
        historical_baselines.get("station_category", {}).get((station.id, category)),
        config["station_category_weight"],
        config["station_category_min_count"],
    )

    network_entries = [
        historical_baselines.get("network_category", {}).get((network, category))
        for network in _station_networks(station)
    ]
    network_entries = [
        entry
        for entry in network_entries
        if entry and entry.get("count", 0) >= config["network_category_min_count"]
    ]
    if network_entries:
        avg_counts = sum(
            np.asarray(entry["counts"], dtype=float) for entry in network_entries
        ) / len(network_entries)
        combined_counts += avg_counts * config["network_category_weight"]
        evidence += (
            sum(entry["count"] for entry in network_entries) / len(network_entries)
        ) * config["network_category_weight"]

    full_total = combined_counts.sum()
    if full_total <= 0:
        return None, evidence

    full_probs = combined_counts / full_total
    aligned = np.asarray(
        [
            full_probs[int(cls)] if 0 <= int(cls) < len(full_probs) else 0.0
            for cls in classes
        ],
        dtype=float,
    )
    aligned_total = aligned.sum()
    if aligned_total <= 0:
        return None, evidence
    return aligned / aligned_total, evidence


def _blend_prediction_probs(model_probs, baseline_probs, baseline_evidence, config=None):
    if baseline_probs is None or baseline_evidence <= 0:
        return model_probs

    config = {**HISTORICAL_BASELINE_DEFAULTS, **(config or {})}
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
# Prediction
# ---------------------------------------------------------------------------


def predict_duration(incident):
    """Public entry point. Wraps :func:`_predict_duration` in a broad
    try/except so a single misbehaving prediction can never crash a cron
    worker or a web request. On any failure we log with traceback and
    return ``(None, None)`` so callers treat it as "no prediction".
    """
    try:
        return _predict_duration(incident)
    except Exception:
        logger.exception(
            "predict_duration failed for incident %s at %s",
            getattr(incident, "pk", None),
            getattr(getattr(incident, "station", None), "name", "?"),
        )
        return None, None


def _predict_duration(incident):
    """Returns (timedelta, confidence) or (None, None).

    The model is a classifier over "block offset" — how many coarse time
    blocks after the incident's start block it resolves in. We don't decode
    the single most-likely block; instead we walk the cumulative
    distribution and return the earliest block by which there's a
    ``PREDICTION_COVERAGE_TARGET`` chance it's resolved. The duration
    therefore represents a one-sided "fixed by" bound, and ``confidence``
    is the cumulative probability mass under that bound.
    """
    data = _load_model()
    if data is None:
        return None, None

    if _is_planned_work(incident):
        return None, None

    model = data["model"]
    vectorizer = data.get("vectorizer")
    historical_baselines = data.get("historical_baselines")
    metadata = data.get("metadata") or {}
    feature_version = metadata.get("feature_version", 1)
    use_v2_features = feature_version >= 2

    from django.db.models import Q
    from django.utils import timezone as tz_module

    from incidents.models import Incident

    # Roll the station up to its parent — a station like "Bank" has
    # separate child records per line but is one place from the user's
    # point of view, and that's where the station-manager-level signal lives.
    station = incident.station.parent_station or incident.station
    feature_text = normalise_incident_text(incident.text) if use_v2_features else (incident.text or "")
    text = feature_text.lower()
    start_time_local = _to_local(incident.start_time) if use_v2_features else incident.start_time

    # Historical filter: the effective station itself, or any of its
    # children. Covers both possibilities regardless of which record the
    # incident was originally filed against.
    station_filter = Q(station=station) | Q(station__parent_station=station)

    # Per-station historical stats from resolved incidents. We pull the raw
    # start/end times so we can compute both mean duration and mean block
    # offset in a single query — the training pipeline uses the same two
    # features, so they have to match.
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
    else:
        mean_dur = 0
        mean_offset = 0

    if hasattr(incident, "prefetched_reports"):
        num_reports = len(incident.prefetched_reports)
    elif hasattr(incident, "num_reports"):
        num_reports = incident.num_reports or 0
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

    if use_v2_features:
        concurrent = Incident.objects.filter(
            start_time__lte=incident.start_time,
        ).filter(
            Q(resolved=False) | Q(end_time__gt=incident.start_time)
        ).exclude(pk=incident.pk).count()
    else:
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
        "hour_of_day": start_time_local.hour,
        "day_of_week": start_time_local.weekday(),
        "month": start_time_local.month,
        "is_weekend": int(start_time_local.weekday() >= 5),
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
        tfidf_matrix = vectorizer.transform([feature_text])
        tfidf_cols = [f"tfidf_{name}" for name in vectorizer.get_feature_names_out()]
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(), columns=tfidf_cols, index=df.index
        )
        df = pd.concat([df, tfidf_df], axis=1)

    expected_cols = list(getattr(model, "feature_names_in_", df.columns))
    df = df.reindex(columns=expected_cols, fill_value=0)

    # Classifier predicts a block offset; confidence is the class probability.
    probs = model.predict_proba(df)[0]
    baseline_probs, baseline_evidence = _historical_baseline_probs(
        historical_baselines,
        station,
        feature_text,
        model.classes_,
    )
    probs = _blend_prediction_probs(
        probs,
        baseline_probs,
        baseline_evidence,
        historical_baselines.get("config") if historical_baselines else None,
    )
    # One-sided "fixed by" bound. classes_ is sorted ascending, so the
    # cumulative sum of probs is P(resolved by the end of that block). We
    # take the earliest block whose cumulative probability reaches the
    # coverage target and tell the user it'll be fixed *by* then. When the
    # model is confident this collapses to a single block; when it's unsure
    # the bound widens automatically instead of pretending to precision.
    order = np.argsort(model.classes_)
    sorted_offsets = np.asarray(model.classes_)[order]
    cumulative = np.cumsum(np.asarray(probs)[order])
    target_pos = int(np.searchsorted(cumulative, PREDICTION_COVERAGE_TARGET))
    if target_pos >= len(sorted_offsets):
        target_pos = len(sorted_offsets) - 1
    pred_offset = int(sorted_offsets[target_pos])
    confidence = float(cumulative[target_pos])
    confidence = max(0.05, min(0.95, confidence))

    # Decode offset → the end of that block. block_start_end returns naive
    # datetimes in local (London) time, so attach the project's configured
    # timezone before doing arithmetic with the (UTC-stored) start_time.
    # We land one second inside the block so block_index() of the predicted
    # end resolves to this block, not the next one (end boundaries are
    # exclusive).
    end_idx = start_idx + pred_offset
    _, block_end_naive = block_start_end(end_idx)
    local_tz = tz_module.get_default_timezone()
    block_end_dt = block_end_naive.replace(tzinfo=local_tz)
    end_time = block_end_dt - timedelta(seconds=1)

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
predict_duration.cache_clear = _clear_model_cache
