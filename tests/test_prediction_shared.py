from datetime import datetime, timedelta, timezone

from incidents import prediction_shared as shared

T0 = datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc)


def fault(started_hours_ago, lasted_hours=None, offset=None,
          category="faulty_lift", key="lift a"):
    start = T0 - timedelta(hours=started_hours_ago)
    end = None if lasted_hours is None else start + timedelta(hours=lasted_hours)
    return (start, end, offset, category, key)


def test_stated_end_rule():
    assert shared.has_stated_end("No step-free access until Sunday 22 November")
    assert shared.has_stated_end("Lift works", information=True)
    assert not shared.has_stated_end("Not available due to planned maintenance.")
    assert not shared.has_stated_end(None)


def test_fault_key_ignores_phone_number_case_and_spacing():
    a = "Faulty lift.  Call us on 0343 222 1234 for help"
    b = "faulty lift. call us on for help"
    assert shared.fault_key(a) == shared.fault_key(b)


def test_history_with_no_past_faults():
    features = shared.history_features([], T0, "faulty_lift", "lift a")
    assert set(features) == set(shared.HISTORY_FEATURE_NAMES)
    assert features["st_count"] == 0
    assert features["st_last_off"] == -1.0
    assert features["st_open_now"] == 0


def test_history_only_uses_what_was_known_at_the_start():
    past = [
        fault(100, lasted_hours=10, offset=2),   # ended 90h ago
        fault(48, lasted_hours=24, offset=5),    # ended 24h ago
        fault(20, lasted_hours=30, offset=6),    # still open at T0
        fault(5),                                # unresolved
        fault(-1, lasted_hours=1, offset=0),     # starts after T0
    ]
    features = shared.history_features(past, T0, "faulty_lift", "lift a")
    assert features["st_count"] == 2
    assert features["st_mean_off"] == 3.5
    assert features["st_last_off"] == 5.0
    assert features["st_days_since_last_end"] == 1.0
    assert features["st_open_now"] == 2
    assert features["st_started_7d"] == 4
    assert features["st_started_30d"] == 4


def test_history_groups_by_category_and_same_fault():
    past = [
        fault(100, 10, 2, category="staff_issue", key="staff"),
        fault(80, 10, 9, category="faulty_lift", key="lift b"),
        fault(60, 10, 1, category="faulty_lift", key="lift a"),
    ]
    features = shared.history_features(past, T0, "faulty_lift", "lift a")
    assert features["st_count"] == 3
    assert features["sc_count"] == 2
    assert features["sc_long_share"] == 0.5
    assert features["sf_count"] == 1
    assert features["sf_last_off"] == 1.0
    assert features["sf_started_7d"] == 1


def test_history_statistics_use_last_fifty_capped_offsets():
    past = [fault(1000 - i, 1, 200) for i in range(10)] + [
        fault(900 - i, 1, i % 10) for i in range(50)
    ]
    features = shared.history_features(past, T0, "faulty_lift", "lift a")
    assert features["st_count"] == 60
    assert features["st_max_off"] == 9.0   # the 200s fell out of the window
    assert features["st_median_off"] == 4.5
    assert features["st_p90_off"] == 8.0
    capped = shared.history_features(past[:10], T0, "faulty_lift", "lift a")
    assert capped["st_max_off"] == 60.0


def test_tail_distribution_counts_and_overflow():
    table = shared.build_tail_distribution(
        {"faulty_lift": [3, 8, 8, 12, 120, 121, 500]}
    )
    entry = table["by_category"]["faulty_lift"]
    assert entry["total"] == 6          # 3 is not a tail fault
    assert entry["counts"][0] == 2      # offset 8
    assert entry["counts"][4] == 1      # offset 12
    assert entry["counts"][-1] == 1     # offset 120
    assert sum(entry["counts"]) == 4    # 121 and 500 overflow
    assert table["all"]["total"] == 6


def test_cdf_without_tail_stops_before_the_open_ended_class():
    probs = [0.5, 0.2, 0.1, 0, 0, 0, 0, 0, 0.2]
    cdf = shared.extended_cdf(probs)
    assert len(cdf) == 8
    assert shared.bound_offset(cdf, 0.75) == 2
    assert shared.bound_offset(cdf, 0.95) is None


def test_cdf_spreads_tail_probability_along_the_table():
    table = shared.build_tail_distribution({"faulty_lift": [8] * 20 + [12] * 20})
    probs = [0.5, 0.2, 0.1, 0, 0, 0, 0, 0, 0.2]
    cdf = shared.extended_cdf(probs, table, "faulty_lift")
    assert len(cdf) == 121
    assert abs(cdf[8] - 0.9) < 1e-9
    assert abs(cdf[12] - 1.0) < 1e-9
    assert shared.bound_offset(cdf, 0.95) == 12


def test_overflow_mass_is_never_assigned_to_a_date():
    table = shared.build_tail_distribution({"faulty_lift": [8] * 20 + [500] * 20})
    cdf = shared.extended_cdf([0, 0, 0, 0, 0, 0, 0, 0, 1.0], table, "faulty_lift")
    assert abs(cdf[-1] - 0.5) < 1e-9
    assert shared.bound_offset(cdf, 0.95) is None


def test_sparse_category_falls_back_to_all_categories():
    table = shared.build_tail_distribution(
        {"staff_issue": [9] * 5, "faulty_lift": [10] * 40}
    )
    cdf = shared.extended_cdf([0] * 8 + [1.0], table, "staff_issue")
    assert shared.bound_offset(cdf, 0.95) == 10
    nothing = shared.build_tail_distribution({"staff_issue": [9] * 5})
    assert len(shared.extended_cdf([0] * 8 + [1.0], nothing, "staff_issue")) == 8


def test_round_up_to_service_day():
    assert shared.round_up_to_service_day(start_block_index=1, offset=3) == 3
    # daytime (1) + 8 blocks is daytime two days on: push to that day's overnight
    assert shared.round_up_to_service_day(start_block_index=1, offset=8) == 10
    assert shared.round_up_to_service_day(start_block_index=1, offset=10) == 10
    assert shared.round_up_to_service_day(start_block_index=3, offset=9) == 12
