"""Sessionization edge cases (hand-built data in the throwaway database)."""
import pytest
from sqlalchemy.exc import DBAPIError

pytestmark = pytest.mark.db


def seqs(scn):
    return scn.session_events().session_seq.tolist()


def test_gap_of_exactly_30_minutes_stays_in_same_session(scn):
    scn.add_user(1)
    scn.add_event(1, "2024-03-01 10:00:00.000")
    scn.add_event(1, "2024-03-01 10:30:00.000")           # gap == 30:00.000 -> same session
    scn.build_sessions()
    assert seqs(scn) == [1, 1]


def test_gap_one_millisecond_over_30_minutes_starts_new_session(scn):
    scn.add_user(1)
    scn.add_event(1, "2024-03-01 10:00:00.000")
    scn.add_event(1, "2024-03-01 10:30:00.001")           # gap == 30:00.001 -> new session
    scn.build_sessions()
    assert seqs(scn) == [1, 2]


def test_gap_one_millisecond_under_30_minutes_stays_in_same_session(scn):
    scn.add_user(1)
    scn.add_event(1, "2024-03-01 10:00:00.000")
    scn.add_event(1, "2024-03-01 10:29:59.999")
    scn.build_sessions()
    assert seqs(scn) == [1, 1]


def test_chain_of_29_minute_gaps_is_one_long_session(scn):
    """Inactivity is measured between consecutive events, not from the session start."""
    scn.add_user(1)
    for i in range(5):
        scn.add_event(1, f"2024-03-01 {10 + (i * 29) // 60:02d}:{(i * 29) % 60:02d}:00.000")
    scn.build_sessions()
    s = scn.sessions()
    assert len(s) == 1 and s.event_count[0] == 5 and s.duration_seconds[0] == 4 * 29 * 60


def test_session_boundaries_and_aggregates(scn):
    scn.add_user(1)
    for ts, typ in [("2024-03-01 09:00:00", "page_view"), ("2024-03-01 09:01:00", "product_view"),
                    ("2024-03-01 09:02:00", "add_to_cart"), ("2024-03-01 09:03:00", "checkout_start"),
                    ("2024-03-01 09:04:00", "purchase"),
                    ("2024-03-01 15:00:00", "page_view")]:
        scn.add_event(1, ts, typ)
    scn.build_sessions()
    s = scn.sessions()
    assert len(s) == 2
    a, b = s.iloc[0], s.iloc[1]
    assert (a.event_count, a.page_view_count, a.product_view_count, a.add_to_cart_count,
            a.checkout_started_count, a.purchase_event_count) == (5, 1, 1, 1, 1, 1)
    assert a.duration_seconds == 240
    assert (b.event_count, b.duration_seconds, b.page_view_count) == (1, 0, 1)   # single-event session
    assert (a.reconstructed_session_id, b.reconstructed_session_id) == (1_000_001, 1_000_002)


def test_simultaneous_events_are_ordered_by_event_id(scn):
    scn.add_user(1)
    scn.add_event(1, "2024-03-01 10:00:00.000", device="desktop", event_id=5)   # inserted first...
    scn.add_event(1, "2024-03-01 10:00:00.000", device="mobile", event_id=3)    # ...but lower id
    scn.build_sessions()
    se = scn.session_events()
    assert se.event_id.tolist() == [3, 5] and seqs(scn) == [1, 1]
    s = scn.sessions()
    assert s.device_type[0] == "mobile"          # device of the FIRST event (lowest event_id on a tie)
    assert s.distinct_device_count[0] == 2


def test_simultaneous_events_after_a_gap_all_belong_to_new_session(scn):
    scn.add_user(1)
    scn.add_event(1, "2024-03-01 10:00:00.000")
    for _ in range(3):
        scn.add_event(1, "2024-03-01 12:00:00.000")
    scn.build_sessions()
    assert seqs(scn) == [1, 2, 2, 2]


def test_source_session_id_is_ignored(scn):
    scn.add_user(1)
    same_src = "11111111-1111-1111-1111-111111111111"
    scn.add_event(1, "2024-03-01 10:00:00", source_session_id=same_src)
    scn.add_event(1, "2024-03-01 14:00:00", source_session_id=same_src)           # same source id, 4h apart
    scn.add_event(1, "2024-03-01 14:01:00")                                       # different id, 1 min later
    scn.build_sessions()
    assert seqs(scn) == [1, 2, 2]


def test_users_are_sessionized_independently(scn):
    scn.add_users([1, 2])
    scn.add_event(1, "2024-03-01 10:00:00")
    scn.add_event(2, "2024-03-01 10:05:00")
    scn.add_event(1, "2024-03-01 10:10:00")        # user 2's event in between does not split user 1
    scn.build_sessions()
    se = scn.session_events()
    assert se[se.user_id == 1].session_seq.tolist() == [1, 1]
    assert se[se.user_id == 2].session_seq.tolist() == [1]
    assert se.reconstructed_session_id.is_unique is False and scn.sessions().reconstructed_session_id.is_unique


def test_rebuild_is_idempotent_and_deterministic(scn):
    scn.add_user(1)
    for ts in ["2024-03-01 10:00:00", "2024-03-01 10:00:00", "2024-03-01 12:00:00"]:
        scn.add_event(1, ts)
    scn.build_sessions()
    first = scn.session_events()
    scn.build_sessions()
    assert scn.session_events().equals(first)


def test_threshold_is_configurable_and_consistent(scn):
    scn.add_user(1)
    scn.add_event(1, "2024-03-01 10:00:00")
    scn.add_event(1, "2024-03-01 10:20:00")
    scn.build_sessions(minutes=30)
    assert seqs(scn) == [1, 1]
    scn.build_sessions(minutes=10)
    assert seqs(scn) == [1, 2]


def test_invalid_threshold_is_rejected(scn):
    with pytest.raises(DBAPIError, match="positive integer"):
        scn.build_sessions(minutes=0)


def test_user_without_events_has_no_sessions(scn):
    scn.add_users([1, 2])
    scn.add_event(1, "2024-03-01 10:00:00")
    scn.build_sessions()
    assert scn.sessions().user_id.tolist() == [1]
