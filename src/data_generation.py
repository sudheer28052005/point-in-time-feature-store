"""Deterministic synthetic data generator (NumPy for volume, Faker for the catalog).

Everything is derived from one seed, so the same (seed, sizes, window) always yields
byte-identical DataFrames (see `dataset_fingerprint`). Nothing here touches the database.

Behavioural model (kept deliberately simple and explainable)
* Signups: spread over ~2 years with a mild growth trend and seasonality.
* Each user has a latent order rate (heavy-tailed gamma, scaled by segment), a
  "buyer" flag (a share of users never purchase) and a lifetime (exponential; 30% are
  loyal long-lifetime users). Activity stops after the lifetime -> real inactive users.
* Orders: Poisson count over the active window, dates drawn from a day-weight curve
  (growth, Nov/Dec peak, weekend lift), evening-heavy hour-of-day.
  Status mix ~88% completed / 7% cancelled / 5% refunded; refunds land days later.
* Web: ~60% of orders have a tracked "purchase session" (browse -> cart -> checkout ->
  purchase) ending at order_ts, plus browse-only sessions for everyone. Inside a session events are seconds
  apart; ~4% of sessions contain an idle gap of 31-120 minutes while keeping the same
  source_session_id, and ~25% of consecutive sessions REUSE the source id. That is
  why analytics must reconstruct sessions from gaps instead of trusting source ids.
* Volumes are targets, not exact: orders ~ Poisson, events ~ geometric session sizes.
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from faker import Faker

from . import config as C

log = logging.getLogger(__name__)

MS_DAY = 86_400_000
_COUNTRIES = ["US", "GB", "DE", "FR", "CA", "AU", "IN", "BR", "ES", "NL"]
_COUNTRY_P = [0.32, 0.12, 0.09, 0.08, 0.08, 0.06, 0.10, 0.07, 0.05, 0.03]
_CHANNEL_P = [0.30, 0.22, 0.18, 0.12, 0.10, 0.08]
_SEGMENT_P = [0.30, 0.45, 0.15, 0.10]
_SEG_RATE = np.array([0.15, 0.25, 0.45, 0.60])        # mean orders / active month
_SEG_BUYER_P = np.array([0.55, 0.65, 0.75, 0.80])     # share who ever purchase
_SEG_MU = np.array([3.3, 3.9, 4.6, 5.2])              # log-mean of gross order value
_CHANNEL_LIFE = np.array([1.0, 0.7, 0.9, 1.2, 1.2, 0.7])  # lifetime multiplier by channel
_MONTH_F = np.array([0.95, 0.90, 0.95, 1.0, 1.0, 0.95, 0.90, 0.95, 1.0, 1.1, 1.35, 1.5])
_HOUR_W = np.array([1, 1, 1, 1, 1, 2, 3, 5, 6, 6, 6, 7, 8, 7, 6, 6, 7, 8, 10, 12, 12, 10, 6, 3], float)
_HOUR_P = _HOUR_W / _HOUR_W.sum()


@dataclass(frozen=True)
class GenerationConfig:
    num_users: int = 50_000
    num_orders: int = 200_000
    num_events: int = 1_000_000
    seed: int = 42
    obs_start: datetime = C.OBS_START
    signup_end: datetime = C.SIGNUP_END
    obs_end: datetime = C.OBS_END


@dataclass
class Dataset:
    users: pd.DataFrame
    orders: pd.DataFrame
    events: pd.DataFrame


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _to_dt(ms: np.ndarray, origin: datetime) -> np.ndarray:
    return np.datetime64(origin, "ms") + ms.astype("timedelta64[ms]")


class _TimeSampler:
    """Draws timestamps (ms since obs_start) inside per-row [lo, hi) windows, weighted by day."""

    def __init__(self, cfg: GenerationConfig):
        n_days = (cfg.obs_end - cfg.obs_start).days + 2
        days = pd.date_range(cfg.obs_start, periods=n_days, freq="D")
        w = _MONTH_F[days.month - 1] * np.where(days.dayofweek >= 5, 1.15, 1.0)
        w = w * (1 + 0.6 * np.arange(n_days) / n_days)  # business growth
        self.cum = np.concatenate([[0.0], np.cumsum(w)])

    def sample(self, rng: np.random.Generator, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        n = len(lo)
        d_lo, d_hi = lo // MS_DAY, hi // MS_DAY + 1
        u = self.cum[d_lo] + rng.random(n) * (self.cum[d_hi] - self.cum[d_lo])
        day = np.clip(np.searchsorted(self.cum, u, side="right") - 1, d_lo, d_hi - 1)
        tod = rng.choice(24, size=n, p=_HOUR_P) * 3_600_000 + rng.integers(0, 3_600_000, n)
        ts = day * MS_DAY + tod
        bad = (ts < lo) | (ts >= hi)  # day-of-window edges: fall back to uniform inside window
        ts[bad] = lo[bad] + (rng.random(int(bad.sum())) * (hi[bad] - lo[bad])).astype(np.int64)
        return ts


def _choice(rng, options, p, size):
    return np.asarray(options, dtype=object)[rng.choice(len(options), size=size, p=p)]


# --------------------------------------------------------------------------- #
# generator
# --------------------------------------------------------------------------- #
def generate_dataset(cfg: GenerationConfig) -> Dataset:
    if min(cfg.num_users, cfg.num_orders, cfg.num_events) < 1:
        raise ValueError("num_users, num_orders and num_events must all be >= 1")
    rng = np.random.default_rng(cfg.seed)
    sampler = _TimeSampler(cfg)
    obs_len = int((cfg.obs_end - cfg.obs_start).total_seconds() * 1000)
    signup_len = int((cfg.signup_end - cfg.obs_start).total_seconds() * 1000)
    n = cfg.num_users

    # ---------------- users ----------------
    signup_ms = np.sort(sampler.sample(rng, np.zeros(n, np.int64), np.full(n, signup_len, np.int64)))
    country = _choice(rng, _COUNTRIES, _COUNTRY_P, n)
    chan_idx = rng.choice(len(C.ACQUISITION_CHANNELS), size=n, p=_CHANNEL_P)
    seg_idx = rng.choice(len(C.CUSTOMER_SEGMENTS), size=n, p=_SEGMENT_P)
    signup_dt = _to_dt(signup_ms, cfg.obs_start)
    age_days = (rng.normal(38, 12, n).clip(18, 75) * 365.25).astype(int)
    dob = signup_dt.astype("datetime64[D]") - age_days.astype("timedelta64[D]")
    dob = np.where(rng.random(n) < 0.12, np.datetime64("NaT"), dob)  # 12% undisclosed
    users = pd.DataFrame({
        "user_id": np.arange(1, n + 1, dtype=np.int64),
        "signup_at": signup_dt,
        "country": country,
        "acquisition_channel": np.asarray(C.ACQUISITION_CHANNELS, dtype=object)[chan_idx],
        "customer_segment": np.asarray(C.CUSTOMER_SEGMENTS, dtype=object)[seg_idx],
        "date_of_birth": dob.astype("datetime64[ms]"),
        "is_marketing_opted_in": rng.random(n) < 0.55,
    })

    # ---------------- latent behaviour ----------------
    rate = rng.gamma(0.8, _SEG_RATE[seg_idx] / 0.8)
    buyer = rng.random(n) < _SEG_BUYER_P[seg_idx]
    loyal = rng.random(n) < 0.30
    life_ms = (rng.exponential(np.where(loyal, 30.0, 7.0)) * _CHANNEL_LIFE[chan_idx] * 30.44 * MS_DAY).astype(np.int64)
    active_end = np.minimum(np.maximum(signup_ms + life_ms, signup_ms + MS_DAY), obs_len)
    months = (active_end - signup_ms) / (30.44 * MS_DAY)

    # ---------------- orders ----------------
    lam = rate * buyer * months
    counts = rng.poisson(lam * (cfg.num_orders / lam.sum()))
    owner = np.repeat(np.arange(n), counts)
    m = len(owner)
    o_ts = sampler.sample(rng, signup_ms[owner], active_end[owner])
    order = np.lexsort((o_ts, owner))  # clustered-index order: (user, ts)
    owner, o_ts = owner[order], o_ts[order]
    status = _choice(rng, C.ORDER_STATUSES, [0.88, 0.07, 0.05], m)
    refund_ms = o_ts + 3_600_000 + rng.exponential(6 * MS_DAY, m).astype(np.int64)
    refund_ms = np.maximum(np.minimum(refund_ms, obs_len - 1), o_ts)
    gross = np.maximum(np.exp(rng.normal(_SEG_MU[seg_idx[owner]], 0.55, m)) * rng.lognormal(0, 0.3, n)[owner], 1.0)
    gross = np.round(gross, 2)
    pct = np.where(rng.random(m) < 0.35, rng.choice([5, 10, 15, 20, 25, 30], m), 0)
    orders = pd.DataFrame({
        "order_id": np.arange(1, m + 1, dtype=np.int64),
        "user_id": owner.astype(np.int64) + 1,
        "order_ts": _to_dt(o_ts, cfg.obs_start),
        "order_status": status,
        "gross_amount": gross,
        "discount_amount": np.round(gross * pct / 100, 2),
        "payment_method": _choice(rng, C.PAYMENT_METHODS, [0.55, 0.20, 0.08, 0.04, 0.13], m),
        "refunded_at": np.where(status == "refunded", _to_dt(refund_ms, cfg.obs_start), np.datetime64("NaT")),
    })
    orders["refunded_at"] = orders["refunded_at"].astype("datetime64[ms]")

    events = _generate_events(cfg, rng, sampler, obs_len, signup_ms, active_end, rate, buyer, months,
                              chan_idx, owner, o_ts, status)
    log.info("generated users=%d orders=%d events=%d", len(users), len(orders), len(events))
    return Dataset(users, orders, events)


def _generate_events(cfg, rng, sampler, obs_len, signup_ms, active_end, rate, buyer, months,
                     chan_idx, o_owner, o_ts, o_status) -> pd.DataFrame:
    n = len(signup_ms)
    T = {t: i for i, t in enumerate(C.EVENT_TYPES)}
    PV, PRV, CART, CHK, PUR = (T[t] for t in C.EVENT_TYPES)

    # ---- purchase sessions: ~60% of orders have a tracked session ending exactly at order_ts
    # (the rest model tracking gaps: app/phone/consent). Keeps most of the event budget for
    # browse sessions, so non-buyers also have web activity.
    tracked = rng.random(len(o_owner)) < 0.60
    o_owner, o_ts, o_status = o_owner[tracked], o_ts[tracked], o_status[tracked]
    m = len(o_owner)
    has_purchase = o_status != "cancelled"
    n_pre = rng.integers(0, 2, m)
    total = n_pre + 3 + has_purchase
    p_start = np.concatenate([[0], np.cumsum(total)[:-1]])
    p_sess = np.repeat(np.arange(m), total)
    pos = np.arange(total.sum()) - np.repeat(p_start, total)
    from_end = np.repeat(total, total) - 1 - pos
    hp = np.repeat(has_purchase, total)
    # position-from-end -> type; sessions without a purchase event are shifted by one
    k = from_end + (~hp)  # k: 0 purchase, 1 checkout_start, 2 add_to_cart, 3 product_view
    p_type = np.select([k == 0, k == 1, k == 2, k == 3], [PUR, CHK, CART, PRV], default=-1)
    pre = p_type == -1
    p_type[pre] = rng.choice([PV, PRV], size=int(pre.sum()), p=[0.4, 0.6])
    gap = rng.integers(15_000, 180_000, len(p_sess))
    gap[np.concatenate([p_start[1:] - 1, [len(gap) - 1]])] = 0  # last event of each session
    c = np.cumsum(gap)
    end_idx = np.repeat(p_start + total - 1, total)
    remaining = c[end_idx] - c + gap
    p_ts = np.repeat(o_ts, total) - remaining
    p_user = np.repeat(o_owner, total)
    p_ts = np.maximum(p_ts, signup_ms[p_user])  # never before signup; stays monotone

    # ---- browse sessions ----
    mean_size = 4.5
    # every user has a short signup-visit session (2-4 events, within 10 min of signup), plus
    # extra browse sessions allocated by a heavy-tailed engagement weight (buyers browse more)
    n_browse = max(int(round(max(cfg.num_events - len(p_sess) - 3 * n, 0) / mean_size)), 1)
    web_w = rng.gamma(0.9, 1.0, n) * (0.5 + buyer) * np.maximum(months, 0.05)
    extra_user = np.repeat(np.arange(n), rng.multinomial(n_browse, web_w / web_w.sum()))
    s_user = np.concatenate([np.arange(n), extra_user])
    ns = len(s_user)
    size = np.concatenate([rng.integers(2, 5, n), np.minimum(rng.geometric(1 / mean_size, len(extra_user)), 40)])
    s_start = np.concatenate([
        signup_ms + rng.integers(0, 600_000, n),
        sampler.sample(rng, signup_ms[extra_user], active_end[extra_user]),
    ])
    b_start = np.concatenate([[0], np.cumsum(size)[:-1]])
    b_sess = np.repeat(np.arange(ns), size)
    bpos = np.arange(size.sum()) - np.repeat(b_start, size)
    b_gap = np.minimum(3_000 + rng.exponential(70_000, len(b_sess)), 25 * 60_000).astype(np.int64)
    b_gap[bpos == 0] = 0
    idle = (rng.random(ns) < 0.04) & (size >= 3)  # idle gap inside one source session
    cut = 1 + (rng.random(ns) * (size - 1)).astype(int)
    hit = idle[b_sess] & (bpos == cut[b_sess])
    b_gap[hit] = rng.integers(31 * 60_000, 120 * 60_000, int(hit.sum()))
    cb = np.cumsum(b_gap)
    b_ts = s_start[b_sess] + cb - cb[np.repeat(b_start, size)]
    b_type = rng.choice([PV, PRV, CART, CHK], size=len(b_sess), p=[0.35, 0.50, 0.10, 0.05])
    b_type[bpos == 0] = PV
    b_user = s_user[b_sess]

    # ---- combine ----
    sess = np.concatenate([p_sess, m + b_sess])
    user = np.concatenate([p_user, b_user])
    ts = np.minimum(np.concatenate([p_ts, b_ts]), obs_len - 1)
    etype = np.concatenate([p_type, b_type])
    first = np.concatenate([pos == 0, bpos == 0])
    ntot = m + ns
    sess_user = np.concatenate([o_owner, s_user])
    sess_start = np.full(ntot, np.iinfo(np.int64).max)
    np.minimum.at(sess_start, sess, ts)

    # session-level attributes
    pref = rng.choice(3, size=n, p=[0.45, 0.45, 0.10])
    dev_s = np.where(rng.random(ntot) < 0.7, pref[sess_user], rng.choice(3, size=ntot, p=[0.45, 0.45, 0.10]))
    ref_s = rng.choice(len(C.REFERRER_TYPES), size=ntot, p=[0.30, 0.25, 0.15, 0.15, 0.10, 0.05])
    ref_null = rng.random(ntot) < 0.10

    # source_session_id: new id per session except ~25% that reuse the user's previous id
    ordr = np.lexsort((sess_start, sess_user))
    same_prev = np.concatenate([[False], sess_user[ordr][1:] == sess_user[ordr][:-1]])
    reuse = same_prev & (rng.random(ntot) < 0.25)
    grp_sorted = np.cumsum(~reuse) - 1
    grp = np.empty(ntot, np.int64)
    grp[ordr] = grp_sorted
    raw = rng.bytes(16 * (int(grp_sorted[-1]) + 1))
    uuids = np.array([str(uuid.UUID(bytes=raw[i:i + 16], version=4)) for i in range(0, len(raw), 16)], dtype=object)

    # page urls from a Faker-built (seeded) catalog
    fake = Faker()
    Faker.seed(cfg.seed)
    fake.seed_instance(cfg.seed)
    cats = np.array([fake.word() for _ in range(20)], dtype=object)
    prods = np.array([fake.slug() for _ in range(400)], dtype=object)
    ne = len(sess)
    url = np.empty(ne, dtype=object)
    url[:] = "/"
    mask = etype == PV
    url[mask] = "/category/" + cats[rng.integers(0, len(cats), int(mask.sum()))]
    mask = etype == PRV
    url[mask] = "/product/" + prods[rng.integers(0, len(prods), int(mask.sum()))]
    url[etype == CART] = "/cart"
    url[etype == CHK] = "/checkout"
    url[etype == PUR] = "/checkout/confirmation"
    url[rng.random(ne) < 0.03] = None

    ev_ref = np.where(first & ~ref_null[sess], np.asarray(C.REFERRER_TYPES, dtype=object)[ref_s[sess]], None)
    df = pd.DataFrame({
        "user_id": user.astype(np.int64) + 1,
        "event_ts": _to_dt(ts, cfg.obs_start),
        "source_session_id": uuids[grp[sess]],
        "event_type": np.asarray(C.EVENT_TYPES, dtype=object)[etype],
        "page_url": url,
        "device_type": np.asarray(C.DEVICE_TYPES, dtype=object)[dev_s[sess]],
        "referrer_type": ev_ref,
    })
    # Deterministic order = clustered index order. Stable sort keeps in-session order for equal ts.
    df = df.sort_values(["user_id", "event_ts"], kind="stable", ignore_index=True)
    df.insert(0, "event_id", np.arange(1, len(df) + 1, dtype=np.int64))
    return df


def dataset_fingerprint(ds: Dataset) -> str:
    """SHA-256 over every cell of every table; used to prove reproducibility."""
    h = hashlib.sha256()
    for name in ("users", "orders", "events"):
        df = getattr(ds, name)
        h.update(name.encode())
        h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return h.hexdigest()
