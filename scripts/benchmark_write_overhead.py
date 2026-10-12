"""Measure the WRITE cost of the candidate time indexes (the other half of every index decision).

Copies the real event/order rows into scratch tables (schema analytics, prefix zz_wo_) that have either
the base layout (PK nonclustered + clustered user/time) or base + the candidate nonclustered index,
inserting in clustered-key order, TRUNCATE between reps. Scratch tables are dropped at the end.
Results -> reports/write_overhead.json
"""
from __future__ import annotations

import json
import logging
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent))

from src import config as C  # noqa: E402
from src.benchmarking import Measurer  # noqa: E402
from src.database import build_odbc_connection_string  # noqa: E402

log = logging.getLogger("write_overhead")
REPS = 3

CASES = {
    "raw_web_events": dict(
        src="dbo.raw_web_events", cols="event_id, user_id, event_ts, source_session_id, event_type, device_type, referrer_type, created_at",
        pk="event_id", cix="user_id, event_ts, event_id", extra=("IX_ts", "event_ts", "event_type")),
    "orders": dict(
        src="dbo.orders", cols="order_id, user_id, order_ts, order_status, gross_amount, discount_amount, payment_method, refunded_at, created_at",
        pk="order_id", cix="user_id, order_ts, order_id", extra=("IX_ts", "order_ts", "order_status, refunded_at, net_amount")),
    "session_events": dict(
        src="analytics.session_events", cols="event_id, user_id, event_ts, event_type, device_type, session_seq, reconstructed_session_id",
        pk="event_id", cix="user_id, event_ts, event_id", extra=("IX_ts", "event_ts", "event_type, reconstructed_session_id")),
}


def main() -> int:
    C.configure_logging()
    m = Measurer(build_odbc_connection_string(C.get_settings()))
    out = {}
    try:
        for name, c in CASES.items():
            out[name] = {}
            for variant in ("base", "base_plus_time_nci"):
                t = f"analytics.zz_wo_{name}_{variant}"
                m.cur.execute(f"DROP TABLE IF EXISTS {t}")
                m.cur.execute(f"SELECT {c['cols']} {', net_amount' if name == 'orders' else ''} INTO {t} FROM {c['src']} WHERE 1 = 0".replace(", net_amount", ""))
                if name == "orders":
                    m.cur.execute(f"ALTER TABLE {t} ADD net_amount AS (gross_amount - discount_amount) PERSISTED")
                m.cur.execute(f"ALTER TABLE {t} ADD CONSTRAINT PK_zz_{name}_{variant} PRIMARY KEY NONCLUSTERED ({c['pk']})")
                m.cur.execute(f"CREATE UNIQUE CLUSTERED INDEX CIX_zz_{name}_{variant} ON {t} ({c['cix']})")
                if variant != "base":
                    ix, key, inc = c["extra"]
                    m.cur.execute(f"CREATE NONCLUSTERED INDEX {ix}_{name} ON {t} ({key}) INCLUDE ({inc})")
                times, cpus, reads = [], [], []
                for _ in range(REPS):
                    m.cur.execute(f"TRUNCATE TABLE {t}")
                    r = m.run(f"INSERT INTO {t} ({c['cols']}) SELECT {c['cols']} FROM {c['src']} ORDER BY {c['cix']}", [], cold=False)
                    times.append(r["elapsed_ms"]); cpus.append(r["cpu_ms"]); reads.append(r["logical_reads"])
                n = m.cur.execute(f"SELECT COUNT_BIG(*) FROM {t}").fetchone()[0]
                out[name][variant] = dict(rows=int(n), insert_ms_median=statistics.median(times), insert_ms_all=times,
                                          cpu_ms_median=statistics.median(cpus), logical_reads_median=statistics.median(reads))
                log.info("%-15s %-20s rows=%9s insert median %8.0f ms (runs %s) cpu %6.0f ms", name, variant, f"{n:,}",
                         statistics.median(times), [round(x) for x in times], statistics.median(cpus))
                m.cur.execute(f"DROP TABLE IF EXISTS {t}")
            b, e = out[name]["base"]["insert_ms_median"], out[name]["base_plus_time_nci"]["insert_ms_median"]
            out[name]["overhead_pct"] = round(100 * (e - b) / b, 1)
    finally:
        for name in CASES:
            for variant in ("base", "base_plus_time_nci"):
                m.cur.execute(f"DROP TABLE IF EXISTS analytics.zz_wo_{name}_{variant}")
        m.close()
    (C.PROJECT_ROOT / "reports").mkdir(exist_ok=True)
    (C.PROJECT_ROOT / "reports" / "write_overhead.json").write_text(json.dumps(out, indent=2))
    for name, v in out.items():
        log.info("%s: extra time index adds %+.1f%% to a bulk insert", name, v["overhead_pct"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
