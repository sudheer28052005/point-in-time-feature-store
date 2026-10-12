"""CSV -> SQL Server bulk loader.

Uses pyodbc fast_executemany (parameter arrays, not row-by-row INSERTs) in batches.
Constraints stay ENABLED during the load; the whole load is ONE transaction, so a
failure leaves the tables untouched. Cross-table rules are verified afterwards by
src/validation.py.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from .config import Settings
from .database import get_engine

log = logging.getLogger(__name__)

TABLES = {
    # table: (csv name, columns in INSERT order, datetime columns to parse)
    "dbo.users": ("users.csv",
                  ["user_id", "signup_at", "country", "acquisition_channel", "customer_segment",
                   "date_of_birth", "is_marketing_opted_in"], ["signup_at", "date_of_birth"]),
    "dbo.orders": ("orders.csv",
                   ["order_id", "user_id", "order_ts", "order_status", "gross_amount", "discount_amount",
                    "payment_method", "refunded_at"], ["order_ts", "refunded_at"]),
    "dbo.raw_web_events": ("events.csv",
                           ["event_id", "user_id", "event_ts", "source_session_id", "event_type", "page_url",
                            "device_type", "referrer_type"], ["event_ts"]),
}


class LoadError(RuntimeError):
    pass


def _rows(df: pd.DataFrame) -> list[tuple]:
    """DataFrame -> list of tuples with NaN/NaT converted to None (SQL NULL)."""
    obj = df.astype(object).where(df.notna(), None)
    return list(obj.itertuples(index=False, name=None))


def _count(conn, table: str) -> int:
    return int(conn.execute(text(f"SELECT COUNT_BIG(*) FROM {table}")).scalar_one())  # whitelisted names


def load_csv_dir(settings: Settings, data_dir: Path, truncate: bool = False) -> dict[str, int]:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise LoadError(f"{manifest_path} not found - run scripts/generate_data.py first.")
    manifest = json.loads(manifest_path.read_text())
    start = time.perf_counter()
    loaded: dict[str, int] = {}

    with get_engine(settings).begin() as conn:  # one transaction for everything
        existing = {t: _count(conn, t) for t in TABLES}
        if any(existing.values()):
            if not truncate:
                raise LoadError(f"Tables are not empty {existing}. Re-run with --truncate to replace the "
                                "data (deletes all rows from the three source tables; the database is kept).")
            log.warning("--truncate given: deleting existing rows %s", existing)
            for t in reversed(list(TABLES)):  # children first because of foreign keys
                conn.execute(text(f"DELETE FROM {t}"))
        for table, (csv_name, cols, dates) in TABLES.items():
            sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
            total, t0 = 0, time.perf_counter()
            for chunk in pd.read_csv(data_dir / csv_name, usecols=cols, parse_dates=dates,
                                     chunksize=settings.batch_size, keep_default_na=True):
                conn.exec_driver_sql(sql, _rows(chunk[cols]))
                total += len(chunk)
                if total % (settings.batch_size * 10) < settings.batch_size:
                    log.info("%s: %s rows loaded", table, f"{total:,}")
            log.info("%s: %s rows in %.1fs", table, f"{total:,}", time.perf_counter() - t0)
            expected = manifest["rows"][csv_name.split(".")[0]]
            if total != expected:
                raise LoadError(f"{table}: loaded {total} rows but manifest says {expected}")
            loaded[table] = total
        conn.execute(
            text("""INSERT INTO analytics.data_load_log
                    (random_seed, num_users, num_orders, num_events, fingerprint, load_seconds, notes)
                    VALUES (:seed, :u, :o, :e, :fp, :secs, :notes)"""),
            {"seed": manifest["seed"], "u": loaded["dbo.users"], "o": loaded["dbo.orders"],
             "e": loaded["dbo.raw_web_events"], "fp": manifest["fingerprint"],
             "secs": round(time.perf_counter() - start, 2), "notes": "truncate" if truncate else "initial"},
        )
    log.info("load committed in %.1fs", time.perf_counter() - start)
    return loaded
