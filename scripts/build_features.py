"""Build every derived analytics object inside SQL Server (all heavy work is T-SQL).

    python scripts/build_features.py                  # sessions + monthly snapshots + cohorts + checks
    python scripts/build_features.py --first 2024-01-01 --last 2024-06-01

Snapshot grid: first-of-month cutoffs. Snapshots whose 30-day outcome window is not
fully observed (cutoff + 30d > observation end) get features but NO labels (scoring snapshots).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent))

from sqlalchemy import text  # noqa: E402

from src import config as C  # noqa: E402
from src.database import DatabaseError, check_connection, get_engine  # noqa: E402
from src.validation import run_data_quality_checks  # noqa: E402

log = logging.getLogger("build_features")


def _date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--first", type=_date, default=datetime(2023, 4, 1), help="first cutoff (UTC date)")
    p.add_argument("--last", type=_date, default=C.OBS_END, help="last cutoff (UTC date)")
    p.add_argument("--observation-end", type=_date, default=C.OBS_END)
    p.add_argument("--inactivity-minutes", type=int, default=30)
    p.add_argument("--skip-sessions", action="store_true")
    args = p.parse_args(argv)
    C.configure_logging()
    try:
        log.info("connected: %s", check_connection())
    except DatabaseError as exc:
        log.error("%s", exc)
        return 2

    eng = get_engine()
    t0 = time.perf_counter()
    with eng.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        if not args.skip_sessions:
            conn.execute(text("EXEC analytics.usp_build_sessions @inactivity_minutes = :m"), {"m": args.inactivity_minutes})
            log.info("sessions built (%.1fs)", time.perf_counter() - t0)
        cutoff = args.first
        while cutoff <= args.last:
            t1 = time.perf_counter()
            conn.execute(text("EXEC analytics.usp_build_feature_snapshot @T0 = :t, @observation_end = :e"),
                         {"t": cutoff, "e": args.observation_end})
            log.info("snapshot %s built (%.1fs)", cutoff.date(), time.perf_counter() - t1)
            cutoff = datetime(cutoff.year + (cutoff.month // 12), cutoff.month % 12 + 1, 1)
        for definition in ("purchase", "web"):
            conn.execute(text("EXEC analytics.usp_build_cohort_retention @observation_end = :e, @activity_definition = :d"),
                         {"e": args.observation_end, "d": definition})
            log.info("cohort retention built (%s)", definition)
    log.info("all derived objects built in %.1fs", time.perf_counter() - t0)

    checks = run_data_quality_checks(derived=True)
    bad = checks[checks.violations != 0]
    print(checks.to_string(index=False))
    if not bad.empty:
        log.error("%d derived-data check(s) FAILED", len(bad))
        return 1
    log.info("all %d derived-data checks passed", len(checks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
