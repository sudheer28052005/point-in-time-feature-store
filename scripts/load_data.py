"""Bulk-load the generated CSVs into SQL Server, then run the data-quality checks.

    python scripts/load_data.py                 # fails if tables already contain rows
    python scripts/load_data.py --truncate      # explicit opt-in: replace existing rows
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent))

from src.config import configure_logging, get_settings  # noqa: E402
from src.database import DatabaseError, check_connection  # noqa: E402
from src.loading import LoadError, load_csv_dir  # noqa: E402
from src.validation import run_data_quality_checks, table_row_counts  # noqa: E402

log = logging.getLogger("load_data")


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=s.data_dir)
    p.add_argument("--batch-size", type=int, default=s.batch_size)
    p.add_argument("--truncate", action="store_true",
                   help="delete all existing rows from users/orders/raw_web_events before loading")
    args = p.parse_args(argv)
    configure_logging()
    s = dataclasses.replace(s, batch_size=args.batch_size)
    try:
        log.info("connected: %s", check_connection(s))
        load_csv_dir(s, args.data_dir, truncate=args.truncate)
    except (DatabaseError, LoadError) as exc:
        log.error("%s", exc)
        return 2
    log.info("row counts: %s", table_row_counts(s))
    checks = run_data_quality_checks(s)
    print(checks.to_string(index=False))
    bad = checks[checks.violations != 0]
    if not bad.empty:
        log.error("%d data-quality check(s) FAILED", len(bad))
        return 1
    log.info("all %d data-quality checks passed", len(checks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
