"""Generate the synthetic dataset to CSV files in data/ (git-ignored).

    python scripts/generate_data.py                       # volumes from .env (50k/200k/1M)
    python scripts/generate_data.py --users 2000 --orders 8000 --events 40000   # dev size
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import configure_logging, get_settings  # noqa: E402
from src.data_generation import GenerationConfig, dataset_fingerprint, generate_dataset  # noqa: E402

log = logging.getLogger("generate_data")
CSV_DATES = "%Y-%m-%d %H:%M:%S.%f"


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--users", type=int, default=s.num_users)
    p.add_argument("--orders", type=int, default=s.num_orders)
    p.add_argument("--events", type=int, default=s.num_events)
    p.add_argument("--seed", type=int, default=s.random_seed)
    p.add_argument("--out-dir", type=Path, default=s.data_dir)
    args = p.parse_args(argv)
    configure_logging()

    cfg = GenerationConfig(args.users, args.orders, args.events, args.seed)
    t0 = time.perf_counter()
    ds = generate_dataset(cfg)
    log.info("generation took %.1fs", time.perf_counter() - t0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in (("users", ds.users), ("orders", ds.orders), ("events", ds.events)):
        df.to_csv(args.out_dir / f"{name}.csv", index=False, date_format=CSV_DATES)
        log.info("wrote %s.csv (%s rows)", name, f"{len(df):,}")

    manifest = {
        "seed": cfg.seed,
        "requested": {"users": cfg.num_users, "orders": cfg.num_orders, "events": cfg.num_events},
        "rows": {"users": len(ds.users), "orders": len(ds.orders), "events": len(ds.events)},
        "observation_window_utc": [cfg.obs_start.isoformat(), cfg.obs_end.isoformat()],
        "fingerprint": dataset_fingerprint(ds),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("fingerprint %s", manifest["fingerprint"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
