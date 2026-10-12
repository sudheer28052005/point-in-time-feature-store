"""End-to-end orchestration: each step is the standalone script, run in order, stopping on the first failure.

    python scripts/run_pipeline.py                         # features -> train -> evaluate (data already loaded)
    python scripts/run_pipeline.py --steps train,evaluate
    python scripts/run_pipeline.py --all --truncate        # generate -> load (REPLACING rows) -> ... -> evaluate

Database creation/DDL is a separate, explicit step:  scripts/run_sql.ps1  (never done implicitly here).
Replacing loaded data needs --truncate; without it the loader refuses to touch non-empty tables.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.path.abspath(__file__)).parent.parent
STEPS = {
    "generate": ["scripts/generate_data.py"],
    "load": ["scripts/load_data.py"],
    "features": ["scripts/build_features.py"],
    "train": ["scripts/train_model.py"],
    "evaluate": ["scripts/evaluate_model.py"],
}
DEFAULT = ["features", "train", "evaluate"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", default=",".join(DEFAULT), help=f"comma list from {list(STEPS)}")
    p.add_argument("--all", action="store_true", help="run every step including generate and load")
    p.add_argument("--truncate", action="store_true", help="allow the load step to replace existing rows")
    p.add_argument("--population", default="active_90d", choices=["active_90d", "all_buyers"])
    args = p.parse_args(argv)
    steps = list(STEPS) if args.all else [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        p.error(f"unknown step(s): {unknown}")

    t_all = time.perf_counter()
    for s in steps:
        cmd = [sys.executable, *STEPS[s]]
        if s == "load" and args.truncate:
            cmd.append("--truncate")
        if s in ("train",):
            cmd += ["--population", args.population]
        if s == "evaluate":
            cmd += ["--tag", args.population]
        print(f"\n===== {s}: {' '.join(cmd[1:])} =====", flush=True)
        t = time.perf_counter()
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        print(f"===== {s}: exit {rc} in {time.perf_counter() - t:.0f}s =====", flush=True)
        if rc != 0:
            print(f"pipeline stopped: step '{s}' failed", file=sys.stderr)
            return rc
    print(f"\npipeline finished: {', '.join(steps)} in {time.perf_counter() - t_all:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
