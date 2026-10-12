"""Run the SQL Server performance benchmarks defined in sql/12_performance_benchmarks.sql.

    python scripts/run_benchmarks.py                       # all designs A-D, all benchmarks
    python scripts/run_benchmarks.py --designs B,C --only B03,B04 --final-design B

WARNING (dev instance only): each design DROPS and REBUILDS indexes on dbo.orders,
dbo.raw_web_events and analytics.session_events, and cold runs execute DBCC DROPCLEANBUFFERS
(server-wide). No data is changed. At the end the --final-design layout is restored.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent))

import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402

from src import config as C  # noqa: E402
from src.benchmarking import Measurer, median, parse_benchmark_file, summarize_plans  # noqa: E402
from src.database import build_odbc_connection_string, get_engine  # noqa: E402

log = logging.getLogger("benchmarks")
SQL_FILE = C.PROJECT_ROOT / "sql" / "12_performance_benchmarks.sql"
STATS_SQL = ("UPDATE STATISTICS dbo.users WITH FULLSCAN; UPDATE STATISTICS dbo.orders WITH FULLSCAN; "
             "UPDATE STATISTICS dbo.raw_web_events WITH FULLSCAN; UPDATE STATISTICS analytics.session_events WITH FULLSCAN;")


def load_params(m: Measurer) -> dict:
    one = lambda q: m.cur.execute(q).fetchone()[0]  # noqa: E731
    t0, early = datetime(2025, 1, 1), datetime(2023, 7, 1)
    return {
        # the busiest customer / a typical 4-order customer (deterministic for a given dataset)
        "u_heavy": one("SELECT TOP 1 user_id FROM dbo.orders GROUP BY user_id ORDER BY COUNT(*) DESC, user_id"),
        "u_typical": one("SELECT TOP 1 user_id FROM dbo.orders GROUP BY user_id HAVING COUNT(*) = 4 ORDER BY user_id"),
        "t0": t0, "t0_early": early,
        "win90_from": datetime(2024, 10, 3), "win365_from": datetime(2024, 1, 1),
        "day_from": datetime(2024, 12, 10), "day_to": datetime(2024, 12, 11),
    }


def environment(m: Measurer) -> dict:
    q = lambda sql: [list(r) for r in m.cur.execute(sql).fetchall()]  # noqa: E731
    info = q("SELECT cpu_count, hyperthread_ratio, physical_memory_kb/1024, committed_target_kb/1024 FROM sys.dm_os_sys_info")[0]
    cfg = dict(q("SELECT name, CAST(value_in_use AS BIGINT) FROM sys.configurations WHERE name IN "
                 "('max server memory (MB)','max degree of parallelism','cost threshold for parallelism')"))
    props = q("SELECT CAST(SERVERPROPERTY('Edition') AS VARCHAR(100)), CAST(SERVERPROPERTY('ProductVersion') AS VARCHAR(50)), "
              "(SELECT compatibility_level FROM sys.databases WHERE name = DB_NAME())")[0]
    counts = {t: int(m.cur.execute(f"SELECT COUNT_BIG(*) FROM {t}").fetchone()[0])
              for t in ("dbo.users", "dbo.orders", "dbo.raw_web_events", "analytics.session_events")}
    return {"sql_server_edition": props[0], "sql_server_version": props[1], "compatibility_level": props[2],
            "logical_cpus": info[0], "hyperthread_ratio": info[1], "physical_memory_mb": info[2],
            "sql_target_memory_mb": info[3], "config": cfg, "os": platform.platform(), "row_counts": counts,
            "client": f"python {platform.python_version()}, pyodbc", "cache_method": "cold = CHECKPOINT + DBCC DROPCLEANBUFFERS; "
            "warm = repeated runs after one uncounted warm-up; plan compiled before measured runs"}


def design_sizes(m: Measurer) -> dict:
    rows = m.cur.execute("""SELECT OBJECT_SCHEMA_NAME(object_id) + '.' + OBJECT_NAME(object_id), SUM(used_page_count) * 8 / 1024.0
                            FROM sys.dm_db_partition_stats
                            WHERE object_id IN (OBJECT_ID('dbo.orders'), OBJECT_ID('dbo.raw_web_events'), OBJECT_ID('analytics.session_events'))
                            GROUP BY object_id""").fetchall()
    return {r[0]: round(float(r[1]), 1) for r in rows}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--designs", default="A,B,C,D,E")
    p.add_argument("--only", default="", help="comma-separated benchmark id prefixes, e.g. B03,B04")
    p.add_argument("--cold", type=int, default=3)
    p.add_argument("--warm", type=int, default=5)
    p.add_argument("--final-design", default="B", help="design to leave in place afterwards (A-D)")
    p.add_argument("--notes", default="")
    args = p.parse_args(argv)
    C.configure_logging()

    bf = parse_benchmark_file(SQL_FILE)
    by_letter = {name[0]: name for name in bf.designs if not name.startswith("_")}
    wanted = [d.strip() for d in args.designs.split(",")]
    only = [o.strip() for o in args.only.split(",") if o.strip()]
    run_id = datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    plan_dir = C.PROJECT_ROOT / "reports" / "plans" / run_id
    plan_dir.mkdir(parents=True, exist_ok=True)

    m = Measurer(build_odbc_connection_string(C.get_settings()))
    results, plans, hashes, sizes = [], [], {}, {}
    try:
        m.run_script(bf.setup)
        params = load_params(m)
        env = environment(m)
        log.info("run %s | env: %s | params: %s", run_id, {k: env[k] for k in ("sql_server_edition", "logical_cpus", "physical_memory_mb")}, params)

        for letter in wanted:
            current = letter.upper() == "CURRENT"      # measure the deployed layout as-is (no index changes)
            design = "F_final_deployed" if current else by_letter[letter]
            if current:
                log.info("=== measuring the CURRENTLY DEPLOYED design (no changes applied) ===")
                m.run_script(STATS_SQL)
            else:
                log.info("=== applying design %s ===", design)
                m.run_script(bf.designs["_reset"])
                m.run_script(bf.designs[design])
                m.run_script(STATS_SQL)
            sizes[design] = design_sizes(m)
            for b in bf.benchmarks:
                if (not current and letter not in b.designs) or (only and not any(b.benchmark_id.startswith(o) for o in only)):
                    continue
                bound = [(n, t, params[k]) for n, k, t in b.params]
                shown = ", ".join(f"{n}={v:%Y-%m-%d}" if isinstance(v, datetime) else f"{n}={v}" for n, _, v in bound)
                m.execute(b.sql, bound)                                   # uncounted warm-up: compiles + caches the plan
                reps = [("cold", i) for i in range(args.cold if b.cold is None else b.cold)] + \
                       [("warm", i) for i in range(args.warm if b.warm is None else b.warm)]
                row_hashes = set()
                for state, i in reps:
                    r = m.run(b.sql, bound, cold=(state == "cold"))
                    row_hashes.add(r["result_hash"])
                    results.append(dict(run_id=run_id, benchmark_id=b.benchmark_id, workload=b.workload, variant=b.variant,
                                        design=design, cache_state=state, rep=i + 1, parameters=shown or None,
                                        **{k: r[k] for k in ("logical_reads", "physical_reads", "cpu_ms", "elapsed_ms", "result_hash")}))
                hashes[(b.benchmark_id, design)] = row_hashes
                plan_xml = m.capture_plans(b.sql, bound)
                ps = summarize_plans(plan_xml)
                fname = f"{b.benchmark_id}__{design[0]}.sqlplan"
                try:
                    plan_dir.mkdir(parents=True, exist_ok=True)
                    (plan_dir / fname).write_text("\n".join(plan_xml), encoding="utf-8")
                except OSError as exc:     # the summary row below is stored regardless
                    log.warning("could not write %s: %s", fname, exc)
                plans.append(dict(run_id=run_id, benchmark_id=b.benchmark_id, design=design, plan_file=f"plans/{run_id}/{fname}", **ps))
                warm = [x for x in results if x["benchmark_id"] == b.benchmark_id and x["design"] == design and x["cache_state"] == "warm"]
                log.info("%-34s %-32s warm median: %8.1f ms | %9d logical reads | %s", b.benchmark_id, design,
                         median([x["elapsed_ms"] for x in warm]), int(median([x["logical_reads"] for x in warm])),
                         ps["access_paths"][:90])
    finally:
        if args.final_design.upper() != "KEEP":     # KEEP = leave the layout untouched (used with --designs current)
            final = by_letter[args.final_design]
            log.info("=== restoring final design %s ===", final)
            m.run_script(bf.designs["_reset"])
            m.run_script(bf.designs[final])
            m.run_script(STATS_SQL)
        m.close()

    # ---- correctness gate: every design (and every equiv variant) must return identical results ----
    problems = []
    for b in bf.benchmarks:
        hs = {h for (bid, d), v in hashes.items() if bid == b.benchmark_id for h in v}
        if len(hs) > 1:
            problems.append(f"{b.benchmark_id}: result differs between runs/designs {hs}")
    for grp in {b.equiv for b in bf.benchmarks if b.equiv}:
        hs = {h for b in bf.benchmarks if b.equiv == grp for (bid, d), v in hashes.items() if bid == b.benchmark_id for h in v}
        if len(hs) > 1:
            problems.append(f"equiv group {grp}: variants return different results {hs}")
    for pr in problems:
        log.error("RESULT MISMATCH: %s", pr)

    env["design_sizes_mb"] = sizes
    env["result_equivalence_problems"] = problems
    eng = get_engine()
    with eng.begin() as c:
        c.execute(text("INSERT analytics.benchmark_runs (run_id, environment, notes) VALUES (:r, :e, :n)"),
                  {"r": run_id, "e": json.dumps(env, default=str), "n": args.notes or None})
        c.exec_driver_sql("INSERT analytics.benchmark_results (run_id, benchmark_id, workload, variant, design, cache_state, rep, "
                          "logical_reads, physical_reads, cpu_ms, elapsed_ms, result_hash, parameters) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                          [tuple(r[k] for k in ("run_id", "benchmark_id", "workload", "variant", "design", "cache_state", "rep",
                                                "logical_reads", "physical_reads", "cpu_ms", "elapsed_ms", "result_hash", "parameters")) for r in results])
        cols = ["run_id", "benchmark_id", "design", "index_scans", "index_seeks", "key_lookups", "sorts", "hash_ops", "spills",
                "columnstore_scans", "batch_mode_ops", "parallel_ops", "memory_grant_kb", "memory_used_kb", "worst_estimate_factor",
                "worst_estimate_operator", "access_paths", "missing_index_hint", "plan_file"]
        c.exec_driver_sql(f"INSERT analytics.benchmark_plans ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                          [tuple(p_[k] for k in cols) for p_ in plans])
    (C.PROJECT_ROOT / "reports" / "benchmark_environment.json").write_text(json.dumps(env, indent=2, default=str))
    df = pd.DataFrame(results)
    summary = (df.groupby(["benchmark_id", "design", "cache_state"])
                 .agg(logical_reads=("logical_reads", "median"), cpu_ms=("cpu_ms", "median"),
                      elapsed_ms=("elapsed_ms", "median"), elapsed_min=("elapsed_ms", "min"), elapsed_max=("elapsed_ms", "max"),
                      physical_reads=("physical_reads", "median"), runs=("rep", "count")).reset_index())
    summary.to_csv(C.PROJECT_ROOT / "reports" / "benchmark_summary.csv", index=False)
    log.info("stored run %s: %d measurements, %d plans; summary -> reports/benchmark_summary.csv", run_id, len(results), len(plans))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
