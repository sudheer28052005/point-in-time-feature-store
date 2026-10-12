"""Generate reports/performance_report.md + reports/benchmark_summary.csv from analytics.benchmark_*.

Every number comes from the database; runs whose notes start with INVALID are ignored; for each
(benchmark, design) the most recent valid run is used.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent))

import pandas as pd  # noqa: E402

from src import config as C  # noqa: E402
from src.database import read_sql_df  # noqa: E402

DESIGNS = {"A": "A baseline (surrogate clustered PK)", "B": "B user-time clustered", "C": "C B + time NCIs",
           "D": "D C + columnstore", "E": "E B + batch mode", "F": "F final deployed"}
# (label, (benchmark, design), (benchmark, design))  ->  "before" vs "after"
COMPARISONS = [
    ("Customer order history, heaviest user", ("B01_order_history_heavy_user", "A"), ("B01_order_history_heavy_user", "B")),
    ("Customer order history, typical user", ("B02_order_history_typical_user", "A"), ("B02_order_history_typical_user", "B")),
    ("90-day window aggregate, all users (clustering alone)", ("B03_window_agg_90d", "B"), ("B03_window_agg_90d", "B")),
    ("90-day window aggregate: add covering time index", ("B03_window_agg_90d", "B"), ("B03_window_agg_90d", "C")),
    ("1-day window aggregate: add covering time index", ("B04_window_agg_1day", "B"), ("B04_window_agg_1day", "C")),
    ("Non-sargable YEAR()/MONTH() -> range predicate (no time index)", ("B05a_one_month_nonsargable", "B"), ("B05b_one_month_sargable", "B")),
    ("Non-sargable -> range predicate (with time index)", ("B05a_one_month_nonsargable", "C"), ("B05b_one_month_sargable", "C")),
    ("Uncovered 1-day query: time index seek + key lookups", ("B06c_selective_uncovered", "B"), ("B06c_selective_uncovered", "C")),
    ("One user's events in a year", ("B07_events_one_user_window", "A"), ("B07_events_one_user_window", "B")),
    ("Events in one day, all users: add time index", ("B08_events_one_day_slice", "B"), ("B08_events_one_day_slice", "C")),
    ("Full-table event aggregate: nonclustered columnstore", ("B09_events_full_aggregate", "B"), ("B09_events_full_aggregate", "D")),
    ("Full-table event aggregate: batch-mode enabler", ("B09_events_full_aggregate", "B"), ("B09_events_full_aggregate", "E")),
    ("Sessionization core: clustering by user/time (plan has no sort)", ("B10_sessionization_core", "A"), ("B10_sessionization_core", "B")),
    ("Sessionization core: nonclustered columnstore", ("B10_sessionization_core", "B"), ("B10_sessionization_core", "D")),
    ("Sessionization core: batch-mode enabler", ("B10_sessionization_core", "B"), ("B10_sessionization_core", "E")),
    ("Feature snapshot, early cutoff: baseline -> clustered", ("B11a_features_early_cutoff", "A"), ("B11a_features_early_cutoff", "B")),
    ("Feature snapshot, early cutoff: + batch mode", ("B11a_features_early_cutoff", "B"), ("B11a_features_early_cutoff", "E")),
    ("Feature snapshot, late cutoff: baseline -> clustered", ("B11b_features_late_cutoff", "A"), ("B11b_features_late_cutoff", "B")),
    ("Feature snapshot, late cutoff: + batch mode", ("B11b_features_late_cutoff", "B"), ("B11b_features_late_cutoff", "E")),
    ("Session rebuild procedure: batch mode", ("B13_build_sessions_procedure", "B"), ("B13_build_sessions_procedure", "E")),
    ("Cohort retention: correlated sub-select vs project procedure", ("B12a_cohort_correlated_subquery", "B"), ("B12b_cohort_single_pass_procedure", "B")),
    ("END TO END, late-cutoff feature snapshot: baseline A -> deployed", ("B11b_features_late_cutoff", "A"), ("B11b_features_late_cutoff", "F")),
    ("END TO END, early-cutoff feature snapshot: baseline A -> deployed", ("B11a_features_early_cutoff", "A"), ("B11a_features_early_cutoff", "F")),
    ("END TO END, sessionization core: baseline A -> deployed", ("B10_sessionization_core", "A"), ("B10_sessionization_core", "F")),
]


def load() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    runs = read_sql_df("SELECT run_id, notes, environment FROM analytics.benchmark_runs ORDER BY run_id")
    valid = runs[~runs.notes.fillna("").str.startswith("INVALID")]
    ids = "','".join(valid.run_id)
    res = read_sql_df(f"SELECT * FROM analytics.benchmark_results WHERE run_id IN ('{ids}')")   # ids come from the DB, not user input
    pl = read_sql_df(f"SELECT * FROM analytics.benchmark_plans WHERE run_id IN ('{ids}')")
    for col in ("elapsed_ms", "logical_reads", "cpu_ms", "physical_reads"):
        res[col] = res[col].astype(float)
    res["d"] = res.design.str[0]
    pl["d"] = pl.design.str[0]
    res = res[res.run_id == res.groupby(["benchmark_id", "d"]).run_id.transform("max")]
    pl = pl[pl.run_id == pl.groupby(["benchmark_id", "d"]).run_id.transform("max")]
    env = json.loads(valid.environment.iloc[-1])
    env["runs_used"] = {r.run_id: (r.notes or "") for r in valid.itertuples()}
    return res, pl, env


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and pd.isna(v)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def fmt_ms(v: float) -> str:
    return "" if pd.isna(v) else (f"{v:,.0f}" if v >= 100 else f"{v:,.1f}")


def main() -> int:
    res, pl, env = load()
    warm = res[res.cache_state == "warm"].groupby(["benchmark_id", "d"]).agg(
        ms=("elapsed_ms", "median"), ms_min=("elapsed_ms", "min"), ms_max=("elapsed_ms", "max"),
        reads=("logical_reads", "median"), cpu=("cpu_ms", "median"), n=("rep", "count"))
    cold = res[res.cache_state == "cold"].groupby(["benchmark_id", "d"]).agg(ms=("elapsed_ms", "median"), phys=("physical_reads", "median"))
    out = C.PROJECT_ROOT / "reports"
    summary = warm.join(cold, rsuffix="_cold").reset_index()
    summary.to_csv(out / "benchmark_summary.csv", index=False)

    bench_ids = sorted(res.benchmark_id.unique())
    rows_t, rows_r = [], []
    for b in bench_ids:
        t, r = {"benchmark": b}, {"benchmark": b}
        for d in "ABCDEF":
            if (b, d) in warm.index:
                w = warm.loc[(b, d)]
                t[d], r[d] = fmt_ms(w.ms), f"{int(w.reads):,}"
        rows_t.append(t)
        rows_r.append(r)

    comp = []
    for label, (b1, d1), (b2, d2) in COMPARISONS:
        if (b1, d1) not in warm.index or (b2, d2) not in warm.index or (b1, d1) == (b2, d2):
            continue
        x, y = warm.loc[(b1, d1)], warm.loc[(b2, d2)]
        comp.append({"comparison": label, "before ms": fmt_ms(x.ms), "after ms": fmt_ms(y.ms),
                     "time x": f"{x.ms / y.ms:,.1f}", "before reads": f"{int(x.reads):,}", "after reads": f"{int(y.reads):,}",
                     "reads x": f"{x.reads / max(y.reads, 1):,.1f}"})

    pf = pl.sort_values(["benchmark_id", "d"])[["benchmark_id", "d", "index_scans", "index_seeks", "key_lookups", "sorts", "hash_ops",
                                                "spills", "columnstore_scans", "batch_mode_ops", "memory_grant_kb", "worst_estimate_factor"]]
    pf = pf.rename(columns={"d": "design", "index_scans": "scans", "index_seeks": "seeks", "key_lookups": "lookups",
                            "columnstore_scans": "cs scans", "batch_mode_ops": "batch ops", "memory_grant_kb": "grant KB",
                            "worst_estimate_factor": "worst est. factor"})
    wo = json.loads((out / "write_overhead.json").read_text()) if (out / "write_overhead.json").exists() else {}
    wo_rows = [{"table": t, "rows": f"{v['base']['rows']:,}", "base insert ms": f"{v['base']['insert_ms_median']:,.0f}",
                "with time index ms": f"{v['base_plus_time_nci']['insert_ms_median']:,.0f}", "overhead": f"{v['overhead_pct']:+.0f}%"}
               for t, v in wo.items()]
    sizes = pd.DataFrame({d[:1]: pd.Series({k.split('.')[1]: v for k, v in s.items()}) for d, s in env.get("design_sizes_mb", {}).items()}).T
    sizes["total"] = sizes.sum(axis=1)

    md = [f"""# Performance report

*Generated by `scripts/benchmark_report.py` from `analytics.benchmark_results` / `analytics.benchmark_plans`.
Nothing in the tables below is typed by hand.*

## Environment

| | |
|---|---|
| SQL Server | {env['sql_server_edition']} {env['sql_server_version']}, database compatibility level {env['compatibility_level']} |
| Hardware | {env['logical_cpus']} logical CPUs visible to SQL Server, {env['physical_memory_mb']:,} MB RAM, NVMe SSD (laptop, shared with other applications) |
| Server settings | {json.dumps(env['config'])} (SQL Server memory target {env['sql_target_memory_mb']:,} MB) |
| Dataset | {', '.join(f'{k} = {v:,}' for k, v in env['row_counts'].items())} |
| Method | {env['cache_method']}. Warm = median of the warm repetitions; spread (min-max) is in `reports/benchmark_summary.csv`. |
| Runs used | {'; '.join(f'`{k}`: {v}' for k, v in env['runs_used'].items())} |

Physical designs: {'; '.join(DESIGNS.values())}.
Designs A-D ran with batch mode OFF (the `bm_enabler` synonym points at a rowstore twin); E turns it on; F is the
layout actually deployed in this database after the decisions below (B + `IX_orders_ts` + batch mode on).

## Median warm elapsed time (ms)

{md_table(pd.DataFrame(rows_t))}

## Median warm logical reads (pages)

{md_table(pd.DataFrame(rows_r))}

## Before / after

{md_table(pd.DataFrame(comp))}

## Actual-plan facts (per benchmark and design)

`spills` = sort/hash spills to tempdb reported in the actual plan; `worst est. factor` = max over operators of
max(estimated/actual, actual/estimated) rows (operators with >= 100 rows).

{md_table(pf.fillna(""))}

## Index storage (MB used by orders + raw_web_events + session_events)

{md_table(sizes.round(1).reset_index().rename(columns={'index': 'design'}))}

## Write overhead of a candidate time index (bulk insert in clustered-key order, median of 3)

{md_table(pd.DataFrame(wo_rows))}
"""]
    def W(b, d, col="ms"):
        return warm.loc[(b, d)][col]

    findings = f"""
## Findings and decisions

**Hypotheses that were CONFIRMED**
- *Clustering by (user_id, ts, id) makes per-user history cheap.* Heaviest user's orders: {int(W('B01_order_history_heavy_user','A','reads')):,} -> {int(W('B01_order_history_heavy_user','B','reads')):,} logical reads; one user's yearly events: {int(W('B07_events_one_user_window','A','reads')):,} -> {int(W('B07_events_one_user_window','B','reads')):,}.
- *Functions on indexed columns hurt.* The `YEAR()/MONTH()` form costs {W('B05a_one_month_nonsargable','B'):.0f} ms vs {W('B05b_one_month_sargable','B'):.0f} ms for the equivalent range on the same plan shape (pure CPU: reads are identical), and with a time index {W('B05a_one_month_nonsargable','C'):.0f} ms / {int(W('B05a_one_month_nonsargable','C','reads')):,} reads vs {W('B05b_one_month_sargable','C'):.0f} ms / {int(W('B05b_one_month_sargable','C','reads')):,} reads.
- *SQL Server correctly scans when a large share of the table is read.* The uncovered one-year query (B06b) reads {int(W('B06b_large_share_uncovered','C','reads')):,} pages with the time index present - the same clustered scan as without it - while the same query on one day (B06c) switches to seek + key lookups.
- *Batch mode is a large win for window functions / DISTINCT aggregates.* Sessionization core {W('B10_sessionization_core','B')/1000:.1f} s -> {W('B10_sessionization_core','E')/1000:.1f} s with identical results.

**Hypotheses that were REJECTED or only partly true**
- *"Clustering by user/time removes the sort and speeds up sessionization."* The plan facts confirm design A has a Sort (about {int(pl[(pl.benchmark_id=='B10_sessionization_core')&(pl.d=='A')].memory_grant_kb.iloc[0]/1024)} MB grant, no spill) and B does not, but runtime is unchanged ({W('B10_sessionization_core','A')/1000:.1f} s vs {W('B10_sessionization_core','B')/1000:.1f} s): the cost is the row-mode window spool, not the sort.
- *"The single-pass cohort procedure is faster than a naive formulation."* No. SQL Server already collapses the correlated sub-select to one pass (3 scans in the plan); the procedure is {W('B12b_cohort_single_pass_procedure','B')/W('B12a_cohort_correlated_subquery','B'):.1f}x the time because it also persists the long-format triangle with observed/unobserved flags. It was kept for its semantics, not speed.
- *"A time index on every table is worth it."* Reads fall sharply for time-slice queries, but a bulk insert gets roughly 2x slower on orders/events and about 7x slower on `session_events` (table above). Only `IX_orders_ts` was adopted (sql/11).
- *Late-cutoff feature builds barely benefit from clustering alone* ({W('B11b_features_late_cutoff','A')/1000:.1f} s -> {W('B11b_features_late_cutoff','B')/1000:.1f} s) because they read most of the history; batch mode is what helped ({W('B11b_features_late_cutoff','E')/1000:.1f} s).

**Deployed design (F)** = clustered (user_id, ts, id) + nonclustered PK + `IX_orders_ts` + batch-mode enabler on.
Not adopted: `IX_session_events_ts`, `IX_raw_web_events_ts` (optional), nonclustered columnstore on events, filtered indexes
(not benchmarked), table partitioning (not evaluated). Rationale in `sql/11_create_indexes.sql`.

**Missing-index suggestions** appeared in the actual plans for the order-window queries ((order_ts)-leading, INCLUDE status/net_amount).
They were treated as hypotheses, tested as design C, and adopted only for `orders`.

## Caveats (read before quoting any number)
- One laptop, other applications running, SQL Server **Express** (buffer pool ~1.4 GB, batch-mode DOP 1, columnstore memory cap). Absolute times will differ elsewhere; ratios and logical reads travel better.
- **Logical reads are exact; elapsed times are not.** Between design blocks (a few minutes apart) there is visible environmental drift: e.g. identical plans on the same index differed up to 3x for sub-50 ms queries (B03 on C vs D). Do not read meaning into differences of that size on small queries.
- Warm = median of 5 (3 for features, 2 for session rebuild); cold = median of 3 (1 for the long procedures). Cold runs drop the whole server buffer pool; the plan is already compiled, so cold numbers exclude compilation.
- Designs A-D and E were measured in two consecutive runs (E in a separate run minutes later); F in a third. Run 1 was discarded because of a harness bug (design D parsed as bare heaps) - it is kept in the database marked INVALID.
- Result hashes were compared for every benchmark across every design and every equivalent variant: all identical. A faster query that returned different rows would have failed the run.
- Batch mode here relies on an empty-columnstore-table join; a supported alternative is a real columnstore index (design D) at a storage and write cost. Batch-mode plans request larger memory grants (see `grant KB`); check them if concurrency matters.
- Not tested: concurrent workloads, SQL Server Standard/Enterprise/Developer, other data volumes, table partitioning, filtered indexes.
"""
    md.append(findings)
    (out / "performance_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out / 'performance_report.md'} and benchmark_summary.csv ({len(comp)} comparisons)")
    print(md_table(pd.DataFrame(comp)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
