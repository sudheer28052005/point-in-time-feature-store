"""Benchmark harness helpers: parse sql/12_performance_benchmarks.sql, measure queries, analyse plans.

Measurement method
* logical/physical reads and CPU come from sys.dm_exec_sessions deltas of ONE dedicated connection
  (exact counters, no text parsing); elapsed = client wall clock including fetching all results.
* cold = CHECKPOINT + DBCC DROPCLEANBUFFERS before the run (server-wide; dev instance only);
  warm = repeated runs after one uncounted warm-up. The plan is compiled (cached) before cold runs,
  so cold numbers measure buffer-pool misses, not compilation.
* every query is run through sp_executesql with real parameter values, like an application would.
* ACTUAL plans are captured separately with SET STATISTICS XML ON and summarised.
"""
from __future__ import annotations

import hashlib
import logging
import re
import statistics
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyodbc

log = logging.getLogger(__name__)
NS = {"sp": "http://schemas.microsoft.com/sqlserver/2004/07/showplan"}
_DIRECTIVE = re.compile(r"^-- @(\w+)\s*(.*)$")


@dataclass
class Benchmark:
    benchmark_id: str
    workload: str
    variant: str
    designs: list[str]
    params: list[tuple[str, str, str]]       # (@name, registry key, SQL type)
    equiv: str | None
    cold: int | None
    warm: int | None
    sql: str


@dataclass
class BenchFile:
    setup: str = ""
    designs: dict[str, str] = field(default_factory=dict)
    benchmarks: list[Benchmark] = field(default_factory=list)


def split_batches(sql: str) -> list[str]:
    return [b.strip() for b in re.split(r"(?im)^\s*GO\s*$", sql) if b.strip()]


def parse_benchmark_file(path: Path) -> BenchFile:
    text = path.read_text(encoding="utf-8")
    out, cur, body = BenchFile(), None, []

    def flush():
        if cur is None:
            return
        kind, arg = cur
        content = "\n".join(body).strip()
        if kind == "setup":
            out.setup = content
        elif kind == "design":
            # first block comment that starts on its own line = the DDL (a leading prose comment may precede it,
            # and section banners may follow it), so match non-greedily
            m = re.search(r"/\*[ \t]*\n(.*?)\n\*/", content, re.S)
            if not m or not m.group(1).strip():
                raise ValueError(f"design '{arg}' has no DDL block - refusing to benchmark an empty design")
            out.designs[arg] = m.group(1)
        elif kind == "benchmark":
            content = re.sub(r"/\*.*?\*/", "", content, flags=re.S).strip()   # drop trailing section banners
            out.benchmarks.append(_parse_benchmark(arg, content))

    for line in text.splitlines():
        m = _DIRECTIVE.match(line)
        if m:
            flush()
            cur, body = (m.group(1), m.group(2).strip()), []
        else:
            body.append(line)
    flush()
    # the `_reset` design block's first inner comment is documentation; keep only the executable part
    return out


def _parse_benchmark(header: str, sql: str) -> Benchmark:
    parts = [p.strip() for p in header.split("|")]
    kv = dict(p.split("=", 1) for p in parts[1:] if "=" in p)
    params = []
    for item in filter(None, (kv.get("params") or "").split(",")):
        name, key, typ = item.split(":", 2) if item.count(":") >= 2 else (*item.split(":", 1), "")
        params.append((name, key, typ))
    # DATETIME2(3) contains no comma, so the simple split above is safe
    return Benchmark(
        benchmark_id=parts[0], workload=kv["workload"], variant=kv.get("variant", "single"),
        designs=kv["designs"].split(","), params=params, equiv=kv.get("equiv"),
        cold=int(kv["cold"]) if "cold" in kv else None, warm=int(kv["warm"]) if "warm" in kv else None,
        sql=sql.strip())


# --------------------------------------------------------------------------- #
# plan analysis
# --------------------------------------------------------------------------- #
def summarize_plans(plan_xmls: list[str]) -> dict[str, Any]:
    s = dict(index_scans=0, index_seeks=0, key_lookups=0, sorts=0, hash_ops=0, spills=0, columnstore_scans=0,
             batch_mode_ops=0, parallel_ops=0, memory_grant_kb=None, memory_used_kb=None,
             worst_estimate_factor=None, worst_estimate_operator=None, access_paths="", missing_index_hint="")
    paths, missing = [], []
    for xml in plan_xmls:
        root = ET.fromstring(xml)
        for qp in root.iter(f"{{{NS['sp']}}}QueryPlan"):
            mg = qp.find("sp:MemoryGrantInfo", NS)
            if mg is not None:
                for key, attr in (("memory_grant_kb", "GrantedMemory"), ("memory_used_kb", "MaxUsedMemory")):
                    v = int(mg.get(attr, 0))
                    s[key] = max(s[key] or 0, v)
            for mi in qp.findall("sp:MissingIndexes/sp:MissingIndexGroup", NS):
                idx = mi.find("sp:MissingIndex", NS)
                cols = {g.get("Usage"): [c.get("Name") for c in g.findall("sp:Column", NS)] for g in idx.findall("sp:ColumnGroup", NS)}
                missing.append(f"impact {float(mi.get('Impact')):.0f}% on {idx.get('Schema')}.{idx.get('Table')}: "
                               f"EQ{cols.get('EQUALITY', [])} INEQ{cols.get('INEQUALITY', [])} INC{cols.get('INCLUDE', [])}")
        for rel in root.iter(f"{{{NS['sp']}}}RelOp"):
            op, logical = rel.get("PhysicalOp", ""), rel.get("LogicalOp", "")
            scan = next((c for c in rel if c.tag.endswith(("IndexScan", "TableScan"))), None)
            obj = scan.find("sp:Object", NS) if scan is not None else None
            is_lookup = scan is not None and scan.get("Lookup") in ("1", "true")
            if is_lookup:
                s["key_lookups"] += 1
            elif op in ("Clustered Index Scan", "Index Scan", "Table Scan"):
                if scan is not None and scan.get("Storage") == "ColumnStore":
                    s["columnstore_scans"] += 1
                else:
                    s["index_scans"] += 1
            elif op in ("Clustered Index Seek", "Index Seek"):
                s["index_seeks"] += 1
            if obj is not None and not is_lookup:
                kind = "Scan" if "Scan" in op else "Seek"
                if scan.get("Storage") == "ColumnStore":
                    kind = "ColumnstoreScan"
                paths.append(f"{kind} {obj.get('Schema', '').strip('[]')}.{obj.get('Table', '').strip('[]')}.{obj.get('Index', '').strip('[]')}")
            if op in ("Sort", "Top N Sort", "Distinct Sort"):
                s["sorts"] += 1
            if op == "Hash Match":
                s["hash_ops"] += 1
            if op == "Parallelism":
                s["parallel_ops"] += 1
            if rel.get("EstimatedExecutionMode") == "Batch" or rel.get("ActualExecutionMode") == "Batch":
                s["batch_mode_ops"] += 1
            for w in rel.findall("sp:Warnings", NS):
                s["spills"] += len([c for c in w if c.tag.split("}")[1] in ("SpillToTempDb", "SortSpillDetails", "HashSpillDetails")])
            counters = rel.findall("sp:RunTimeInformation/sp:RunTimeCountersPerThread", NS)
            if counters:
                actual = sum(int(c.get("ActualRows", 0)) for c in counters)
                est = float(rel.get("EstimateRows", 0)) * (1 + float(rel.get("EstimateRebinds", 0)) + float(rel.get("EstimateRewinds", 0)))
                if max(actual, est) >= 100:
                    factor = max((est + 1) / (actual + 1), (actual + 1) / (est + 1))
                    if s["worst_estimate_factor"] is None or factor > s["worst_estimate_factor"]:
                        s["worst_estimate_factor"] = round(factor, 1)
                        s["worst_estimate_operator"] = f"{op} est={est:,.0f} actual={actual:,}"[:200]
    # distinct, ordered, compact
    s["access_paths"] = "; ".join(dict.fromkeys(paths))[:1000]
    s["missing_index_hint"] = " | ".join(dict.fromkeys(missing))[:1000]
    return s


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
class Measurer:
    """One dedicated autocommit connection so session counters belong to exactly one client."""

    def __init__(self, odbc_connection_string: str):
        self.conn = pyodbc.connect(odbc_connection_string, autocommit=True)
        self.cur = self.conn.cursor()
        self.cur.execute("SET ARITHABORT ON")     # same setting as SSMS, so plans match what a reader reproduces there

    def _drain(self) -> tuple[list[tuple], list[str]]:
        last_rows, plans = [], []
        while True:
            if self.cur.description:
                rows = self.cur.fetchall()
                if len(rows) == 1 and len(rows[0]) == 1 and isinstance(rows[0][0], str) and rows[0][0].startswith("<ShowPlanXML"):
                    plans.append(rows[0][0])
                elif rows:
                    last_rows = [tuple(r) for r in rows]
            if not self.cur.nextset():
                break
        return last_rows, plans

    def _stats(self) -> tuple[int, int, int]:
        r = self.cur.execute("SELECT logical_reads, reads, cpu_time FROM sys.dm_exec_sessions WHERE session_id = @@SPID").fetchone()
        return int(r[0]), int(r[1]), int(r[2])

    def execute(self, sql: str, params: list[tuple[str, str, Any]]) -> tuple[list[tuple], list[str]]:
        if params:
            decl = ", ".join(f"{n} {t}" for n, t, _ in params)
            bind = ", ".join(f"{n} = ?" for n, _, _ in params)
            self.cur.execute(f"EXEC sys.sp_executesql ?, ?, {bind}", (sql, decl, *[v for *_, v in params]))
        else:
            self.cur.execute("EXEC sys.sp_executesql ?", (sql,))
        return self._drain()

    def run(self, sql: str, params, cold: bool) -> dict[str, Any]:
        if cold:
            self.cur.execute("CHECKPOINT")
            self.cur.execute("DBCC DROPCLEANBUFFERS")
            self._drain()
        l0, p0, c0 = self._stats()
        t0 = time.perf_counter()
        rows, _ = self.execute(sql, params)
        wall = (time.perf_counter() - t0) * 1000
        l1, p1, c1 = self._stats()
        return dict(logical_reads=l1 - l0, physical_reads=p1 - p0, cpu_ms=c1 - c0, elapsed_ms=round(wall, 2),
                    result_hash=result_hash(rows), rows=len(rows))

    def capture_plans(self, sql: str, params) -> list[str]:
        self.cur.execute("SET STATISTICS XML ON")
        try:
            _, plans = self.execute(sql, params)
        finally:
            self.cur.execute("SET STATISTICS XML OFF")
            self._drain()
        return plans

    def run_script(self, script: str) -> None:
        for batch in split_batches(script):
            self.cur.execute(batch)
            self._drain()

    def close(self):
        self.conn.close()


def result_hash(rows: list[tuple]) -> str:
    """Order-independent hash of a result set (floats rounded so designs compare equal)."""
    norm = sorted(repr(tuple(round(v, 4) if isinstance(v, float) else v for v in r)) for r in rows)
    return hashlib.sha1("\n".join(norm).encode()).hexdigest()[:16]


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")
