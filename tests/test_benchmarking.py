"""Unit tests for the benchmark harness (no database needed)."""
import textwrap

import pytest

from src.benchmarking import parse_benchmark_file, result_hash, summarize_plans

NS = 'xmlns="http://schemas.microsoft.com/sqlserver/2004/07/showplan"'


def write(tmp_path, body):
    p = tmp_path / "bench.sql"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_last_design_block_is_not_swallowed_by_following_banner(tmp_path):
    """Regression: a greedy regex once gave the LAST design an empty body (it silently benchmarked bare heaps)."""
    p = write(tmp_path, """
        -- @design X_first
        /*
        CREATE INDEX a ON t (c);
        */

        -- @design Y_last
        /*
        CREATE INDEX b ON t (d);
        GO
        CREATE INDEX e ON t (f);
        */

        /* ===================== 1. section banner ===================== */

        -- @benchmark B01_q | workload=w | variant=single | designs=X,Y | params=@u:u_heavy:BIGINT
        SELECT 1;
    """)
    f = parse_benchmark_file(p)
    assert "CREATE INDEX a" in f.designs["X_first"] and "CREATE INDEX b" in f.designs["Y_last"]
    assert "CREATE INDEX e" in f.designs["Y_last"] and "banner" not in f.designs["Y_last"]


def test_design_without_ddl_is_rejected(tmp_path):
    p = write(tmp_path, """
        -- @design Z_empty
        /* just prose, no DDL block on its own line */
        -- @benchmark B_x | workload=w | designs=Z
        SELECT 1;
    """)
    with pytest.raises(ValueError, match="no DDL block"):
        parse_benchmark_file(p)


def test_benchmark_directive_fields(tmp_path):
    p = write(tmp_path, """
        -- @benchmark B09_x | workload=agg | variant=baseline | designs=A,B | equiv=grp | cold=1 | warm=2 | params=@from:d1:DATETIME2(3),@to:d2:DATETIME2(3)
        /* a banner that must not leak into the query */
        SELECT COUNT(*) FROM t WHERE c >= @from AND c < @to;
    """)
    b = parse_benchmark_file(p).benchmarks[0]
    assert (b.benchmark_id, b.workload, b.variant, b.designs, b.equiv, b.cold, b.warm) == \
           ("B09_x", "agg", "baseline", ["A", "B"], "grp", 1, 2)
    assert b.params == [("@from", "d1", "DATETIME2(3)"), ("@to", "d2", "DATETIME2(3)")]
    assert b.sql.startswith("SELECT COUNT(*)") and "banner" not in b.sql


def test_real_benchmark_file_is_wellformed():
    from src.config import PROJECT_ROOT
    f = parse_benchmark_file(PROJECT_ROOT / "sql" / "12_performance_benchmarks.sql")
    designs = {n[0] for n in f.designs if not n.startswith("_")}
    assert designs == {"A", "B", "C", "D", "E"} and all(f.designs.values())
    assert len({b.benchmark_id for b in f.benchmarks}) == len(f.benchmarks)           # unique ids
    for b in f.benchmarks:
        assert set(b.designs) <= designs, b.benchmark_id
        assert b.sql and "@" not in "".join(n for n, _, _ in []) and all(n in b.sql for n, _, _ in b.params), b.benchmark_id
    assert any("CREATE NONCLUSTERED COLUMNSTORE" in v for v in f.designs.values())


def test_result_hash_is_order_independent_and_value_sensitive():
    a = [(1, "x", 2.00001), (2, "y", 3.0)]
    assert result_hash(a) == result_hash(list(reversed(a)))
    assert result_hash(a) != result_hash([(1, "x", 2.5), (2, "y", 3.0)])


def test_plan_summary_counts_operators_spills_lookups_and_estimates():
    xml = f"""<ShowPlanXML {NS}><BatchSequence><Batch><Statements><StmtSimple><QueryPlan>
      <MemoryGrantInfo GrantedMemory="4096" MaxUsedMemory="1024"/>
      <MissingIndexes><MissingIndexGroup Impact="85.5"><MissingIndex Database="[d]" Schema="[dbo]" Table="[orders]">
        <ColumnGroup Usage="INEQUALITY"><Column Name="[order_ts]"/></ColumnGroup>
        <ColumnGroup Usage="INCLUDE"><Column Name="[net_amount]"/></ColumnGroup></MissingIndex></MissingIndexGroup></MissingIndexes>
      <RelOp PhysicalOp="Sort" LogicalOp="Sort" EstimateRows="10" ><Warnings><SpillToTempDb SpillLevel="1"/></Warnings>
        <RunTimeInformation><RunTimeCountersPerThread Thread="0" ActualRows="10" ActualExecutions="1"/></RunTimeInformation>
        <RelOp PhysicalOp="Nested Loops" LogicalOp="Inner Join" EstimateRows="1000" EstimateRebinds="0" EstimateRewinds="0">
          <RunTimeInformation><RunTimeCountersPerThread Thread="0" ActualRows="50000" ActualExecutions="1"/></RunTimeInformation>
          <RelOp PhysicalOp="Index Seek" LogicalOp="Index Seek" EstimateRows="1000">
            <IndexScan><Object Schema="[dbo]" Table="[orders]" Index="[IX_orders_ts]"/></IndexScan></RelOp>
          <RelOp PhysicalOp="Clustered Index Seek" LogicalOp="Clustered Index Seek" EstimateRows="1">
            <IndexScan Lookup="1"><Object Schema="[dbo]" Table="[orders]" Index="[CIX_orders_user_ts]"/></IndexScan></RelOp>
          <RelOp PhysicalOp="Clustered Index Scan" LogicalOp="Clustered Index Scan" EstimateRows="5" EstimatedExecutionMode="Batch">
            <IndexScan Storage="ColumnStore"><Object Schema="[dbo]" Table="[raw_web_events]" Index="[NCCI]"/></IndexScan></RelOp>
        </RelOp></RelOp></QueryPlan></StmtSimple></Statements></Batch></BatchSequence></ShowPlanXML>"""
    s = summarize_plans([xml])
    assert (s["index_seeks"], s["key_lookups"], s["sorts"], s["spills"], s["columnstore_scans"], s["batch_mode_ops"]) == (1, 1, 1, 1, 1, 1)
    assert (s["memory_grant_kb"], s["memory_used_kb"]) == (4096, 1024)
    assert s["worst_estimate_factor"] == pytest.approx(50.0, rel=0.05) and "Nested Loops" in s["worst_estimate_operator"]
    assert "impact 86%" in s["missing_index_hint"] or "impact 85%" in s["missing_index_hint"]
    assert "Seek dbo.orders.IX_orders_ts" in s["access_paths"] and "ColumnstoreScan dbo.raw_web_events.NCCI" in s["access_paths"]
    assert "CIX_orders_user_ts" not in s["access_paths"]          # lookups are counted separately, not listed as access paths
