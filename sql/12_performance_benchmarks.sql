/* =============================================================================
   12_performance_benchmarks.sql
   Benchmark DEFINITIONS (not a script to run top-to-bottom!).
   scripts/run_benchmarks.py parses the `-- @...` directives below, applies each physical
   design, runs every query under controlled cache conditions and stores logical reads,
   CPU, elapsed time and a summary of the ACTUAL execution plan in analytics.benchmark_*.

   In SSMS you may run a single benchmark query on its own: declare the @parameters it
   lists (the harness picks them: heavy user, typical user, cutoff dates) and use
   "Include Actual Execution Plan" + SET STATISTICS IO, TIME ON. Design DDL blocks are
   inside /* */ on purpose so an accidental F5 cannot drop indexes.

   PHYSICAL DESIGNS (hypotheses, measured - not assumed)
     A  baseline        clustered PK on the surrogate id only (what a default schema gives you)
     B  user-time       PK nonclustered + clustered (user_id, ts, id): per-user history is contiguous
     C  B + time NCIs   B + covering nonclustered indexes on the timestamp (cross-user time slices)
     D  C + columnstore D + nonclustered columnstore on raw_web_events (analytic scans)
     E  B + batch mode  B + the batch-mode enabler (empty columnstore join, see 06_sessionization.sql)
     Designs A-D run with batch mode OFF: the _reset block points analytics.bm_enabler at a rowstore twin.
   ============================================================================= */

-- @setup
IF OBJECT_ID(N'analytics.benchmark_runs', N'U') IS NULL
    CREATE TABLE analytics.benchmark_runs
    (
        run_id        VARCHAR(40)   NOT NULL CONSTRAINT PK_benchmark_runs PRIMARY KEY,
        started_at    DATETIME2(3)  NOT NULL CONSTRAINT DF_benchmark_runs_started DEFAULT (SYSUTCDATETIME()),
        environment   NVARCHAR(MAX) NOT NULL,        -- JSON: hardware, SQL Server version/edition, settings, row counts
        notes         NVARCHAR(400) NULL
    );
GO
IF OBJECT_ID(N'analytics.benchmark_results', N'U') IS NULL
    CREATE TABLE analytics.benchmark_results
    (
        result_id     INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_benchmark_results PRIMARY KEY,
        run_id        VARCHAR(40)   NOT NULL,
        benchmark_id  VARCHAR(60)   NOT NULL,
        workload      VARCHAR(60)   NOT NULL,
        variant       VARCHAR(30)   NOT NULL,        -- baseline | optimized | single
        design        VARCHAR(40)   NOT NULL,
        cache_state   VARCHAR(10)   NOT NULL,        -- cold (buffer pool dropped) | warm
        rep           INT           NOT NULL,
        logical_reads BIGINT        NOT NULL,
        physical_reads BIGINT       NOT NULL,
        cpu_ms        INT           NOT NULL,
        elapsed_ms    DECIMAL(12,2) NOT NULL,        -- client wall clock incl. fetching results
        result_hash   CHAR(16)      NULL,            -- hash of the result set: all designs/variants must agree
        parameters    NVARCHAR(400) NULL
    );
GO
IF OBJECT_ID(N'analytics.benchmark_plans', N'U') IS NULL
    CREATE TABLE analytics.benchmark_plans
    (
        run_id        VARCHAR(40)   NOT NULL,
        benchmark_id  VARCHAR(60)   NOT NULL,
        design        VARCHAR(40)   NOT NULL,
        index_scans   INT           NOT NULL,
        index_seeks   INT           NOT NULL,
        key_lookups   INT           NOT NULL,
        sorts         INT           NOT NULL,
        hash_ops      INT           NOT NULL,
        spills        INT           NOT NULL,        -- sort/hash spills to tempdb reported in the actual plan
        columnstore_scans INT       NOT NULL,
        batch_mode_ops INT          NOT NULL,
        parallel_ops  INT           NOT NULL,
        memory_grant_kb BIGINT      NULL,
        memory_used_kb  BIGINT      NULL,
        worst_estimate_factor DECIMAL(14,1) NULL,    -- max(est/actual, actual/est) over operators
        worst_estimate_operator VARCHAR(200) NULL,
        access_paths  NVARCHAR(1000) NULL,           -- e.g. "Seek dbo.orders.CIX_orders_user_ts; Scan ..."
        missing_index_hint NVARCHAR(1000) NULL,      -- a HYPOTHESIS, never applied automatically
        plan_file     VARCHAR(200)  NULL,
        CONSTRAINT PK_benchmark_plans PRIMARY KEY (run_id, benchmark_id, design)
    );
GO

-- @design _reset
/* Drops every project index/constraint on the three big tables, leaving heaps. Each design
   then builds its own layout, so designs can be applied in any order. */
/*
DECLARE @sql NVARCHAR(MAX) = N'';
SELECT @sql += N'DROP INDEX ' + QUOTENAME(i.name) + N' ON ' + QUOTENAME(s.name) + N'.' + QUOTENAME(t.name) + N';'
FROM sys.indexes AS i
JOIN sys.tables  AS t ON t.object_id = i.object_id
JOIN sys.schemas AS s ON s.schema_id = t.schema_id
WHERE t.name IN (N'orders', N'raw_web_events', N'session_events') AND s.name IN (N'dbo', N'analytics')
  AND i.type > 0 AND i.is_primary_key = 0 AND i.is_unique_constraint = 0;
EXEC (@sql);
GO
DECLARE @sql NVARCHAR(MAX) = N'';
SELECT @sql += N'ALTER TABLE ' + QUOTENAME(s.name) + N'.' + QUOTENAME(t.name) + N' DROP CONSTRAINT ' + QUOTENAME(k.name) + N';'
FROM sys.key_constraints AS k
JOIN sys.tables  AS t ON t.object_id = k.parent_object_id
JOIN sys.schemas AS s ON s.schema_id = t.schema_id
WHERE k.type = 'PK' AND t.name IN (N'orders', N'raw_web_events', N'session_events') AND s.name IN (N'dbo', N'analytics');
EXEC (@sql);
GO
EXEC analytics.usp_set_batch_mode @enabled = 0;   -- every design starts in row mode
*/

-- @design A_baseline_surrogate_clustered
/*
ALTER TABLE dbo.orders             ADD CONSTRAINT PK_orders          PRIMARY KEY CLUSTERED (order_id);
ALTER TABLE dbo.raw_web_events     ADD CONSTRAINT PK_raw_web_events  PRIMARY KEY CLUSTERED (event_id);
ALTER TABLE analytics.session_events ADD CONSTRAINT PK_session_events PRIMARY KEY CLUSTERED (event_id);
*/

-- @design B_user_time_clustered
/*
ALTER TABLE dbo.orders             ADD CONSTRAINT PK_orders          PRIMARY KEY NONCLUSTERED (order_id);
CREATE UNIQUE CLUSTERED INDEX CIX_orders_user_ts          ON dbo.orders (user_id, order_ts, order_id);
GO
ALTER TABLE dbo.raw_web_events     ADD CONSTRAINT PK_raw_web_events  PRIMARY KEY NONCLUSTERED (event_id);
CREATE UNIQUE CLUSTERED INDEX CIX_raw_web_events_user_ts  ON dbo.raw_web_events (user_id, event_ts, event_id);
GO
ALTER TABLE analytics.session_events ADD CONSTRAINT PK_session_events PRIMARY KEY NONCLUSTERED (event_id);
CREATE UNIQUE CLUSTERED INDEX CIX_session_events_user_ts  ON analytics.session_events (user_id, event_ts, event_id);
*/

-- @design C_user_time_plus_time_nci
/*
ALTER TABLE dbo.orders             ADD CONSTRAINT PK_orders          PRIMARY KEY NONCLUSTERED (order_id);
CREATE UNIQUE CLUSTERED INDEX CIX_orders_user_ts          ON dbo.orders (user_id, order_ts, order_id);
GO
ALTER TABLE dbo.raw_web_events     ADD CONSTRAINT PK_raw_web_events  PRIMARY KEY NONCLUSTERED (event_id);
CREATE UNIQUE CLUSTERED INDEX CIX_raw_web_events_user_ts  ON dbo.raw_web_events (user_id, event_ts, event_id);
GO
ALTER TABLE analytics.session_events ADD CONSTRAINT PK_session_events PRIMARY KEY NONCLUSTERED (event_id);
CREATE UNIQUE CLUSTERED INDEX CIX_session_events_user_ts  ON analytics.session_events (user_id, event_ts, event_id);
GO
-- The clustering key (user_id, ts, id) is carried in every nonclustered index automatically.
CREATE NONCLUSTERED INDEX IX_orders_ts          ON dbo.orders (order_ts)           INCLUDE (order_status, refunded_at, net_amount);
CREATE NONCLUSTERED INDEX IX_raw_web_events_ts  ON dbo.raw_web_events (event_ts)   INCLUDE (event_type);
CREATE NONCLUSTERED INDEX IX_session_events_ts  ON analytics.session_events (event_ts) INCLUDE (event_type, reconstructed_session_id);
*/

-- @design D_plus_columnstore
/*
ALTER TABLE dbo.orders             ADD CONSTRAINT PK_orders          PRIMARY KEY NONCLUSTERED (order_id);
CREATE UNIQUE CLUSTERED INDEX CIX_orders_user_ts          ON dbo.orders (user_id, order_ts, order_id);
GO
ALTER TABLE dbo.raw_web_events     ADD CONSTRAINT PK_raw_web_events  PRIMARY KEY NONCLUSTERED (event_id);
CREATE UNIQUE CLUSTERED INDEX CIX_raw_web_events_user_ts  ON dbo.raw_web_events (user_id, event_ts, event_id);
GO
ALTER TABLE analytics.session_events ADD CONSTRAINT PK_session_events PRIMARY KEY NONCLUSTERED (event_id);
CREATE UNIQUE CLUSTERED INDEX CIX_session_events_user_ts  ON analytics.session_events (user_id, event_ts, event_id);
GO
CREATE NONCLUSTERED INDEX IX_orders_ts          ON dbo.orders (order_ts)           INCLUDE (order_status, refunded_at, net_amount);
CREATE NONCLUSTERED INDEX IX_raw_web_events_ts  ON dbo.raw_web_events (event_ts)   INCLUDE (event_type);
CREATE NONCLUSTERED INDEX IX_session_events_ts  ON analytics.session_events (event_ts) INCLUDE (event_type, reconstructed_session_id);
GO
CREATE NONCLUSTERED COLUMNSTORE INDEX NCCI_raw_web_events
    ON dbo.raw_web_events (event_id, user_id, event_ts, event_type, device_type);
*/

-- @design E_B_plus_batch_mode
/*
ALTER TABLE dbo.orders             ADD CONSTRAINT PK_orders          PRIMARY KEY NONCLUSTERED (order_id);
CREATE UNIQUE CLUSTERED INDEX CIX_orders_user_ts          ON dbo.orders (user_id, order_ts, order_id);
GO
ALTER TABLE dbo.raw_web_events     ADD CONSTRAINT PK_raw_web_events  PRIMARY KEY NONCLUSTERED (event_id);
CREATE UNIQUE CLUSTERED INDEX CIX_raw_web_events_user_ts  ON dbo.raw_web_events (user_id, event_ts, event_id);
GO
ALTER TABLE analytics.session_events ADD CONSTRAINT PK_session_events PRIMARY KEY NONCLUSTERED (event_id);
CREATE UNIQUE CLUSTERED INDEX CIX_session_events_user_ts  ON analytics.session_events (user_id, event_ts, event_id);
GO
EXEC analytics.usp_set_batch_mode @enabled = 1;
*/

/* ===================== 1. customer order history ===================== */

-- @benchmark B01_order_history_heavy_user | workload=order_history | variant=single | designs=A,B,C,D | params=@u:u_heavy:BIGINT
SELECT order_id, order_ts, order_status, net_amount
FROM dbo.orders
WHERE user_id = @u
ORDER BY order_ts DESC;

-- @benchmark B02_order_history_typical_user | workload=order_history | variant=single | designs=A,B,C,D | params=@u:u_typical:BIGINT
SELECT order_id, order_ts, order_status, net_amount
FROM dbo.orders
WHERE user_id = @u
ORDER BY order_ts DESC;

/* ===================== 2. time-windowed transaction aggregation ===================== */

-- @benchmark B03_window_agg_90d | workload=window_agg | variant=single | designs=A,B,C,D | params=@from:win90_from:DATETIME2(3),@to:t0:DATETIME2(3)
SELECT COUNT_BIG(*) AS users, SUM(orders) AS orders, SUM(spend) AS spend     -- aggregated server-side: no client fetch noise
FROM (SELECT user_id, COUNT(*) AS orders, SUM(net_amount) AS spend
      FROM dbo.orders
      WHERE order_ts >= @from AND order_ts < @to AND order_status = 'completed'
      GROUP BY user_id) AS x;

-- @benchmark B04_window_agg_1day | workload=window_agg_selective | variant=single | designs=A,B,C,D | params=@from:day_from:DATETIME2(3),@to:day_to:DATETIME2(3)
SELECT COUNT_BIG(*) AS users, SUM(orders) AS orders, SUM(spend) AS spend
FROM (SELECT user_id, COUNT(*) AS orders, SUM(net_amount) AS spend
      FROM dbo.orders
      WHERE order_ts >= @from AND order_ts < @to AND order_status = 'completed'
      GROUP BY user_id) AS x;

-- Sargability: same question, a function wrapped around the indexed column vs an explicit range.
-- @benchmark B05a_one_month_nonsargable | workload=sargability | variant=baseline | designs=B,C | equiv=sargability
SELECT COUNT_BIG(*) AS orders, SUM(net_amount) AS spend
FROM dbo.orders
WHERE YEAR(order_ts) = 2024 AND MONTH(order_ts) = 12 AND order_status = 'completed';

-- @benchmark B05b_one_month_sargable | workload=sargability | variant=optimized | designs=B,C | equiv=sargability
SELECT COUNT_BIG(*) AS orders, SUM(net_amount) AS spend
FROM dbo.orders
WHERE order_ts >= '2024-12-01' AND order_ts < '2025-01-01' AND order_status = 'completed';

-- A range covering ~45% of the table: SQL Server SHOULD scan even when a time index exists.
-- @benchmark B06_large_share_one_year | workload=large_share_scan | variant=single | designs=B,C | params=
SELECT COUNT_BIG(*) AS orders, SUM(net_amount) AS spend
FROM dbo.orders
WHERE order_ts >= '2024-01-01' AND order_ts < '2025-01-01' AND order_status = 'completed';

-- payment_method is NOT in IX_orders_ts, so the index no longer covers the query (lookups cost ~3 reads/row).
-- ~45% of the table: the optimizer should choose a clustered scan, not 90k key lookups.
-- @benchmark B06b_large_share_uncovered | workload=index_choice_large | variant=single | designs=B,C | params=
SELECT COUNT_BIG(*) AS orders, SUM(net_amount) AS spend, MAX(payment_method) AS any_method
FROM dbo.orders
WHERE order_ts >= '2024-01-01' AND order_ts < '2025-01-01';

-- Same uncovered query on one day (~0.2% of rows): index seek + key lookups is now cheaper than scanning.
-- @benchmark B06c_selective_uncovered | workload=index_choice_small | variant=single | designs=B,C | params=
SELECT COUNT_BIG(*) AS orders, SUM(net_amount) AS spend, MAX(payment_method) AS any_method
FROM dbo.orders
WHERE order_ts >= '2024-12-10' AND order_ts < '2024-12-11';

/* ===================== 3. web-event filtering ===================== */

-- @benchmark B07_events_one_user_window | workload=event_user_window | variant=single | designs=A,B,C,D | params=@u:u_heavy:BIGINT,@from:win365_from:DATETIME2(3),@to:t0:DATETIME2(3)
SELECT event_ts, event_type
FROM dbo.raw_web_events
WHERE user_id = @u AND event_ts >= @from AND event_ts < @to
ORDER BY event_ts, event_id;

-- @benchmark B08_events_one_day_slice | workload=event_time_slice | variant=single | designs=A,B,C,D | params=@from:day_from:DATETIME2(3),@to:day_to:DATETIME2(3)
SELECT event_type, COUNT_BIG(*) AS events
FROM dbo.raw_web_events
WHERE event_ts >= @from AND event_ts < @to
GROUP BY event_type;

-- Full-table analytic aggregate (reads 100% of the rows): columnstore territory, if the edition allows.
-- @benchmark B09_events_full_aggregate | workload=event_full_aggregate | variant=single | designs=B,C,D,E | params=
SELECT YEAR(event_ts) AS yr, MONTH(event_ts) AS mon, event_type, device_type, COUNT_BIG(*) AS events
FROM dbo.raw_web_events AS e
LEFT JOIN analytics.bm_enabler AS bm ON 1 = 0    -- row mode unless the enabler (design E) or a columnstore index (D) is present
GROUP BY YEAR(event_ts), MONTH(event_ts), event_type, device_type;

/* ===================== 4. session reconstruction ===================== */

-- @benchmark B10_sessionization_core | workload=sessionization | variant=single | designs=A,B,C,D,E | params=
;WITH ordered AS
(
    SELECT user_id, event_ts, event_id,
           LAG(event_ts) OVER (PARTITION BY user_id ORDER BY event_ts, event_id) AS prev_ts
    FROM dbo.raw_web_events AS e
    LEFT JOIN analytics.bm_enabler AS bm ON 1 = 0   -- batch-mode enabler: adds no rows; active only in design E
),
flagged AS
(
    SELECT user_id, event_ts, event_id,
           CASE WHEN prev_ts IS NULL OR DATEDIFF_BIG(MILLISECOND, prev_ts, event_ts) > 1800000 THEN 1 ELSE 0 END AS is_new
    FROM ordered
),
numbered AS
(
    SELECT user_id,
           SUM(is_new) OVER (PARTITION BY user_id ORDER BY event_ts, event_id ROWS UNBOUNDED PRECEDING) AS session_seq
    FROM flagged
)
SELECT COUNT_BIG(*) AS users, SUM(CAST(m AS BIGINT)) AS sessions
FROM (SELECT user_id, MAX(session_seq) AS m FROM numbered GROUP BY user_id) AS x;

/* ===================== 5. point-in-time feature generation ===================== */

-- @benchmark B11a_features_early_cutoff | workload=feature_build_early | variant=single | designs=A,B,C,E | params=@t0:t0_early:DATETIME2(3) | cold=1 | warm=3
EXEC analytics.usp_build_customer_features @T0 = @t0;

-- @benchmark B11b_features_late_cutoff | workload=feature_build_late | variant=single | designs=A,B,C,E | params=@t0:t0:DATETIME2(3) | cold=1 | warm=3
EXEC analytics.usp_build_customer_features @T0 = @t0;

-- Whole session rebuild (TRUNCATE + INSERT of ~1M rows + aggregation): what the enabler buys end-to-end.
-- @benchmark B13_build_sessions_procedure | workload=sessions_build | variant=single | designs=B,E | params= | cold=1 | warm=2
EXEC analytics.usp_build_sessions;

/* ===================== 6. cohort retention ===================== */

-- Written as one correlated sub-select per (cohort, month index) cell. The expectation was repeated scans of
-- orders; the ACTUAL plan shows SQL Server collapses it to a single pass, so this is a fair comparison.
-- @benchmark B12a_cohort_correlated_subquery | workload=cohort_retention | variant=baseline | designs=B,C | equiv=cohort | cold=1 | warm=1
;WITH cohorts AS
(
    SELECT DATEFROMPARTS(YEAR(signup_at), MONTH(signup_at), 1) AS cohort_month
    FROM dbo.users WHERE signup_at < '2025-07-01'
    GROUP BY DATEFROMPARTS(YEAR(signup_at), MONTH(signup_at), 1)
),
grid AS
(
    SELECT c.cohort_month, CAST(g.value AS INT) AS idx
    FROM cohorts AS c CROSS JOIN GENERATE_SERIES(0, 29) AS g
    WHERE DATEADD(MONTH, CAST(g.value AS INT), c.cohort_month) <= '2025-06-01'
)
SELECT SUM(CAST(active AS BIGINT)) AS total_active_cells, COUNT_BIG(*) AS cells
FROM (
    SELECT g.cohort_month, g.idx,
           (SELECT COUNT(DISTINCT o.user_id)
            FROM dbo.orders AS o JOIN dbo.users AS u ON u.user_id = o.user_id
            WHERE DATEFROMPARTS(YEAR(u.signup_at), MONTH(u.signup_at), 1) = g.cohort_month
              AND DATEFROMPARTS(YEAR(o.order_ts), MONTH(o.order_ts), 1) = DATEADD(MONTH, g.idx, g.cohort_month)
              AND o.order_status = 'completed') AS active
    FROM grid AS g
) AS x;

-- The project's procedure: orders read once into distinct (user, month) pairs, aggregated, and the full long-format
-- triangle (incl. unobserved cells) is persisted (08_retention_cohorts.sql). Not expected to be faster: it does more.
-- @benchmark B12b_cohort_single_pass_procedure | workload=cohort_retention | variant=alternative | designs=B,C | equiv=cohort
EXEC analytics.usp_build_cohort_retention @observation_end = '2025-07-01', @activity_definition = 'purchase';
SELECT SUM(CAST(active_customers AS BIGINT)) AS total_active_cells, COUNT_BIG(*) AS cells
FROM analytics.cohort_retention_long
WHERE activity_definition = 'purchase' AND is_observed = 1;

/* ===================== Query Store (run in SSMS after a benchmark session) =====================
   SELECT TOP (10) q.query_id, SUM(rs.count_executions) AS execs,
          SUM(rs.avg_logical_io_reads * rs.count_executions) AS total_logical_reads,
          SUM(rs.avg_cpu_time * rs.count_executions) / 1000 AS total_cpu_ms,
          MAX(CAST(qt.query_sql_text AS NVARCHAR(200))) AS query_text
   FROM sys.query_store_query q
   JOIN sys.query_store_query_text qt ON qt.query_text_id = q.query_text_id
   JOIN sys.query_store_plan p ON p.query_id = q.query_id
   JOIN sys.query_store_runtime_stats rs ON rs.plan_id = p.plan_id
   GROUP BY q.query_id ORDER BY total_logical_reads DESC;
   ============================================================================================= */
