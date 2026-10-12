/* =============================================================================
   11_create_indexes.sql
   Workload-driven indexes ON TOP of the base layout from 03_create_tables.sql
   (PK nonclustered + clustered (user_id, ts, id) = benchmark design "B").
   Every decision below is backed by sql/12_performance_benchmarks.sql results and the
   write-overhead experiment (scripts/benchmark_write_overhead.py); see reports/performance_report.md.

   CREATED
     IX_orders_ts  (order_ts) INCLUDE (order_status, refunded_at, net_amount)
        + covers cross-user time-window queries (user_id/order_id come from the clustering key)
        + measured: 1-day window 2,369 -> 8 logical reads; sargable month 2,369 -> 118 reads;
          uncovered 1-day query uses seek + key lookups (1,721 reads vs 2,369 scan)
        - cost: +11 MB (+47% of the table) and ~+1.4 s on a 200k-row bulk load
        It does NOT help (and SQL Server correctly ignores it for) uncovered queries that read ~45% of
        the table (B06b: clustered scan chosen, 2,369 reads either way).

   EVALUATED AND DELIBERATELY NOT CREATED
     IX_session_events_ts  (event_ts) INCLUDE (event_type, reconstructed_session_id)
        no benefit beyond measurement noise for the feature build, +50 MB, and bulk insert ~6.9x slower.
     IX_raw_web_events_ts  (event_ts) INCLUDE (event_type)          -- optional, see below
        one-day cross-user slice: 11,930 -> 18 logical reads (~660x), BUT +43 MB and bulk insert +129%
        (6.2 s -> 14.1 s per 1M rows). No core workload slices events across users by time, so it is not
        created by default. Uncomment if you add such a dashboard query.
     NCCI_raw_web_events   nonclustered columnstore on events
        full-table aggregate 2.0 s -> 0.51 s and sessionization core 8.2 s -> 2.3 s, but the batch-mode
        enabler (06_sessionization.sql) reaches 1.1 s for sessionization with zero extra storage and no
        write penalty. Not needed.
     Filtered indexes   not benchmarked: the qualifying-order predicate (completed OR refunded after T)
        removes only the ~7% cancelled rows, so a filtered index would be nearly as large as IX_orders_ts.
     Table partitioning not evaluated and not adopted: every core query is per-user or whole-history;
        the data is ~260 MB; partition elimination would not help per-user seeks, and the clustered key
        would have to contain the partition column. Revisit at 100x the data volume.
   Missing-index DMV/plan suggestions were treated as hypotheses: the (order_ts)-leading index above is
   what the optimizer suggested for the time-window queries, validated by the benchmark before adoption.
   ============================================================================= */
SET NOCOUNT ON;
USE CustomerFeatureStore;
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_orders_ts' AND object_id = OBJECT_ID(N'dbo.orders'))
    CREATE NONCLUSTERED INDEX IX_orders_ts
        ON dbo.orders (order_ts)
        INCLUDE (order_status, refunded_at, net_amount);
GO

/* OPTIONAL (not created by default; trade-off documented above)
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_raw_web_events_ts' AND object_id = OBJECT_ID(N'dbo.raw_web_events'))
    CREATE NONCLUSTERED INDEX IX_raw_web_events_ts ON dbo.raw_web_events (event_ts) INCLUDE (event_type);
*/

-- Keep optimizer statistics fresh after bulk loads (cheap at this size).
UPDATE STATISTICS dbo.orders WITH FULLSCAN;
GO
PRINT 'Workload indexes ready (IX_orders_ts).';
GO
