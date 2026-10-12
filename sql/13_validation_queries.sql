/* =============================================================================
   13_validation_queries.sql
   Data-quality checks. ONE result set: (check_name, violations). Every check
   must return 0. Run in SSMS, or via src/validation.py / tests/test_data_quality.py
   (which read this file, so SQL is the single source of truth).

   Section A: source data (this phase). Later phases append derived-data checks
   (sessions, features, cohorts, labels).
   ============================================================================= */
SET NOCOUNT ON;
USE CustomerFeatureStore;

/* ---- A. source data ----------------------------------------------------- */
SELECT check_name, violations FROM (
    -- identifiers
    SELECT 'users_duplicate_id'  AS check_name, CAST(COUNT(*) - COUNT(DISTINCT user_id)  AS BIGINT) AS violations FROM dbo.users
    UNION ALL SELECT 'orders_duplicate_id',  COUNT(*) - COUNT(DISTINCT order_id)  FROM dbo.orders
    UNION ALL SELECT 'events_duplicate_id',  COUNT(*) - COUNT(DISTINCT event_id)  FROM dbo.raw_web_events

    -- referential integrity (FKs are trusted, but verify the data really satisfies them)
    UNION ALL SELECT 'orders_orphan_user', COUNT(*)
        FROM dbo.orders o LEFT JOIN dbo.users u ON u.user_id = o.user_id WHERE u.user_id IS NULL
    UNION ALL SELECT 'events_orphan_user', COUNT(*)
        FROM dbo.raw_web_events e LEFT JOIN dbo.users u ON u.user_id = e.user_id WHERE u.user_id IS NULL

    -- money
    UNION ALL SELECT 'orders_negative_money', COUNT(*)
        FROM dbo.orders WHERE gross_amount < 0 OR discount_amount < 0 OR net_amount < 0
    UNION ALL SELECT 'orders_invalid_status', COUNT(*)
        FROM dbo.orders WHERE order_status NOT IN ('completed','cancelled','refunded')
    UNION ALL SELECT 'orders_refund_inconsistent', COUNT(*)
        FROM dbo.orders
        WHERE (order_status = 'refunded' AND refunded_at IS NULL)
           OR (order_status <> 'refunded' AND refunded_at IS NOT NULL)
           OR refunded_at < order_ts

    -- timestamps are plausible UTC values (not NULL by schema; not absurd; not in the future)
    UNION ALL SELECT 'users_invalid_signup_ts', COUNT(*)
        FROM dbo.users WHERE signup_at < '2000-01-01' OR signup_at > SYSUTCDATETIME()
    UNION ALL SELECT 'orders_invalid_ts', COUNT(*)
        FROM dbo.orders WHERE order_ts < '2000-01-01' OR order_ts > SYSUTCDATETIME()
    UNION ALL SELECT 'events_invalid_ts', COUNT(*)
        FROM dbo.raw_web_events WHERE event_ts < '2000-01-01' OR event_ts > SYSUTCDATETIME()

    -- cross-table temporal rules (cannot be CHECK constraints)
    UNION ALL SELECT 'orders_before_signup', COUNT(*)
        FROM dbo.orders o JOIN dbo.users u ON u.user_id = o.user_id WHERE o.order_ts < u.signup_at
    UNION ALL SELECT 'events_before_signup', COUNT(*)
        FROM dbo.raw_web_events e JOIN dbo.users u ON u.user_id = e.user_id WHERE e.event_ts < u.signup_at
) AS checks
ORDER BY check_name;
