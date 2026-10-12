/* =============================================================================
   13b_validation_derived.sql
   Integrity checks for DERIVED objects (sessions, features, labels, cohorts).
   One result set (check_name, violations); every value must be 0.
   Run after 06-10 have been built. (Source-data checks live in 13_validation_queries.sql.)
   ============================================================================= */
SET NOCOUNT ON;
USE CustomerFeatureStore;

SELECT check_name, violations FROM (
    /* ---- sessions (default 30-minute rule) ---- */
    SELECT 'session_events_row_count_mismatch' AS check_name,
           CAST(ABS((SELECT COUNT_BIG(*) FROM analytics.session_events) - (SELECT COUNT_BIG(*) FROM dbo.raw_web_events)) AS BIGINT) AS violations
    UNION ALL SELECT 'sessions_event_total_mismatch',
           ABS(ISNULL((SELECT SUM(CAST(event_count AS BIGINT)) FROM analytics.sessions), 0) - (SELECT COUNT_BIG(*) FROM dbo.raw_web_events))
    UNION ALL SELECT 'session_internal_gap_over_30min', COUNT(*)
        FROM (SELECT DATEDIFF_BIG(MILLISECOND,
                       LAG(event_ts) OVER (PARTITION BY user_id, session_seq ORDER BY event_ts, event_id), event_ts) AS gap_ms
              FROM analytics.session_events) AS g
        WHERE g.gap_ms > 1800000
    UNION ALL SELECT 'session_boundary_gap_not_over_30min', COUNT(*)
        FROM (SELECT DATEDIFF_BIG(MILLISECOND, session_end_ts,
                       LEAD(session_start_ts) OVER (PARTITION BY user_id ORDER BY session_seq)) AS gap_ms
              FROM analytics.sessions) AS g
        WHERE g.gap_ms IS NOT NULL AND g.gap_ms <= 1800000
    UNION ALL SELECT 'sessions_bad_duration_or_counts', COUNT(*)
        FROM analytics.sessions
        WHERE duration_seconds < 0 OR session_end_ts < session_start_ts
           OR event_count <> page_view_count + product_view_count + add_to_cart_count
                             + checkout_started_count + purchase_event_count

    /* ---- feature snapshots ---- */
    UNION ALL SELECT 'features_last_purchase_not_before_cutoff', COUNT(*)
        FROM analytics.customer_features WHERE last_purchase_ts >= snapshot_ts
    UNION ALL SELECT 'features_user_not_registered_at_cutoff', COUNT(*)
        FROM analytics.customer_features f JOIN dbo.users u ON u.user_id = f.user_id WHERE u.signup_at >= f.snapshot_ts
    UNION ALL SELECT 'features_population_mismatch', COUNT(*)
        FROM (SELECT s.snapshot_ts,
                     (SELECT COUNT(*) FROM analytics.customer_features f WHERE f.snapshot_ts = s.snapshot_ts) AS n_feat,
                     (SELECT COUNT(*) FROM dbo.users u WHERE u.signup_at < s.snapshot_ts)                  AS n_users
              FROM (SELECT DISTINCT snapshot_ts FROM analytics.customer_features) AS s) AS p
        WHERE p.n_feat <> p.n_users
    UNION ALL SELECT 'features_window_not_monotonic', COUNT(*)
        FROM analytics.customer_features
        WHERE orders_30d > orders_90d OR orders_90d > orders_365d OR orders_365d > lifetime_orders
           OR spend_30d > spend_90d OR spend_90d > spend_365d OR spend_365d > lifetime_spend
           OR sessions_7d > sessions_30d OR sessions_30d > sessions_90d
           OR events_7d > events_30d OR events_30d > events_90d
    UNION ALL SELECT 'features_null_vs_zero_violation', COUNT(*)
        FROM analytics.customer_features
        WHERE (has_purchased = 0 AND (lifetime_orders <> 0 OR lifetime_spend <> 0 OR avg_order_value IS NOT NULL
                                      OR days_since_last_purchase IS NOT NULL OR recency_quintile IS NOT NULL))
           OR (has_purchased = 1 AND (lifetime_orders < 1 OR avg_order_value IS NULL OR spend_quintile IS NULL))
    UNION ALL SELECT 'features_quintile_out_of_range', COUNT(*)
        FROM analytics.customer_features
        WHERE recency_quintile NOT BETWEEN 1 AND 5 OR frequency_quintile NOT BETWEEN 1 AND 5 OR spend_quintile NOT BETWEEN 1 AND 5
    UNION ALL SELECT 'features_conversion_out_of_range', COUNT(*)
        FROM analytics.customer_features
        WHERE conversion_rate_30d < 0 OR conversion_rate_30d > 1 OR (conversion_rate_30d IS NOT NULL AND sessions_30d = 0)

    /* ---- labels ---- */
    UNION ALL SELECT 'labels_incomplete_outcome_window', COUNT(*)
        FROM analytics.churn_labels WHERE outcome_window_end > observation_end
    UNION ALL SELECT 'labels_window_not_adjacent_to_cutoff', COUNT(*)
        FROM analytics.churn_labels WHERE outcome_window_start <> snapshot_ts OR outcome_window_end <> DATEADD(DAY, 30, snapshot_ts)
    UNION ALL SELECT 'labels_without_features', COUNT(*)
        FROM analytics.churn_labels l
        WHERE NOT EXISTS (SELECT 1 FROM analytics.customer_features f WHERE f.snapshot_ts = l.snapshot_ts AND f.user_id = l.user_id)
    UNION ALL SELECT 'labels_inconsistent_with_order_count', COUNT(*)
        FROM analytics.churn_labels
        WHERE churn_label <> CASE WHEN qualifying_orders_in_window = 0 THEN 1 ELSE 0 END
    UNION ALL SELECT 'labels_prior_purchase_flag_mismatch', COUNT(*)
        FROM analytics.churn_labels l JOIN analytics.customer_features f ON f.snapshot_ts = l.snapshot_ts AND f.user_id = l.user_id
        WHERE l.has_prior_purchase <> f.has_purchased

    /* ---- cohorts ---- */
    UNION ALL SELECT 'cohort_unobserved_cell_has_value', COUNT(*)
        FROM analytics.cohort_retention_long
        WHERE (is_observed = 0 AND (active_customers IS NOT NULL OR retention_pct IS NOT NULL))
           OR (is_observed = 1 AND (active_customers IS NULL OR retention_pct IS NULL))
    UNION ALL SELECT 'cohort_retention_over_100_or_active_over_size', COUNT(*)
        FROM analytics.cohort_retention_long WHERE retention_pct > 100 OR active_customers > cohort_size
    UNION ALL SELECT 'cohort_size_mismatch', COUNT(*)
        FROM analytics.cohort_retention_long c
        WHERE c.cohort_size <> (SELECT COUNT(*) FROM dbo.users u
                                WHERE DATEFROMPARTS(YEAR(u.signup_at), MONTH(u.signup_at), 1) = c.cohort_month)
) AS checks
ORDER BY check_name;
