/* =============================================================================
   07_customer_features.sql
   Point-in-time customer feature snapshots.

   TEMPORAL CONTRACT (one rule for the whole project)
     * A snapshot is defined by a cutoff @T0 (UTC).
     * Features use information with timestamp  <  @T0  (the cutoff is EXCLUDED).
       Windows are half-open:  [@T0 - N days, @T0).   No row with ts >= @T0 is read.
     * Population: users with signup_at < @T0.
     * The outcome window (09_churn_labels.sql) is [@T0, @T0 + 30 days): it starts exactly
       where the feature window ends -> no overlap, no gap.

   QUALIFYING ORDER at time t (analytics.fn_qualifying_orders)
     status = 'completed'                                    -> qualifies
     status = 'refunded' AND refunded_at >= t                -> qualifies (refund not yet happened at t)
     status = 'refunded' AND refunded_at <  t, or 'cancelled'-> does not qualify
     Revenue = net_amount (gross - discount). Refunds are full-order reversals, so each
     order is counted once or not at all: no negative revenue, no double counting.
     For features t = @T0, so an order refunded AFTER @T0 still counts in that snapshot
     (that is what was knowable then).

   MISSING vs ZERO
     counts and sums           -> 0 when there is nothing (zero is the true value)
     averages, stdevs, medians,
     days_since_last_purchase  -> NULL when undefined (never fabricated as 0)
     quintiles / RFM scores    -> NULL for users without a purchase; their segment is 'No Purchase'

   RFM: scores 1..5 (5 = best) via NTILE(5) over customers who HAVE purchased as of @T0
     (non-purchasers would only pollute the lowest bucket). R: days since last purchase
     (5 = most recent). F: lifetime orders. M: lifetime spend. NTILE splits tied values
     arbitrarily across buckets, so ties are broken by user_id: deterministic, but ties in
     frequency (many 1-order customers) are split artificially. Known limitation.
   ============================================================================= */
SET NOCOUNT ON;
USE CustomerFeatureStore;
GO

CREATE OR ALTER FUNCTION analytics.fn_qualifying_orders
(
    @from_inclusive DATETIME2(3),
    @to_exclusive   DATETIME2(3),
    @status_asof    DATETIME2(3)   -- the moment at which refund status is evaluated
)
RETURNS TABLE
AS RETURN
(
    SELECT o.order_id, o.user_id, o.order_ts, o.net_amount
    FROM dbo.orders AS o
    WHERE o.order_ts >= @from_inclusive          -- sargable range on the clustered key (user_id, order_ts)
      AND o.order_ts <  @to_exclusive
      AND (   o.order_status = 'completed'
           OR (o.order_status = 'refunded' AND o.refunded_at >= @status_asof))
);
GO

IF OBJECT_ID(N'analytics.customer_features', N'U') IS NULL
BEGIN
    CREATE TABLE analytics.customer_features
    (
        snapshot_ts                 DATETIME2(3)  NOT NULL,   -- the cutoff @T0 (history is strictly before it)
        user_id                     BIGINT        NOT NULL,
        -- static attributes known at signup
        country                     CHAR(2)       NOT NULL,
        acquisition_channel         VARCHAR(30)   NOT NULL,
        customer_segment            VARCHAR(20)   NOT NULL,
        is_marketing_opted_in       BIT           NOT NULL,
        account_age_days            INT           NOT NULL,
        -- recency
        has_purchased               BIT           NOT NULL,
        last_purchase_ts            DATETIME2(3)  NULL,
        days_since_last_purchase    DECIMAL(9,2)  NULL,
        -- frequency
        orders_30d                  INT           NOT NULL,
        orders_90d                  INT           NOT NULL,
        orders_365d                 INT           NOT NULL,
        lifetime_orders             INT           NOT NULL,
        orders_prev_30d             INT           NOT NULL,   -- orders in [T0-60d, T0-30d)
        order_count_change_30d      INT           NOT NULL,   -- orders_30d - orders_prev_30d
        -- monetary (net_amount of qualifying orders)
        spend_30d                   DECIMAL(14,2) NOT NULL,
        spend_90d                   DECIMAL(14,2) NOT NULL,
        spend_365d                  DECIMAL(14,2) NOT NULL,
        lifetime_spend              DECIMAL(14,2) NOT NULL,
        avg_order_value             DECIMAL(12,2) NULL,
        max_order_value             DECIMAL(12,2) NULL,
        std_order_value             DECIMAL(12,2) NULL,       -- sample stdev, needs >= 2 orders
        -- purchase intervals (days between consecutive qualifying orders)
        mean_purchase_interval_days   DECIMAL(9,2) NULL,
        std_purchase_interval_days    DECIMAL(9,2) NULL,      -- sample stdev, needs >= 2 intervals
        median_purchase_interval_days DECIMAL(9,2) NULL,
        distinct_purchase_dates     INT           NOT NULL,
        -- web behaviour (from reconstructed sessions / events strictly before T0)
        sessions_7d                 INT           NOT NULL,
        sessions_30d                INT           NOT NULL,
        sessions_90d                INT           NOT NULL,
        events_7d                   INT           NOT NULL,
        events_30d                  INT           NOT NULL,
        events_90d                  INT           NOT NULL,
        product_views_30d           INT           NOT NULL,
        cart_additions_30d          INT           NOT NULL,
        checkout_starts_30d         INT           NOT NULL,
        -- conversion_rate_30d = (sessions with >=1 purchase event in the 30d window)
        --                       / (sessions with >=1 event in the 30d window); NULL if no sessions
        conversion_rate_30d         DECIMAL(7,4)  NULL,
        has_web_events              BIT           NOT NULL,   -- any event before T0 (completeness indicator)
        -- segmentation
        recency_quintile            TINYINT       NULL,
        frequency_quintile          TINYINT       NULL,
        spend_quintile              TINYINT       NULL,
        rfm_segment                 VARCHAR(30)   NOT NULL,
        -- lineage
        feature_generated_at        DATETIME2(3)  NOT NULL,
        CONSTRAINT PK_customer_features PRIMARY KEY CLUSTERED (snapshot_ts, user_id)
    );
END
GO

CREATE OR ALTER PROCEDURE analytics.usp_build_customer_features
    @T0      DATETIME2(3),
    @replace BIT = 1          -- 1: rebuild this snapshot if it exists; 0: fail if it exists
AS
BEGIN
    SET NOCOUNT ON;
    SET XACT_ABORT ON;
    IF @T0 IS NULL THROW 50003, '@T0 is required.', 1;
    IF @replace = 0 AND EXISTS (SELECT 1 FROM analytics.customer_features WHERE snapshot_ts = @T0)
        THROW 50004, 'Snapshot already exists; pass @replace = 1 to rebuild it.', 1;

    DECLARE @d7   DATETIME2(3) = DATEADD(DAY,   -7, @T0),
            @d30  DATETIME2(3) = DATEADD(DAY,  -30, @T0),
            @d60  DATETIME2(3) = DATEADD(DAY,  -60, @T0),
            @d90  DATETIME2(3) = DATEADD(DAY,  -90, @T0),
            @d365 DATETIME2(3) = DATEADD(DAY, -365, @T0);

    DROP TABLE IF EXISTS #q, #o, #i, #w, #base, #rank;

    /* 1. Qualifying orders strictly before @T0, with the gap to each user's previous order.
          Computed ONCE; every order-based feature reads this temp table (no re-scans of dbo.orders). */
    SELECT order_id, user_id, order_ts, net_amount,
           CAST(DATEDIFF_BIG(SECOND, LAG(order_ts) OVER (PARTITION BY user_id ORDER BY order_ts, order_id),
                             order_ts) / 86400.0 AS FLOAT) AS interval_days
    INTO #q
    FROM analytics.fn_qualifying_orders('1900-01-01', @T0, @T0) AS fq
    LEFT JOIN analytics.bm_enabler AS bm ON 1 = 0;     -- batch-mode enabler (06_sessionization.sql), adds no rows

    /* 2. One row per user: order aggregates (single pass, conditional aggregation). */
    SELECT user_id,
           COUNT(*)                                                     AS lifetime_orders,
           SUM(net_amount)                                              AS lifetime_spend,
           MAX(order_ts)                                                AS last_purchase_ts,
           SUM(CASE WHEN order_ts >= @d30  THEN 1 ELSE 0 END)           AS orders_30d,
           SUM(CASE WHEN order_ts >= @d90  THEN 1 ELSE 0 END)           AS orders_90d,
           SUM(CASE WHEN order_ts >= @d365 THEN 1 ELSE 0 END)           AS orders_365d,
           SUM(CASE WHEN order_ts >= @d60 AND order_ts < @d30 THEN 1 ELSE 0 END) AS orders_prev_30d,
           SUM(CASE WHEN order_ts >= @d30  THEN net_amount ELSE 0 END)  AS spend_30d,
           SUM(CASE WHEN order_ts >= @d90  THEN net_amount ELSE 0 END)  AS spend_90d,
           SUM(CASE WHEN order_ts >= @d365 THEN net_amount ELSE 0 END)  AS spend_365d,
           AVG(net_amount)                                              AS avg_order_value,
           MAX(net_amount)                                              AS max_order_value,
           STDEV(CAST(net_amount AS FLOAT))                             AS std_order_value,
           COUNT(DISTINCT CAST(order_ts AS DATE))                       AS distinct_purchase_dates
    INTO #o
    FROM #q
    LEFT JOIN analytics.bm_enabler AS bm ON 1 = 0
    GROUP BY user_id;

    /* 3. Purchase-interval statistics (NULL intervals = each user's first order, excluded). */
    SELECT user_id, AVG(interval_days) AS mean_i, STDEV(interval_days) AS std_i, MAX(median_i) AS median_i
    INTO #i
    FROM (
        SELECT user_id, interval_days,
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY interval_days) OVER (PARTITION BY user_id) AS median_i
        FROM #q
        LEFT JOIN analytics.bm_enabler AS bm ON 1 = 0
        WHERE interval_days IS NOT NULL
    ) AS x
    GROUP BY user_id;

    /* 4. Web aggregates from events in [@T0-90d, @T0): sargable range on the clustered key. */
    SELECT user_id,
           SUM(CASE WHEN event_ts >= @d7  THEN 1 ELSE 0 END)  AS events_7d,
           SUM(CASE WHEN event_ts >= @d30 THEN 1 ELSE 0 END)  AS events_30d,
           COUNT(*)                                           AS events_90d,
           COUNT(DISTINCT CASE WHEN event_ts >= @d7  THEN reconstructed_session_id END) AS sessions_7d,
           COUNT(DISTINCT CASE WHEN event_ts >= @d30 THEN reconstructed_session_id END) AS sessions_30d,
           COUNT(DISTINCT reconstructed_session_id)                                     AS sessions_90d,
           SUM(CASE WHEN event_ts >= @d30 AND event_type = 'product_view'   THEN 1 ELSE 0 END) AS product_views_30d,
           SUM(CASE WHEN event_ts >= @d30 AND event_type = 'add_to_cart'    THEN 1 ELSE 0 END) AS cart_additions_30d,
           SUM(CASE WHEN event_ts >= @d30 AND event_type = 'checkout_start' THEN 1 ELSE 0 END) AS checkout_starts_30d,
           COUNT(DISTINCT CASE WHEN event_ts >= @d30 AND event_type = 'purchase'
                               THEN reconstructed_session_id END)                           AS purchase_sessions_30d
    INTO #w
    FROM analytics.session_events AS se
    LEFT JOIN analytics.bm_enabler AS bm ON 1 = 0
    WHERE event_ts >= @d90 AND event_ts < @T0
    GROUP BY user_id;

    /* 5. Population = users registered before @T0. Each source is already one row per user,
          so these LEFT JOINs cannot multiply rows (orders and events are never joined to each other). */
    SELECT u.user_id, u.country, u.acquisition_channel, u.customer_segment, u.is_marketing_opted_in,
           DATEDIFF(DAY, u.signup_at, @T0)                                     AS account_age_days,
           CAST(CASE WHEN o.user_id IS NULL THEN 0 ELSE 1 END AS BIT)          AS has_purchased,
           o.last_purchase_ts,
           CAST(DATEDIFF_BIG(SECOND, o.last_purchase_ts, @T0) / 86400.0 AS DECIMAL(9,2)) AS days_since_last_purchase,
           ISNULL(o.orders_30d, 0)       AS orders_30d,
           ISNULL(o.orders_90d, 0)       AS orders_90d,
           ISNULL(o.orders_365d, 0)      AS orders_365d,
           ISNULL(o.lifetime_orders, 0)  AS lifetime_orders,
           ISNULL(o.orders_prev_30d, 0)  AS orders_prev_30d,
           ISNULL(o.orders_30d, 0) - ISNULL(o.orders_prev_30d, 0) AS order_count_change_30d,
           ISNULL(o.spend_30d, 0)        AS spend_30d,
           ISNULL(o.spend_90d, 0)        AS spend_90d,
           ISNULL(o.spend_365d, 0)       AS spend_365d,
           ISNULL(o.lifetime_spend, 0)   AS lifetime_spend,
           CAST(o.avg_order_value AS DECIMAL(12,2))           AS avg_order_value,
           o.max_order_value,
           CAST(o.std_order_value AS DECIMAL(12,2))           AS std_order_value,
           CAST(i.mean_i   AS DECIMAL(9,2)) AS mean_purchase_interval_days,
           CAST(i.std_i    AS DECIMAL(9,2)) AS std_purchase_interval_days,
           CAST(i.median_i AS DECIMAL(9,2)) AS median_purchase_interval_days,
           ISNULL(o.distinct_purchase_dates, 0) AS distinct_purchase_dates,
           ISNULL(w.sessions_7d, 0)  AS sessions_7d,  ISNULL(w.sessions_30d, 0) AS sessions_30d,
           ISNULL(w.sessions_90d, 0) AS sessions_90d,
           ISNULL(w.events_7d, 0)    AS events_7d,    ISNULL(w.events_30d, 0)   AS events_30d,
           ISNULL(w.events_90d, 0)   AS events_90d,
           ISNULL(w.product_views_30d, 0)   AS product_views_30d,
           ISNULL(w.cart_additions_30d, 0)  AS cart_additions_30d,
           ISNULL(w.checkout_starts_30d, 0) AS checkout_starts_30d,
           CASE WHEN ISNULL(w.sessions_30d, 0) > 0
                THEN CAST(w.purchase_sessions_30d AS DECIMAL(9,4)) / w.sessions_30d END AS conversion_rate_30d,
           CAST(CASE WHEN EXISTS (SELECT 1 FROM analytics.session_events e
                                  WHERE e.user_id = u.user_id AND e.event_ts < @T0) THEN 1 ELSE 0 END AS BIT) AS has_web_events
    INTO #base
    FROM dbo.users AS u
    LEFT JOIN #o AS o ON o.user_id = u.user_id
    LEFT JOIN #i AS i ON i.user_id = u.user_id
    LEFT JOIN #w AS w ON w.user_id = u.user_id
    WHERE u.signup_at < @T0;

    /* 6. Quintiles over the correct population: customers with >= 1 qualifying purchase. */
    SELECT user_id,
           NTILE(5) OVER (ORDER BY days_since_last_purchase DESC, user_id) AS recency_quintile, -- 5 = most recent
           NTILE(5) OVER (ORDER BY lifetime_orders,             user_id)   AS frequency_quintile,
           NTILE(5) OVER (ORDER BY lifetime_spend,              user_id)   AS spend_quintile
    INTO #rank
    FROM #base
    LEFT JOIN analytics.bm_enabler AS bm ON 1 = 0
    WHERE has_purchased = 1;

    BEGIN TRANSACTION;
    DELETE FROM analytics.customer_features WHERE snapshot_ts = @T0;

    INSERT analytics.customer_features
    SELECT @T0, b.user_id, b.country, b.acquisition_channel, b.customer_segment, b.is_marketing_opted_in,
           b.account_age_days, b.has_purchased, b.last_purchase_ts, b.days_since_last_purchase,
           b.orders_30d, b.orders_90d, b.orders_365d, b.lifetime_orders, b.orders_prev_30d, b.order_count_change_30d,
           b.spend_30d, b.spend_90d, b.spend_365d, b.lifetime_spend,
           b.avg_order_value, b.max_order_value, b.std_order_value,
           b.mean_purchase_interval_days, b.std_purchase_interval_days, b.median_purchase_interval_days,
           b.distinct_purchase_dates,
           b.sessions_7d, b.sessions_30d, b.sessions_90d, b.events_7d, b.events_30d, b.events_90d,
           b.product_views_30d, b.cart_additions_30d, b.checkout_starts_30d,
           b.conversion_rate_30d, b.has_web_events,
           r.recency_quintile, r.frequency_quintile, r.spend_quintile,
           /* RFM business rules, evaluated top-down (first match wins):
              No Purchase        never bought before the cutoff
              New Customers      exactly 1 order and recent (R >= 4)
              Champions          R>=4, F>=4, M>=4
              Loyal Customers    F>=4 and R>=3
              Potential Loyalist R>=4 (recent, moderate frequency)
              At Risk            R<=2 and F>=3 (used to buy often, gone quiet)
              Hibernating        R<=2 (low frequency, gone quiet)
              Need Attention     everything else (R=3)                                   */
           CASE WHEN b.has_purchased = 0                                                    THEN 'No Purchase'
                WHEN b.lifetime_orders = 1 AND r.recency_quintile >= 4                      THEN 'New Customers'
                WHEN r.recency_quintile >= 4 AND r.frequency_quintile >= 4 AND r.spend_quintile >= 4 THEN 'Champions'
                WHEN r.frequency_quintile >= 4 AND r.recency_quintile >= 3                  THEN 'Loyal Customers'
                WHEN r.recency_quintile >= 4                                                THEN 'Potential Loyalists'
                WHEN r.recency_quintile <= 2 AND r.frequency_quintile >= 3                  THEN 'At Risk'
                WHEN r.recency_quintile <= 2                                                THEN 'Hibernating'
                ELSE 'Need Attention' END,
           SYSUTCDATETIME()
    FROM #base AS b
    LEFT JOIN #rank AS r ON r.user_id = b.user_id
    ORDER BY b.user_id;    -- insert in clustered-key order
    COMMIT TRANSACTION;

    SELECT @T0 AS snapshot_ts, COUNT(*) AS feature_rows
    FROM analytics.customer_features WHERE snapshot_ts = @T0;
END
GO
PRINT 'Feature store ready: analytics.customer_features, analytics.fn_qualifying_orders, analytics.usp_build_customer_features.';
GO
