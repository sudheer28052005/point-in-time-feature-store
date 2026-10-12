/* =============================================================================
   08_retention_cohorts.sql
   Monthly acquisition-cohort retention.

   DEFINITIONS
     cohort_month   calendar month of users.signup_at (first day of month, UTC).
     activity       PRIMARY  ('purchase'): the user has >= 1 qualifying order in the calendar
                             month (qualifying = analytics.fn_qualifying_orders evaluated at
                             @observation_end; this is a retrospective report, so refunds that
                             happened before the end of observation do remove the order).
                    SECONDARY ('web'): >= 1 product_view / add_to_cart / checkout_start / purchase
                             event in the month (page_view alone is not "meaningful").
     month index    DATEDIFF(MONTH, cohort_month, activity_month); month 0 = signup month
                    (a partial month for most users, as is standard for calendar cohorts).
     cohort size    ALL users who signed up that month (the denominator never shrinks).
     OBSERVED       a (cohort, month index) cell is observed only if that calendar month ENDS on or
                    before @observation_end. The month containing @observation_end is partial and is
                    never reported. Observed cells with no activity are a TRUE 0; unobserved
                    (future) cells are NULL, never 0.

   OUTPUT  analytics.cohort_retention_long  (BI-friendly long format, one row per cohort x month index
           for the full triangle + the unobserved cells flagged is_observed = 0)
           analytics.usp_get_cohort_retention_matrix  (pivoted triangular matrix for reports)

   PERFORMANCE: orders/events are read ONCE into #activity as distinct (user, month) pairs,
   then everything else is aggregation over that small table.
   ============================================================================= */
SET NOCOUNT ON;
USE CustomerFeatureStore;
GO

IF OBJECT_ID(N'analytics.cohort_retention_long', N'U') IS NULL
BEGIN
    CREATE TABLE analytics.cohort_retention_long
    (
        activity_definition      VARCHAR(20)  NOT NULL,   -- 'purchase' | 'web'
        cohort_month             DATE         NOT NULL,
        months_since_acquisition INT          NOT NULL,
        cohort_size              INT          NOT NULL,
        active_customers         INT          NULL,       -- NULL when the month is not yet observed
        retention_pct            DECIMAL(5,2) NULL,       -- 100 * active / cohort_size; NULL when unobserved
        is_observed              BIT          NOT NULL,
        observation_end          DATETIME2(3) NOT NULL,
        CONSTRAINT PK_cohort_retention_long
            PRIMARY KEY CLUSTERED (activity_definition, cohort_month, months_since_acquisition)
    );
END
GO

CREATE OR ALTER PROCEDURE analytics.usp_build_cohort_retention
    @observation_end     DATETIME2(3),
    @activity_definition VARCHAR(20) = 'purchase'
AS
BEGIN
    SET NOCOUNT ON;
    SET XACT_ABORT ON;
    IF @activity_definition NOT IN ('purchase', 'web')
        THROW 50020, 'activity_definition must be purchase or web.', 1;

    -- First day of the last month that is completely observed.
    DECLARE @last_month DATE = DATEADD(MONTH, -1, DATEFROMPARTS(YEAR(@observation_end), MONTH(@observation_end), 1));
    DECLARE @activity_end DATETIME2(3) = DATEADD(MONTH, 1, @last_month);   -- exclusive upper bound for activity

    DROP TABLE IF EXISTS #cohort, #activity;

    SELECT user_id, DATEFROMPARTS(YEAR(signup_at), MONTH(signup_at), 1) AS cohort_month
    INTO #cohort
    FROM dbo.users
    WHERE signup_at < @activity_end;     -- cohorts in the partial month are excluded

    CREATE TABLE #activity (user_id BIGINT NOT NULL, activity_month DATE NOT NULL, PRIMARY KEY (user_id, activity_month));

    IF @activity_definition = 'purchase'
        INSERT #activity (user_id, activity_month)
        SELECT DISTINCT user_id, DATEFROMPARTS(YEAR(order_ts), MONTH(order_ts), 1)
        FROM analytics.fn_qualifying_orders('1900-01-01', @activity_end, @observation_end);
    ELSE
        INSERT #activity (user_id, activity_month)
        SELECT DISTINCT user_id, DATEFROMPARTS(YEAR(event_ts), MONTH(event_ts), 1)
        FROM dbo.raw_web_events
        WHERE event_ts < @activity_end
          AND event_type IN ('product_view', 'add_to_cart', 'checkout_start', 'purchase');

    DECLARE @max_idx INT = (SELECT DATEDIFF(MONTH, MIN(cohort_month), @last_month) FROM #cohort);

    BEGIN TRANSACTION;
    DELETE FROM analytics.cohort_retention_long WHERE activity_definition = @activity_definition;

    ;WITH sizes AS
    (
        SELECT cohort_month, COUNT(*) AS cohort_size
        FROM #cohort
        GROUP BY cohort_month
    ),
    active AS
    (
        SELECT c.cohort_month,
               DATEDIFF(MONTH, c.cohort_month, a.activity_month) AS idx,
               COUNT(*) AS active_customers                     -- (user, month) pairs are already distinct
        FROM #cohort AS c
        JOIN #activity AS a ON a.user_id = c.user_id AND a.activity_month >= c.cohort_month
        GROUP BY c.cohort_month, DATEDIFF(MONTH, c.cohort_month, a.activity_month)
    ),
    grid AS
    (
        SELECT s.cohort_month, s.cohort_size, CAST(g.value AS INT) AS idx,
               CAST(CASE WHEN DATEADD(MONTH, CAST(g.value AS INT), s.cohort_month) <= @last_month
                         THEN 1 ELSE 0 END AS BIT) AS is_observed
        FROM sizes AS s
        CROSS JOIN GENERATE_SERIES(0, @max_idx) AS g
    )
    INSERT analytics.cohort_retention_long
        (activity_definition, cohort_month, months_since_acquisition, cohort_size,
         active_customers, retention_pct, is_observed, observation_end)
    SELECT @activity_definition, g.cohort_month, g.idx, g.cohort_size,
           CASE WHEN g.is_observed = 1 THEN ISNULL(a.active_customers, 0) END,
           CASE WHEN g.is_observed = 1
                THEN CAST(100.0 * ISNULL(a.active_customers, 0) / g.cohort_size AS DECIMAL(5,2)) END,
           g.is_observed, @observation_end
    FROM grid AS g
    LEFT JOIN active AS a ON a.cohort_month = g.cohort_month AND a.idx = g.idx;
    COMMIT TRANSACTION;

    SELECT COUNT(*) AS rows_written, SUM(CAST(is_observed AS INT)) AS observed_cells
    FROM analytics.cohort_retention_long WHERE activity_definition = @activity_definition;
END
GO

/* Pivoted triangular matrix: rows = cohort, columns M0..Mn.
   NULL = not yet observed, 0.00 = observed with no active customers.
   The dynamic SQL only embeds integers from the table via QUOTENAME; the activity
   definition is passed as a parameter. */
CREATE OR ALTER PROCEDURE analytics.usp_get_cohort_retention_matrix
    @activity_definition VARCHAR(20) = 'purchase',
    @metric              VARCHAR(20) = 'retention_pct'     -- retention_pct | active_customers
AS
BEGIN
    SET NOCOUNT ON;
    IF @metric NOT IN ('retention_pct', 'active_customers')
        THROW 50021, 'metric must be retention_pct or active_customers.', 1;

    DECLARE @cols NVARCHAR(MAX) =
    (
        SELECT STRING_AGG(CAST(QUOTENAME('M' + CAST(v AS VARCHAR(3))) AS NVARCHAR(MAX)), N',') WITHIN GROUP (ORDER BY v)
        FROM (SELECT DISTINCT months_since_acquisition AS v
              FROM analytics.cohort_retention_long WHERE activity_definition = @activity_definition) AS x
    );
    IF @cols IS NULL THROW 50022, 'No retention data; run usp_build_cohort_retention first.', 1;

    DECLARE @sql NVARCHAR(MAX) = N'
        SELECT cohort_month, cohort_size, ' + @cols + N'
        FROM (SELECT cohort_month, cohort_size, ''M'' + CAST(months_since_acquisition AS VARCHAR(3)) AS m, '
             + QUOTENAME(@metric) + N' AS val
              FROM analytics.cohort_retention_long WHERE activity_definition = @def) AS src
        PIVOT (MAX(val) FOR m IN (' + @cols + N')) AS p
        ORDER BY cohort_month;';
    EXEC sys.sp_executesql @sql, N'@def VARCHAR(20)', @def = @activity_definition;
END
GO
PRINT 'Cohorts ready: analytics.cohort_retention_long, usp_build_cohort_retention, usp_get_cohort_retention_matrix.';
GO
