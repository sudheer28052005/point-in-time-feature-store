/* =============================================================================
   09_churn_labels.sql
   30-day inactivity labels. This is an OPERATIONAL inactivity definition, not proof of
   permanent churn: a customer labelled 1 may buy on day 31.

   CONTRACT for a snapshot @T0
     feature history : events/orders with ts <  @T0                (07_customer_features.sql)
     outcome window  : [@T0, @T0 + @outcome_days days)            half-open, exactly 30 days
     churn_label = 1 : NO qualifying purchase inside the outcome window
     churn_label = 0 : >= 1 qualifying purchase inside the outcome window
     "qualifying" is evaluated at the END of the window (@T0 + 30d): an order that is
     refunded before the window closes does not count as retention; one refunded later does.

   COMPLETE WINDOWS ONLY: a label is generated only if @T0 + @outcome_days <= @observation_end.
     Otherwise the procedure raises an error and writes nothing (no partial/optimistic labels).

   POPULATION: users with signup_at < @T0 (users who register later are excluded). Users with
     no prior purchase ARE labelled (churn_label = 1 means "did not buy in the next 30 days")
     and flagged has_prior_purchase = 0. Because "will a prospect ever convert" is a different
     problem from "will a customer lapse", the ML training set uses has_prior_purchase = 1 by
     default (documented in the README); the flag lets that be changed without regenerating.

   LEAKAGE: this table is never joined into feature computation. Features are in a separate
   table keyed by the same (snapshot_ts, user_id); they meet only in analytics.vw_training_set.
   ============================================================================= */
SET NOCOUNT ON;
USE CustomerFeatureStore;
GO

IF OBJECT_ID(N'analytics.churn_labels', N'U') IS NULL
BEGIN
    CREATE TABLE analytics.churn_labels
    (
        snapshot_ts                 DATETIME2(3) NOT NULL,
        user_id                     BIGINT       NOT NULL,
        outcome_window_start        DATETIME2(3) NOT NULL,   -- inclusive (= snapshot_ts)
        outcome_window_end          DATETIME2(3) NOT NULL,   -- exclusive
        qualifying_orders_in_window INT          NOT NULL,
        churn_label                 BIT          NOT NULL,
        has_prior_purchase          BIT          NOT NULL,
        observation_end             DATETIME2(3) NOT NULL,   -- data completeness boundary used for this build
        label_generated_at          DATETIME2(3) NOT NULL,
        CONSTRAINT PK_churn_labels PRIMARY KEY CLUSTERED (snapshot_ts, user_id)
    );
END
GO

CREATE OR ALTER PROCEDURE analytics.usp_build_churn_labels
    @T0               DATETIME2(3),
    @observation_end  DATETIME2(3),
    @outcome_days     INT = 30,
    @replace          BIT = 1
AS
BEGIN
    SET NOCOUNT ON;
    SET XACT_ABORT ON;
    IF @T0 IS NULL OR @observation_end IS NULL THROW 50010, '@T0 and @observation_end are required.', 1;
    IF @outcome_days < 1 THROW 50011, '@outcome_days must be >= 1.', 1;

    DECLARE @window_end DATETIME2(3) = DATEADD(DAY, @outcome_days, @T0);
    IF @window_end > @observation_end
        THROW 50012, 'Outcome window is incomplete (T0 + outcome_days > observation_end): label not generated.', 1;
    IF @replace = 0 AND EXISTS (SELECT 1 FROM analytics.churn_labels WHERE snapshot_ts = @T0)
        THROW 50013, 'Labels for this snapshot already exist; pass @replace = 1.', 1;

    BEGIN TRANSACTION;
    DELETE FROM analytics.churn_labels WHERE snapshot_ts = @T0;

    INSERT analytics.churn_labels
        (snapshot_ts, user_id, outcome_window_start, outcome_window_end, qualifying_orders_in_window,
         churn_label, has_prior_purchase, observation_end, label_generated_at)
    SELECT @T0, u.user_id, @T0, @window_end,
           ISNULL(w.n, 0),
           CAST(CASE WHEN ISNULL(w.n, 0) = 0 THEN 1 ELSE 0 END AS BIT),
           CAST(CASE WHEN p.user_id IS NULL THEN 0 ELSE 1 END AS BIT),
           @observation_end, SYSUTCDATETIME()
    FROM dbo.users AS u
    LEFT JOIN (SELECT user_id, COUNT(*) AS n
               FROM analytics.fn_qualifying_orders(@T0, @window_end, @window_end)
               GROUP BY user_id) AS w ON w.user_id = u.user_id
    LEFT JOIN (SELECT DISTINCT user_id
               FROM analytics.fn_qualifying_orders('1900-01-01', @T0, @T0)) AS p ON p.user_id = u.user_id
    WHERE u.signup_at < @T0
    ORDER BY u.user_id;
    COMMIT TRANSACTION;

    SELECT @T0 AS snapshot_ts, COUNT(*) AS label_rows,
           AVG(CAST(churn_label AS FLOAT)) AS churn_rate_all,
           AVG(CASE WHEN has_prior_purchase = 1 THEN CAST(churn_label AS FLOAT) END) AS churn_rate_prior_purchasers
    FROM analytics.churn_labels WHERE snapshot_ts = @T0;
END
GO
PRINT 'Labels ready: analytics.churn_labels, analytics.usp_build_churn_labels.';
GO
