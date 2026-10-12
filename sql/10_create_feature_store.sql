/* =============================================================================
   10_create_feature_store.sql
   Orchestration + the model-facing view.

   analytics.usp_build_feature_snapshot  one cutoff: features (+ labels when the window is complete)
   analytics.usp_build_feature_store     many cutoffs: monthly grid from @first_cutoff to @last_cutoff
   analytics.vw_training_set             features JOIN labels, ONLY for snapshots with complete labels

   The feature table and the label table are separate; the join happens only here, at read time,
   on (snapshot_ts, user_id). Features are never computed from labels or from post-cutoff data.
   ============================================================================= */
SET NOCOUNT ON;
USE CustomerFeatureStore;
GO

/* Model scores written by scripts/evaluate_model.py (read by the dashboard).
     partition 'test'    : labelled held-out snapshots; actual_churn is the real outcome
     partition 'scoring' : latest snapshot, features only; actual_churn is NULL (outcome not yet observed)
   churn_probability is the CALIBRATED probability; predicted_churn applies the validation-chosen threshold. */
IF OBJECT_ID(N'analytics.churn_predictions', N'U') IS NULL
    CREATE TABLE analytics.churn_predictions
    (
        model_version     VARCHAR(60)  NOT NULL,
        partition         VARCHAR(10)  NOT NULL,
        snapshot_ts       DATETIME2(3) NOT NULL,
        user_id           BIGINT       NOT NULL,
        churn_probability DECIMAL(7,6) NOT NULL,
        risk_decile       TINYINT      NOT NULL,      -- 10 = highest-risk 10% within (model_version, partition, snapshot)
        predicted_churn   BIT          NOT NULL,
        actual_churn      BIT          NULL,
        scored_at         DATETIME2(3) NOT NULL CONSTRAINT DF_churn_predictions_scored DEFAULT (SYSUTCDATETIME()),
        CONSTRAINT PK_churn_predictions PRIMARY KEY CLUSTERED (model_version, partition, snapshot_ts, user_id)
    );
GO

CREATE OR ALTER PROCEDURE analytics.usp_build_feature_snapshot
    @T0              DATETIME2(3),
    @observation_end DATETIME2(3),
    @outcome_days    INT = 30
AS
BEGIN
    SET NOCOUNT ON;
    EXEC analytics.usp_build_customer_features @T0 = @T0, @replace = 1;
    IF DATEADD(DAY, @outcome_days, @T0) <= @observation_end
        EXEC analytics.usp_build_churn_labels @T0 = @T0, @observation_end = @observation_end,
                                              @outcome_days = @outcome_days, @replace = 1;
    ELSE
    BEGIN
        -- Scoring-only snapshot: no outcome yet. Remove any stale labels so none can exist for it.
        DELETE FROM analytics.churn_labels WHERE snapshot_ts = @T0;
        PRINT CONCAT('Snapshot ', CONVERT(VARCHAR(23), @T0, 121), ': outcome window incomplete -> features only, no labels.');
    END
END
GO

CREATE OR ALTER PROCEDURE analytics.usp_build_feature_store
    @first_cutoff    DATETIME2(3),
    @last_cutoff     DATETIME2(3),
    @observation_end DATETIME2(3),
    @step_months     INT = 1,
    @outcome_days    INT = 30
AS
BEGIN
    SET NOCOUNT ON;
    IF @step_months < 1 THROW 50030, '@step_months must be >= 1.', 1;
    DECLARE @t DATETIME2(3) = @first_cutoff;
    WHILE @t <= @last_cutoff
    BEGIN
        EXEC analytics.usp_build_feature_snapshot @T0 = @t, @observation_end = @observation_end,
                                                  @outcome_days = @outcome_days;
        SET @t = DATEADD(MONTH, @step_months, @t);
    END
END
GO

/* in_training_population = the customer bought at least once in the 90 days before the cutoff.
   Rationale: customers silent for 90+ days lapse ~97% of the time, so including them makes the
   task trivially easy and inflates metrics. The rule uses only information available at the
   cutoff (orders_90d), so it cannot leak the outcome. Train/evaluate on in_training_population = 1
   (primary experiment); the full has_prior_purchase population is a secondary experiment. */
CREATE OR ALTER VIEW analytics.vw_training_set
AS
SELECT f.*,
       CAST(CASE WHEN f.orders_90d > 0 THEN 1 ELSE 0 END AS BIT) AS in_training_population,
       l.churn_label,
       l.has_prior_purchase,
       l.outcome_window_end,
       l.qualifying_orders_in_window       -- outcome detail: for analysis only, NEVER a model input
FROM analytics.customer_features AS f
JOIN analytics.churn_labels      AS l
  ON l.snapshot_ts = f.snapshot_ts AND l.user_id = f.user_id;
GO
PRINT 'Orchestration ready: usp_build_feature_snapshot, usp_build_feature_store, vw_training_set.';
GO
