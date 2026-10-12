/* =============================================================================
   06_sessionization.sql
   Reconstructs web sessions from inactivity gaps.

   RULE (used everywhere in the project)
     A new session starts at a user's first event, and at any event whose gap to
     that user's PREVIOUS event is STRICTLY GREATER than @inactivity_minutes
     (default 30). A gap of exactly 30:00.000 therefore stays in the SAME session.
     Gaps are measured in milliseconds (event_ts is DATETIME2(3)).
   ORDER: (event_ts, event_id). event_id is the deterministic tie-breaker for
     simultaneous events, so the numbering is identical on every run.
   source_session_id (client-provided) is intentionally ignored: it is reused
     across visits and spans idle gaps (see 03_create_tables.sql).

   IDs: session_seq = 1,2,3.. per user; reconstructed_session_id =
        user_id * 1,000,000 + session_seq   (globally unique, deterministic, human-decodable;
        the procedure fails if any user would exceed 999,999 sessions).

   POINT-IN-TIME SAFETY: session_seq of an event depends only on that event and
   EARLIER events, so numbering events before a cutoff gives the same ids as numbering
   the full history. Features therefore filter analytics.session_events by event_ts < @T0
   and never read session aggregates that could include post-cutoff events.

   session_end_ts = timestamp of the LAST event in the session (the 30-minute
   timeout is not added). device_type = device of the FIRST event of the session;
   distinct_device_count reports sessions that switched device.
   ============================================================================= */
SET NOCOUNT ON;
USE CustomerFeatureStore;
GO

/* ---------------------------------------------------------------------------
   BATCH-MODE ENABLER (performance; see sql/12 + reports/performance_report.md)
   Window functions and DISTINCT aggregates over millions of rows are slow in row mode
   (measured: sessionization core 7.7 s row mode vs 1.35 s batch mode, identical results).
   On this edition/compatibility level the optimizer does not choose batch mode on rowstore
   by itself, so queries LEFT JOIN an EMPTY table that has a columnstore index
   (`LEFT JOIN analytics.bm_enabler ON 1 = 0`): the join never produces a row, but it
   makes the plan eligible for batch mode. analytics.bm_enabler is a SYNONYM, so batch mode
   can be switched off (rowstore twin) for before/after benchmarks:
       EXEC analytics.usp_set_batch_mode @enabled = 0;   -- row mode
       EXEC analytics.usp_set_batch_mode @enabled = 1;   -- batch mode (default)
   --------------------------------------------------------------------------- */
IF OBJECT_ID(N'analytics.bm_rowstore_empty', N'U') IS NULL
    CREATE TABLE analytics.bm_rowstore_empty (bm_dummy INT NOT NULL);
GO
IF OBJECT_ID(N'analytics.bm_columnstore_empty', N'U') IS NULL
    CREATE TABLE analytics.bm_columnstore_empty (bm_dummy INT NOT NULL, INDEX CCI_bm_columnstore_empty CLUSTERED COLUMNSTORE);
GO
CREATE OR ALTER PROCEDURE analytics.usp_set_batch_mode @enabled BIT
AS
BEGIN
    SET NOCOUNT ON;
    IF OBJECT_ID(N'analytics.bm_enabler', N'SN') IS NOT NULL DROP SYNONYM analytics.bm_enabler;
    IF @enabled = 1 CREATE SYNONYM analytics.bm_enabler FOR analytics.bm_columnstore_empty;
    ELSE            CREATE SYNONYM analytics.bm_enabler FOR analytics.bm_rowstore_empty;
END
GO
IF OBJECT_ID(N'analytics.bm_enabler', N'SN') IS NULL EXEC analytics.usp_set_batch_mode @enabled = 1;
GO

IF OBJECT_ID(N'analytics.session_events', N'U') IS NULL
BEGIN
    CREATE TABLE analytics.session_events
    (
        event_id                BIGINT       NOT NULL,
        user_id                 BIGINT       NOT NULL,
        event_ts                DATETIME2(3) NOT NULL,
        event_type              VARCHAR(30)  NOT NULL,
        device_type             VARCHAR(20)  NOT NULL,
        session_seq             INT          NOT NULL,
        reconstructed_session_id BIGINT      NOT NULL,
        CONSTRAINT PK_session_events PRIMARY KEY NONCLUSTERED (event_id)
    );
    CREATE UNIQUE CLUSTERED INDEX CIX_session_events_user_ts
        ON analytics.session_events (user_id, event_ts, event_id);
END
GO

IF OBJECT_ID(N'analytics.sessions', N'U') IS NULL
BEGIN
    CREATE TABLE analytics.sessions
    (
        user_id                  BIGINT       NOT NULL,
        session_seq              INT          NOT NULL,
        reconstructed_session_id BIGINT       NOT NULL,
        session_start_ts         DATETIME2(3) NOT NULL,
        session_end_ts           DATETIME2(3) NOT NULL,
        event_count              INT          NOT NULL,
        duration_seconds         INT          NOT NULL,
        page_view_count          INT          NOT NULL,
        product_view_count       INT          NOT NULL,
        add_to_cart_count        INT          NOT NULL,
        checkout_started_count   INT          NOT NULL,
        purchase_event_count     INT          NOT NULL,
        device_type              VARCHAR(20)  NOT NULL,
        distinct_device_count    INT          NOT NULL,
        CONSTRAINT PK_sessions PRIMARY KEY CLUSTERED (user_id, session_seq),
        CONSTRAINT UQ_sessions_id UNIQUE (reconstructed_session_id)
    );
END
GO

CREATE OR ALTER PROCEDURE analytics.usp_build_sessions
    @inactivity_minutes INT = 30
AS
BEGIN
    SET NOCOUNT ON;
    SET XACT_ABORT ON;
    IF @inactivity_minutes IS NULL OR @inactivity_minutes < 1
        THROW 50001, '@inactivity_minutes must be a positive integer.', 1;

    DECLARE @gap_ms BIGINT = CAST(@inactivity_minutes AS BIGINT) * 60000;

    BEGIN TRANSACTION;
    TRUNCATE TABLE analytics.sessions;
    TRUNCATE TABLE analytics.session_events;

    ;WITH ordered AS
    (
        SELECT event_id, user_id, event_ts, event_type, device_type,
               LAG(event_ts) OVER (PARTITION BY user_id ORDER BY event_ts, event_id) AS prev_ts
        FROM dbo.raw_web_events AS e
        LEFT JOIN analytics.bm_enabler AS bm ON 1 = 0          -- batch-mode enabler, adds no rows
    ),
    flagged AS
    (
        SELECT *,
               CASE WHEN prev_ts IS NULL
                      OR DATEDIFF_BIG(MILLISECOND, prev_ts, event_ts) > @gap_ms   -- strictly greater
                    THEN 1 ELSE 0 END AS is_new_session
        FROM ordered
    ),
    numbered AS
    (
        SELECT *,
               SUM(is_new_session) OVER (PARTITION BY user_id ORDER BY event_ts, event_id
                                         ROWS UNBOUNDED PRECEDING) AS session_seq
        FROM flagged
    )
    INSERT analytics.session_events
        (event_id, user_id, event_ts, event_type, device_type, session_seq, reconstructed_session_id)
    SELECT event_id, user_id, event_ts, event_type, device_type, session_seq,
           user_id * 1000000 + session_seq
    FROM numbered;

    IF EXISTS (SELECT 1 FROM analytics.session_events WHERE session_seq >= 1000000)
        THROW 50002, 'A user has >= 1,000,000 sessions; reconstructed_session_id encoding would collide.', 1;

    INSERT analytics.sessions
        (user_id, session_seq, reconstructed_session_id, session_start_ts, session_end_ts, event_count,
         duration_seconds, page_view_count, product_view_count, add_to_cart_count, checkout_started_count,
         purchase_event_count, device_type, distinct_device_count)
    SELECT user_id, session_seq, MAX(reconstructed_session_id),
           MIN(event_ts), MAX(event_ts), COUNT(*),
           CAST(DATEDIFF_BIG(SECOND, MIN(event_ts), MAX(event_ts)) AS INT),
           SUM(CASE WHEN event_type = 'page_view'      THEN 1 ELSE 0 END),
           SUM(CASE WHEN event_type = 'product_view'   THEN 1 ELSE 0 END),
           SUM(CASE WHEN event_type = 'add_to_cart'    THEN 1 ELSE 0 END),
           SUM(CASE WHEN event_type = 'checkout_start' THEN 1 ELSE 0 END),
           SUM(CASE WHEN event_type = 'purchase'       THEN 1 ELSE 0 END),
           MAX(first_device),
           COUNT(DISTINCT device_type)
    FROM (
        SELECT se.*, FIRST_VALUE(se.device_type) OVER (PARTITION BY se.user_id, se.session_seq
                                                       ORDER BY se.event_ts, se.event_id) AS first_device
        FROM analytics.session_events AS se
        LEFT JOIN analytics.bm_enabler AS bm ON 1 = 0
    ) AS x
    GROUP BY user_id, session_seq;

    COMMIT TRANSACTION;

    SELECT (SELECT COUNT_BIG(*) FROM analytics.session_events) AS session_event_rows,
           (SELECT COUNT_BIG(*) FROM analytics.sessions)       AS session_rows;
END
GO
PRINT 'Sessionization ready: analytics.session_events, analytics.sessions, analytics.usp_build_sessions.';
GO
