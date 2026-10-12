/* =============================================================================
   03_create_tables.sql
   Raw source tables. Idempotent (IF OBJECT_ID ... IS NULL). Constraints beyond
   the primary key, plus defaults, are added in 04_create_constraints.sql.

   KEY DESIGN DECISIONS
   * All timestamps are UTC, DATETIME2(3) (millisecond precision is enough for
     clickstream ordering; event_id is the tie-breaker for identical timestamps).
   * Physical layout: the PRIMARY KEY on each fact table is NONCLUSTERED on the
     surrogate id, and the CLUSTERED index is (user_id, <timestamp>, <id>).
     Why: nearly every analytical query here (sessionization, RFM, point-in-time
     windows) reads one user's history in time order, so clustering by
     (user_id, ts) turns those into range seeks / ordered scans with no sort.
     Phase 5 benchmarks this choice against alternatives; it is a hypothesis
     about the workload, not dogma. A surrogate-id clustered key would give
     cheaper append-only inserts but forces sorts/lookups for per-user work.
   * Only the indexes needed for correctness/integrity are created here.
     Workload-driven indexes are added in 11_create_indexes.sql after the
     baseline is measured.
   ============================================================================= */
SET NOCOUNT ON;
USE CustomerFeatureStore;
GO

/* ---------------------------------------------------------------- dbo.users */
IF OBJECT_ID(N'dbo.users', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.users
    (
        user_id               BIGINT        NOT NULL,
        signup_at             DATETIME2(3)  NOT NULL,   -- UTC
        country               CHAR(2)       NOT NULL,   -- ISO 3166-1 alpha-2
        acquisition_channel   VARCHAR(30)   NOT NULL,
        customer_segment      VARCHAR(20)   NOT NULL,
        date_of_birth         DATE          NULL,       -- optional; PII-like, synthetic only
        is_marketing_opted_in BIT           NOT NULL,
        created_at            DATETIME2(3)  NOT NULL,   -- row load time (UTC)
        CONSTRAINT PK_users PRIMARY KEY CLUSTERED (user_id)
    );
END
GO

/* --------------------------------------------------------------- dbo.orders
   Refund / cancellation policy (used by every downstream object):
     order_status = 'completed'  -> qualifying
     order_status = 'cancelled'  -> NEVER qualifying (cancelled before fulfilment;
                                    the status is known at order time, so no leakage)
     order_status = 'refunded'   -> full refund of the order. It qualifies as revenue
                                    ONLY for cutoffs before refunded_at, because at
                                    those times the refund had not yet happened.
                                    Using current status alone would leak the future.
   Refunds are modelled as full-order reversals, so a refunded order is counted
   once (while it qualifies) and never negatively -> no double counting.
   refunded_at is therefore an addition to the minimum column list.            */
IF OBJECT_ID(N'dbo.orders', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.orders
    (
        order_id        BIGINT         NOT NULL,
        user_id         BIGINT         NOT NULL,
        order_ts        DATETIME2(3)   NOT NULL,   -- UTC
        order_status    VARCHAR(20)    NOT NULL,
        gross_amount    DECIMAL(12,2)  NOT NULL,
        discount_amount DECIMAL(12,2)  NOT NULL,
        -- Derived value is computed by the engine, so it cannot drift from its inputs.
        net_amount      AS (gross_amount - discount_amount) PERSISTED,
        payment_method  VARCHAR(20)    NOT NULL,
        refunded_at     DATETIME2(3)   NULL,       -- UTC; set only when order_status = 'refunded'
        created_at      DATETIME2(3)   NOT NULL,   -- row load time (UTC)
        CONSTRAINT PK_orders PRIMARY KEY NONCLUSTERED (order_id)
    );
    CREATE UNIQUE CLUSTERED INDEX CIX_orders_user_ts
        ON dbo.orders (user_id, order_ts, order_id);
END
GO

/* ------------------------------------------------------- dbo.raw_web_events
   source_session_id is the identifier emitted by the tracking client (e.g. a
   cookie/GUID). It is NOT trusted for analysis: clients reuse or reset it
   arbitrarily (cookie expiry, multi-tab, app backgrounding), so one source id can
   span days and one real visit can carry several ids. Analytics uses the
   reconstructed_session_id derived from inactivity gaps (06_sessionization.sql).
   page_url is NVARCHAR(MAX) per spec: it cannot be an index key and large values
   may be stored off-row, so it is never included in the analytical indexes.   */
IF OBJECT_ID(N'dbo.raw_web_events', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.raw_web_events
    (
        event_id          BIGINT           NOT NULL,
        user_id           BIGINT           NOT NULL,
        event_ts          DATETIME2(3)     NOT NULL,   -- UTC
        source_session_id UNIQUEIDENTIFIER NOT NULL,   -- client-provided; see note above
        event_type        VARCHAR(30)      NOT NULL,
        page_url          NVARCHAR(MAX)    NULL,
        device_type       VARCHAR(20)      NOT NULL,
        referrer_type     VARCHAR(30)      NULL,
        created_at        DATETIME2(3)     NOT NULL,   -- row load time (UTC)
        CONSTRAINT PK_raw_web_events PRIMARY KEY NONCLUSTERED (event_id)
    );
    CREATE UNIQUE CLUSTERED INDEX CIX_raw_web_events_user_ts
        ON dbo.raw_web_events (user_id, event_ts, event_id);
END
GO

PRINT 'Tables ready: dbo.users, dbo.orders, dbo.raw_web_events.';
GO
