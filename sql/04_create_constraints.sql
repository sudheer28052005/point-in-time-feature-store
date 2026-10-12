/* =============================================================================
   04_create_constraints.sql
   Defaults, CHECK constraints and foreign keys. Idempotent: each object is
   created only if OBJECT_ID() says it does not exist.

   Rules that SQL Server cannot express as a column/row CHECK (cross-table):
     * orders.order_ts      >= users.signup_at
     * raw_web_events.event_ts >= users.signup_at
   These are verified by sql/13_validation_queries.sql and tests/test_data_quality.py
   after every load, rather than via triggers (which would slow bulk loads).

   Allowed-value lists below are the single source of truth; the Python data
   generator imports the same vocabulary.
   ============================================================================= */
SET NOCOUNT ON;
USE CustomerFeatureStore;
GO

/* ---------------------------------------------------------------- defaults */
IF OBJECT_ID(N'dbo.DF_users_created_at', N'D') IS NULL
    ALTER TABLE dbo.users ADD CONSTRAINT DF_users_created_at DEFAULT (SYSUTCDATETIME()) FOR created_at;
IF OBJECT_ID(N'dbo.DF_orders_created_at', N'D') IS NULL
    ALTER TABLE dbo.orders ADD CONSTRAINT DF_orders_created_at DEFAULT (SYSUTCDATETIME()) FOR created_at;
IF OBJECT_ID(N'dbo.DF_raw_web_events_created_at', N'D') IS NULL
    ALTER TABLE dbo.raw_web_events ADD CONSTRAINT DF_raw_web_events_created_at DEFAULT (SYSUTCDATETIME()) FOR created_at;
GO

/* ------------------------------------------------------------------- users */
-- Binary collation makes the [A-Z] range genuinely uppercase-only.
IF OBJECT_ID(N'dbo.CK_users_country', N'C') IS NULL
    ALTER TABLE dbo.users ADD CONSTRAINT CK_users_country
        CHECK (country COLLATE Latin1_General_100_BIN2 LIKE '[A-Z][A-Z]');

IF OBJECT_ID(N'dbo.CK_users_acquisition_channel', N'C') IS NULL
    ALTER TABLE dbo.users ADD CONSTRAINT CK_users_acquisition_channel
        CHECK (acquisition_channel IN ('organic','paid_search','social','email','referral','affiliate'));

IF OBJECT_ID(N'dbo.CK_users_customer_segment', N'C') IS NULL
    ALTER TABLE dbo.users ADD CONSTRAINT CK_users_customer_segment
        CHECK (customer_segment IN ('budget','standard','premium','business'));

IF OBJECT_ID(N'dbo.CK_users_date_of_birth', N'C') IS NULL
    ALTER TABLE dbo.users ADD CONSTRAINT CK_users_date_of_birth
        CHECK (date_of_birth IS NULL OR (date_of_birth >= '1900-01-01' AND date_of_birth < CAST(signup_at AS DATE)));
GO

/* ------------------------------------------------------------------ orders */
IF OBJECT_ID(N'dbo.FK_orders_users', N'F') IS NULL
    ALTER TABLE dbo.orders ADD CONSTRAINT FK_orders_users
        FOREIGN KEY (user_id) REFERENCES dbo.users (user_id);

IF OBJECT_ID(N'dbo.CK_orders_status', N'C') IS NULL
    ALTER TABLE dbo.orders ADD CONSTRAINT CK_orders_status
        CHECK (order_status IN ('completed','cancelled','refunded'));

IF OBJECT_ID(N'dbo.CK_orders_amounts', N'C') IS NULL
    ALTER TABLE dbo.orders ADD CONSTRAINT CK_orders_amounts
        CHECK (gross_amount >= 0 AND discount_amount >= 0 AND discount_amount <= gross_amount);

IF OBJECT_ID(N'dbo.CK_orders_payment_method', N'C') IS NULL
    ALTER TABLE dbo.orders ADD CONSTRAINT CK_orders_payment_method
        CHECK (payment_method IN ('card','paypal','bank_transfer','gift_card','wallet'));

-- refunded_at is present exactly when the order is refunded, and cannot precede the order.
IF OBJECT_ID(N'dbo.CK_orders_refund_consistency', N'C') IS NULL
    ALTER TABLE dbo.orders ADD CONSTRAINT CK_orders_refund_consistency
        CHECK (   (order_status =  'refunded' AND refunded_at IS NOT NULL AND refunded_at >= order_ts)
               OR (order_status <> 'refunded' AND refunded_at IS NULL));
GO

/* --------------------------------------------------------- raw_web_events */
IF OBJECT_ID(N'dbo.FK_raw_web_events_users', N'F') IS NULL
    ALTER TABLE dbo.raw_web_events ADD CONSTRAINT FK_raw_web_events_users
        FOREIGN KEY (user_id) REFERENCES dbo.users (user_id);

IF OBJECT_ID(N'dbo.CK_raw_web_events_event_type', N'C') IS NULL
    ALTER TABLE dbo.raw_web_events ADD CONSTRAINT CK_raw_web_events_event_type
        CHECK (event_type IN ('page_view','product_view','add_to_cart','checkout_start','purchase'));

IF OBJECT_ID(N'dbo.CK_raw_web_events_device_type', N'C') IS NULL
    ALTER TABLE dbo.raw_web_events ADD CONSTRAINT CK_raw_web_events_device_type
        CHECK (device_type IN ('desktop','mobile','tablet'));

IF OBJECT_ID(N'dbo.CK_raw_web_events_referrer_type', N'C') IS NULL
    ALTER TABLE dbo.raw_web_events ADD CONSTRAINT CK_raw_web_events_referrer_type
        CHECK (referrer_type IS NULL
               OR referrer_type IN ('direct','organic_search','paid_search','social','email','referral'));
GO

PRINT 'Defaults, CHECK constraints and foreign keys ready.';
GO
