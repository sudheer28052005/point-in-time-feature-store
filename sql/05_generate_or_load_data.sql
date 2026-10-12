/* =============================================================================
   05_generate_or_load_data.sql
   Data is generated in Python (src/data_generation.py, Faker + NumPy) and bulk
   loaded by scripts/load_data.py via pyodbc fast_executemany. T-SQL is a poor
   fit for seeded, reproducible random generation and row-by-row INSERTs are
   explicitly avoided, so this script does not generate rows.

   What it DOES provide: a load-audit table so every dataset in the database can
   be traced to the seed, volumes and content fingerprint that produced it.
   Idempotent.
   ============================================================================= */
SET NOCOUNT ON;
USE CustomerFeatureStore;
GO

IF OBJECT_ID(N'analytics.data_load_log', N'U') IS NULL
BEGIN
    CREATE TABLE analytics.data_load_log
    (
        load_id        INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_data_load_log PRIMARY KEY,
        loaded_at      DATETIME2(3)  NOT NULL CONSTRAINT DF_data_load_log_loaded_at DEFAULT (SYSUTCDATETIME()),
        random_seed    INT           NOT NULL,
        num_users      INT           NOT NULL,
        num_orders     INT           NOT NULL,
        num_events     INT           NOT NULL,
        fingerprint    CHAR(64)      NOT NULL,   -- SHA-256 of the generated dataset
        load_seconds   DECIMAL(9,2)  NULL,
        notes          NVARCHAR(400) NULL
    );
END
GO
PRINT 'analytics.data_load_log ready.';
GO
