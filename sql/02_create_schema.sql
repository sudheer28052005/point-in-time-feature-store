/* =============================================================================
   02_create_schema.sql
   Schema layout:
     dbo        raw SOURCE data (users, orders, raw_web_events). Treated as
                append-only facts; never modified by the analytical pipeline.
     analytics  DERIVED objects (sessions, feature snapshots, cohorts, labels,
                benchmark logs). Fully rebuildable from dbo.

   Temporal contract for the whole project: every *_at / *_ts column holds UTC.
   ============================================================================= */
SET NOCOUNT ON;
USE CustomerFeatureStore;
GO

IF SCHEMA_ID(N'analytics') IS NULL
    EXEC (N'CREATE SCHEMA analytics AUTHORIZATION dbo;');
GO

PRINT 'Schemas ready: dbo (raw), analytics (derived).';
GO
