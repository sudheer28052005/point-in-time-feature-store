/* =============================================================================
   15_dashboard_readonly_role.sql
   A least-privilege DATABASE ROLE for the Streamlit dashboard. It creates no login and no user
   (server-level security is yours to decide); add a principal yourself, e.g.:
       CREATE LOGIN dashboard_ro WITH PASSWORD = '<strong password>';   -- SQL authentication
       CREATE USER  dashboard_ro FOR LOGIN dashboard_ro;
       ALTER ROLE dashboard_reader ADD MEMBER dashboard_ro;
   then set DASHBOARD_SQL_USERNAME / DASHBOARD_SQL_PASSWORD in .env.

   What the role can do
     SELECT   everything in schema analytics (features, labels, predictions, cohorts, benchmarks, sessions)
     SELECT   dbo.orders and dbo.users, but only non-sensitive users columns (NO date_of_birth)
   What it cannot do
     * read raw clickstream (dbo.raw_web_events: free-text page URLs)
     * INSERT / UPDATE / DELETE / ALTER anything, or EXECUTE the build procedures
   The dashboard additionally rejects any non-SELECT statement in Python (dashboard/data.py).
   Idempotent.
   ============================================================================= */
SET NOCOUNT ON;
USE CustomerFeatureStore;
GO

IF DATABASE_PRINCIPAL_ID(N'dashboard_reader') IS NULL
    CREATE ROLE dashboard_reader;
GO

GRANT SELECT ON SCHEMA::analytics TO dashboard_reader;
GRANT SELECT ON dbo.orders TO dashboard_reader;
GRANT SELECT ON dbo.users (user_id, signup_at, country, acquisition_channel, customer_segment, is_marketing_opted_in)
    TO dashboard_reader;

-- defence in depth: explicit denies win over any grant inherited through other roles
DENY SELECT ON dbo.users (date_of_birth) TO dashboard_reader;
DENY SELECT ON dbo.raw_web_events TO dashboard_reader;
DENY INSERT, UPDATE, DELETE, ALTER ON SCHEMA::analytics TO dashboard_reader;
DENY INSERT, UPDATE, DELETE ON dbo.users TO dashboard_reader;
DENY INSERT, UPDATE, DELETE ON dbo.orders TO dashboard_reader;
DENY INSERT, UPDATE, DELETE ON dbo.raw_web_events TO dashboard_reader;
DENY EXECUTE ON SCHEMA::analytics TO dashboard_reader;     -- the build procedures live here
GO
PRINT 'Role dashboard_reader ready (no members yet).';
GO
