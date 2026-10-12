/* =============================================================================
   01_create_database.sql
   Creates the CustomerFeatureStore database if it does not already exist.

   - Idempotent: safe to re-run. It NEVER drops anything.
     (Destructive reset lives in 99_dev_reset_database.sql and needs an explicit flag.)
   - Recovery model SIMPLE: this is a rebuildable analytical dev database; full
     point-in-time log backups would only inflate the transaction log during bulk loads.
   - Data/log files go to the instance default location (works on Express,
     Developer, and Standard without hardcoded paths).
   ============================================================================= */
SET NOCOUNT ON;
USE master;
GO

IF DB_ID(N'CustomerFeatureStore') IS NULL
BEGIN
    CREATE DATABASE CustomerFeatureStore;
    PRINT 'Database CustomerFeatureStore created.';
END
ELSE
    PRINT 'Database CustomerFeatureStore already exists - nothing to do.';
GO

ALTER DATABASE CustomerFeatureStore SET RECOVERY SIMPLE;
-- Query Store is used in the performance phase (SQL Server 2016+; on by default in 2022+).
ALTER DATABASE CustomerFeatureStore SET QUERY_STORE = ON;
GO
