/* =============================================================================
   99_dev_reset_database.sql   --   DESTRUCTIVE, DEVELOPMENT ONLY
   Drops CustomerFeatureStore. Does nothing unless invoked with:
       sqlcmd ... -v CONFIRM_RESET=YES
   (scripts/run_sql.ps1 -Reset passes this for you.)
   It is deliberately NOT part of the numbered build sequence.
   ============================================================================= */
SET NOCOUNT ON;
USE master;
GO

IF N'$(CONFIRM_RESET)' <> N'YES'
BEGIN
    RAISERROR('Reset refused: pass -v CONFIRM_RESET=YES to drop CustomerFeatureStore.', 16, 1);
    SET NOEXEC ON;   -- skip everything below
END
GO

IF DB_ID(N'CustomerFeatureStore') IS NOT NULL
BEGIN
    ALTER DATABASE CustomerFeatureStore SET SINGLE_USER WITH ROLLBACK IMMEDIATE;
    DROP DATABASE CustomerFeatureStore;
    PRINT 'CustomerFeatureStore dropped.';
END
GO
SET NOEXEC OFF;
GO
