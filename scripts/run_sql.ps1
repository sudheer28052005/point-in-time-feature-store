<#
.SYNOPSIS
  Runs the numbered SQL scripts in sql/ against SQL Server using sqlcmd.

.EXAMPLE
  # Windows auth, local Express instance, build foundation (01-04)
  .\scripts\run_sql.ps1 -Server "localhost\SQLEXPRESS" -TrustServerCertificate

.EXAMPLE
  # SQL auth (password read from env var, never from the command line)
  $env:SQL_SERVER_PASSWORD = "..."; .\scripts\run_sql.ps1 -Server "localhost" -User sa

.EXAMPLE
  # Explicit destructive reset of the dev database, then rebuild
  .\scripts\run_sql.ps1 -Server "localhost\SQLEXPRESS" -TrustServerCertificate -Reset
#>
[CmdletBinding()]
param(
    [string]$Server = $(if ($env:SQL_SERVER_HOST) { $env:SQL_SERVER_HOST } else { "localhost" }),
    [string]$User,                       # omit => Windows Authentication
    [string[]]$Files,                    # default: 01-04 (foundation); pass e.g. 06_*.sql to run others
    [switch]$TrustServerCertificate,     # LOCAL DEV ONLY: accept the self-signed server cert
    [switch]$Reset                       # drops the database first (explicit opt-in)
)
$ErrorActionPreference = "Stop"
$sqlDir = Join-Path (Split-Path $PSScriptRoot -Parent) "sql"

$common = @("-S", $Server, "-b", "-I", "-l", "15")   # -b: stop with non-zero exit on error; -I: QUOTED_IDENTIFIER ON
if ($User) {
    if (-not $env:SQL_SERVER_PASSWORD) { throw "Set `$env:SQL_SERVER_PASSWORD for SQL authentication." }
    $common += @("-U", $User, "-P", $env:SQL_SERVER_PASSWORD)
} else { $common += "-E" }
if ($TrustServerCertificate) {
    Write-Warning "Trusting the server certificate (local development exception)."
    $common += "-C"
}
# NOTE: ODBC-based sqlcmd (v18) encrypts by default; we never pass a flag that disables encryption.

function Invoke-SqlFile([string]$path, [string[]]$extra = @()) {
    Write-Host ">> $(Split-Path $path -Leaf)" -ForegroundColor Cyan
    & sqlcmd @common @extra -i $path
    if ($LASTEXITCODE -ne 0) { throw "sqlcmd failed on $path (exit $LASTEXITCODE)" }
}

if ($Reset) { Invoke-SqlFile (Join-Path $sqlDir "99_dev_reset_database.sql") @("-v", "CONFIRM_RESET=YES") }

if (-not $Files) { $Files = @("01_create_database.sql","02_create_schema.sql","03_create_tables.sql","04_create_constraints.sql") }
foreach ($f in $Files) {
    $matches = Get-ChildItem -Path $sqlDir -Filter $f | Sort-Object Name
    if (-not $matches) { throw "No SQL file matches '$f' in $sqlDir" }
    foreach ($m in $matches) { Invoke-SqlFile $m.FullName }
}
Write-Host "Done." -ForegroundColor Green
