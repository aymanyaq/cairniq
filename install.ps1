# =============================================================
#  CairnIQ - Windows Installer (PowerShell 5.1+)
# =============================================================
#  Usage:
#    Right-click -> "Run with PowerShell"
#    powershell -ExecutionPolicy Bypass -File install.ps1
#
#  This file is pure ASCII on purpose: PowerShell 5.1 reads
#  .ps1 files as the system ANSI codepage when there is no
#  UTF-8 BOM. Keeping the source ASCII avoids mojibake and
#  parser failures on default Windows installs.
# =============================================================
#Requires -Version 5.1

$ErrorActionPreference = "Stop"
$ProgressPreference    = "SilentlyContinue"   # speeds up Compress-Archive / web ops

# -------- Helpers --------------------------------------------------
function Write-Info  { param([string]$Msg) Write-Host "  [OK] $Msg"  -ForegroundColor Green  }
function Write-Warn2 { param([string]$Msg) Write-Host "  [!]  $Msg"  -ForegroundColor Yellow }
function Write-Step  { param([string]$Msg) Write-Host "`n  >>  $Msg" -ForegroundColor Cyan   }
function Write-Fatal {
    param([string]$Msg)
    Write-Host "`n  [X] $Msg" -ForegroundColor Red
    Read-Host "Press Enter to exit" | Out-Null
    exit 1
}

function Test-NonInteractive {
    $v = $env:CAIRNIQ_NONINTERACTIVE
    return ($v -eq "1" -or $v -eq "true" -or $v -eq "yes" -or $v -eq "y")
}

function Write-JsonNoBom {
    param([string]$Path, [string]$Content)
    # PowerShell 5.1's Set-Content -Encoding UTF8 writes a BOM that some
    # downstream tools dislike. Use the .NET API to write without a BOM.
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $utf8NoBom)
}

# -------- Header ---------------------------------------------------
Clear-Host
Write-Host ""
Write-Host "  =============================================================" -ForegroundColor Cyan
Write-Host "                    C A I R N I Q   I N S T A L L"            -ForegroundColor Cyan
Write-Host "                    Private Portfolio Intelligence"           -ForegroundColor Cyan
Write-Host "  =============================================================" -ForegroundColor Cyan
Write-Host ""

# -------- Resolve project root ------------------------------------
$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $PROJECT_ROOT) { $PROJECT_ROOT = (Get-Location).Path }
Set-Location -LiteralPath $PROJECT_ROOT

$BACKUP_DIR     = Join-Path $PROJECT_ROOT "backups"
$BACKUP_CREATED = $false
$userData       = Join-Path $PROJECT_ROOT "user_data"

# -------- 1. Data Preservation ------------------------------------
Write-Step "Data Protection: Checking for existing user data..."
if (Test-Path -LiteralPath $userData) {
    New-Item -ItemType Directory -Force -Path $BACKUP_DIR | Out-Null
    $stamp      = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupFile = Join-Path $BACKUP_DIR "user_data_$stamp.zip"
    Write-Step "Existing user_data found. Creating mandatory backup..."
    try {
        Compress-Archive -Path $userData -DestinationPath $backupFile -Force
        $BACKUP_CREATED = $true
        Write-Info "Safe backup created: user_data_$stamp.zip"
    } catch {
        Write-Warn2 "Backup created with warnings: $($_.Exception.Message)"
    }
} else {
    Write-Info "No existing user data found."
}

# -------- 2. Legacy file migration --------------------------------
Write-Step "Migration: Checking for legacy data files..."
$legacyFiles = @("checkpoints.sqlite","chat_history.json","knowledge_graph.json","user_memory.json",".env","my_portfolio.csv")
New-Item -ItemType Directory -Force -Path $userData | Out-Null
$migratedCount = 0
foreach ($f in $legacyFiles) {
    $src = Join-Path $PROJECT_ROOT $f
    $dst = Join-Path $userData $f
    if ((Test-Path -LiteralPath $src) -and -not (Test-Path -LiteralPath $dst)) {
        Move-Item -LiteralPath $src -Destination $dst
        $migratedCount++
        Write-Step ("Moved legacy file '{0}' to user_data\" -f $f)
    }
}
if ($migratedCount -gt 0) {
    Write-Info "Migrated $migratedCount legacy file(s) to user_data\"
} else {
    Write-Info "No legacy files found."
}

# -------- 3. Python version check ---------------------------------
Write-Step "System: Checking Python version..."
$PYTHON_BIN     = $null
$PYTHON_VERSION = $null

# On Windows, 'py' (the Python launcher) is preferred, then 'python'.
$candidates = @("py -3.13","py -3.12","py -3","python","python3")
foreach ($candidate in $candidates) {
    try {
        $parts   = $candidate -split ' '
        $exe     = $parts[0]
        $cmdArgs = @()
        if ($parts.Length -gt 1) { $cmdArgs = $parts[1..($parts.Length - 1)] }

        # Quick existence test
        $null = Get-Command $exe -ErrorAction Stop

        $ver = & $exe @cmdArgs --version 2>&1
        $verText = ($ver | Out-String).Trim()
        if ($verText -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            # Reject Python 3.14+ (and any future 4.x). Pydantic V1 - still
            # pulled in transitively via the pydantic.v1 compat shim used by
            # parts of the LangChain ecosystem - is not compatible with
            # Python 3.14+. Currently tested range is 3.12-3.13. 3.11 dropped:
            # numpy>=2.5 has no 3.11 wheel.
            if ($major -gt 3 -or ($major -eq 3 -and $minor -ge 14)) {
                continue
            }
            if ($major -eq 3 -and $minor -ge 12) {
                $PYTHON_BIN     = $candidate
                $PYTHON_VERSION = "$major.$minor"
                break
            }
        }
    } catch {
        # try next candidate
    }
}

if (-not $PYTHON_BIN) {
    Write-Fatal "Python 3.12-3.13 is required but was not found.`n  Currently supported range: 3.12, 3.13.`n  Python 3.11 is no longer supported (numpy 2.5+ dropped it) and 3.14+ is not yet supported (Pydantic V1 compat shim is incompatible).`n  Download a supported version from https://www.python.org/downloads/ and rerun install.ps1.`n  Important: tick 'Add Python to PATH' in the installer."
}
Write-Info "Python $PYTHON_VERSION found ($PYTHON_BIN)"

# Build a small invoker so we can call the selected interpreter uniformly.
$pyParts = $PYTHON_BIN -split ' '
$pyExe   = $pyParts[0]
$pyArgs  = @()
if ($pyParts.Length -gt 1) { $pyArgs = $pyParts[1..($pyParts.Length - 1)] }

# -------- 4. Port conflict check ----------------------------------
Write-Step "System: Checking port availability..."
$PORT = if ($env:PORT) { $env:PORT } else { "8000" }
try {
    $portInUse = Get-NetTCPConnection -LocalPort ([int]$PORT) -ErrorAction Stop
    if ($portInUse) {
        Write-Warn2 "Port $PORT is already in use. Stop the existing process before launching CairnIQ."
    }
} catch {
    # Get-NetTCPConnection throws when no connections match -> port is free
    Write-Info "Port $PORT available."
}

# -------- 5. Virtual environment ----------------------------------
Write-Step "Environment: Setting up virtual environment..."
$venvDir    = Join-Path $PROJECT_ROOT ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$recreate   = $false

if (Test-Path -LiteralPath $venvDir) {
    if (Test-Path -LiteralPath $venvPython) {
        Write-Info "Existing virtual environment detected."
        if (-not (Test-NonInteractive)) {
            $reply = Read-Host "  Recreate virtual environment? (Recommended for major updates) [y/N]"
            if ($reply -match "^[Yy]$") { $recreate = $true }
        } else {
            Write-Info "Non-interactive mode: keeping existing virtual environment."
        }
    } else {
        Write-Warn2 "Broken virtual environment detected. Recreating..."
        $recreate = $true
    }
} else {
    $recreate = $true
}

if ($recreate) {
    Write-Step "Creating new virtual environment..."
    if (Test-Path -LiteralPath $venvDir) {
        Remove-Item -Recurse -Force -LiteralPath $venvDir -ErrorAction SilentlyContinue
    }
    & $pyExe @pyArgs -m venv $venvDir
    if (-not (Test-Path -LiteralPath $venvPython)) {
        Write-Fatal "Failed to create virtual environment at $venvDir"
    }
    Write-Info "Virtual environment created."
}

$python = Join-Path $venvDir "Scripts\python.exe"

# -------- 6. Dependencies -----------------------------------------
$skipDeps = $env:CAIRNIQ_SKIP_DEPENDENCY_INSTALL
if ($skipDeps -eq "1" -or $skipDeps -eq "true" -or $skipDeps -eq "yes") {
    Write-Warn2 "Skipping dependency installation (CAIRNIQ_SKIP_DEPENDENCY_INSTALL is set)."
} else {
    Write-Step "Dependencies: Upgrading pip / setuptools / wheel..."
    # Use 'python -m pip' to avoid the Windows pip-self-upgrade file lock.
    & $python -m pip install --upgrade pip setuptools wheel --disable-pip-version-check -q
    if ($LASTEXITCODE -ne 0) { Write-Fatal "pip upgrade failed (exit $LASTEXITCODE)" }
    Write-Info "pip upgraded."

    Write-Step "Dependencies: Installing application requirements (5-10 minutes)..."
    & $python -m pip install -r requirements.txt --disable-pip-version-check -q
    if ($LASTEXITCODE -ne 0) { Write-Fatal "Failed to install requirements.txt (exit $LASTEXITCODE)" }
    Write-Info "Core dependencies installed."

    $reqOpt = Join-Path $PROJECT_ROOT "requirements-optional.txt"
    if (Test-Path -LiteralPath $reqOpt) {
        Write-Step "Dependencies: Installing optional packages (FAISS)..."
        & $python -m pip install -r $reqOpt --disable-pip-version-check -q 2>$null

        # Detect FAISS by running it and capturing $LASTEXITCODE separately.
        & $python -c "import faiss" *> $null
        if ($LASTEXITCODE -eq 0) {
            Write-Info "FAISS vector search installed."
        } else {
            Write-Warn2 "FAISS not available - CairnIQ will fall back to BM25 keyword retrieval."
        }
    }

    # Warm the bytecode cache so the FIRST server start isn't slowed by Python
    # compiling the large dependency tree (DSPy, LangChain, …) to .pyc. Pure
    # optimization — never fail the install over it. (Mirrors install.sh.)
    Write-Step "Performance: Precompiling bytecode (one-time, speeds up first launch)..."
    $compileTargets = @(
        (Join-Path $venvDir "Lib"),
        (Join-Path $PROJECT_ROOT "agent"),
        (Join-Path $PROJECT_ROOT "api"),
        (Join-Path $PROJECT_ROOT "tools"),
        (Join-Path $PROJECT_ROOT "lib")
    )
    & $python -m compileall -q -j 0 @compileTargets 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Info "Bytecode cache warmed."
    } else {
        Write-Warn2 "Bytecode precompile skipped (non-fatal); first launch may be slightly slower."
    }
}

# -------- 7. user_data initialisation -----------------------------
Write-Step "Data: Initialising user data directory..."
New-Item -ItemType Directory -Force -Path $userData | Out-Null

# Ensure the full directory structure exists (mirrors install.sh). The runtime
# writes caches, embeddings, per-profile data and logs into these subfolders.
$dirTree = @(
    (Join-Path $PROJECT_ROOT "logs\agent"),
    (Join-Path $PROJECT_ROOT "logs\chat_runtime"),
    (Join-Path $PROJECT_ROOT "logs\frontend"),
    (Join-Path $PROJECT_ROOT "logs\server"),
    (Join-Path $PROJECT_ROOT "logs\tools"),
    (Join-Path $userData "cache"),
    (Join-Path $userData "embeddings"),
    (Join-Path $userData "profiles"),
    (Join-Path $userData "daily_cache"),
    (Join-Path $PROJECT_ROOT "tmp")
)
foreach ($d in $dirTree) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
Write-Info "Directory structure verified."

$envSrc = Join-Path $PROJECT_ROOT ".env.example"
$envDst = Join-Path $userData ".env"
if (-not (Test-Path -LiteralPath $envDst)) {
    if (Test-Path -LiteralPath $envSrc) {
        Copy-Item -LiteralPath $envSrc -Destination $envDst
        Write-Info "Created user_data\.env from .env.example - add your API keys there."
    }
}

$funnelSrc = Join-Path $PROJECT_ROOT "funnel_config.example.json"
$funnelSrc = Join-Path $PROJECT_ROOT "funnel_config.example.json"
$funnelDst = Join-Path $userData "funnel_config.json"
if (-not (Test-Path -LiteralPath $funnelDst)) {
    if (Test-Path -LiteralPath $funnelSrc) {
        Copy-Item -LiteralPath $funnelSrc -Destination $funnelDst
        Write-Info "Created user_data\funnel_config.json from example - tune the opportunity scanner there (see docs/technical/FUNNEL_CONFIG.md)."
    }
}

$portfolioSrc = Join-Path $PROJECT_ROOT "my_portfolio.example.csv"
$portfolioDst = Join-Path $userData "my_portfolio.csv"
if (-not (Test-Path -LiteralPath $portfolioDst)) {
    if (Test-Path -LiteralPath $portfolioSrc) {
        Copy-Item -LiteralPath $portfolioSrc -Destination $portfolioDst
        Write-Info "Created user_data\my_portfolio.csv from template."
    }
}

$chatFile = Join-Path $userData "chat_history.json"
if (-not (Test-Path -LiteralPath $chatFile)) {
    Write-JsonNoBom -Path $chatFile -Content '{"sessions":[]}'
    Write-Info "Initialised user_data\chat_history.json"
}

$memFile = Join-Path $userData "user_memory.json"
if (-not (Test-Path -LiteralPath $memFile)) {
    $memTemplate = @'
{
  "user_profile": {
    "name": null,
    "age": null,
    "risk_tolerance": null,
    "retirement_age": null,
    "annual_income": null,
    "investment_goals": [],
    "accounts": [],
    "last_updated": null
  },
  "key_facts": [],
  "conversation_summaries": [],
  "past_recommendations": [],
  "active_theses": [],
  "lessons_learned": []
}
'@
    Write-JsonNoBom -Path $memFile -Content $memTemplate
    Write-Info "Initialised user_data\user_memory.json"
}

$graphFile = Join-Path $userData "knowledge_graph.json"
if (-not (Test-Path -LiteralPath $graphFile)) {
    Write-JsonNoBom -Path $graphFile -Content '{"directed":true,"multigraph":true,"graph":{},"nodes":[],"links":[]}'
    Write-Info "Initialised user_data\knowledge_graph.json"
}

# -------- 8. Desktop launcher -------------------------------------
Write-Step "Desktop: Updating launcher..."
$batSrc     = Join-Path $PROJECT_ROOT "CairnIQ.bat"
$desktopDir = [Environment]::GetFolderPath("Desktop")
$desktopDst = Join-Path $desktopDir "CairnIQ.bat"

if (-not (Test-Path -LiteralPath $batSrc)) {
    Write-Warn2 "CairnIQ.bat not found in project root - skipping desktop shortcut."
} elseif (Test-Path -LiteralPath $desktopDst) {
    Write-Info "Desktop launcher already exists."
} elseif (-not (Test-Path -LiteralPath $desktopDir)) {
    Write-Warn2 "Desktop folder not found - skipping desktop shortcut."
} else {
    # Wrapper that cd's into the project then calls the repo .bat.
    # cmd.exe handles "C:\path with spaces\..." quoting natively.
    # Use the OEM/system codepage so non-ASCII chars in $PROJECT_ROOT
    # (e.g. C:\Users\Jose\...) survive intact for cmd.exe.
    $wrapper = "@echo off`r`ncd /d `"$PROJECT_ROOT`"`r`ncall `"$batSrc`"`r`n"
    Set-Content -LiteralPath $desktopDst -Value $wrapper -Encoding Default -NoNewline
    Write-Info "Desktop launcher created: $desktopDst"
}

# -------- 9. Integrity check --------------------------------------
Write-Step "Validation: Running data integrity check..."
$verifyScript = Join-Path $PROJECT_ROOT "scripts\install\verify_data.py"
if (Test-Path -LiteralPath $verifyScript) {
    & $python $verifyScript
} else {
    Write-Warn2 "verify_data.py not found - skipping integrity check."
}

# -------- 10. Guided Setup Wizard ---------------------------------
Write-Host ""
$launchWizard = $false
if (Test-NonInteractive) {
    Write-Info "Non-interactive mode: skipping Guided Setup Wizard."
} else {
    $reply = Read-Host "  Would you like to launch the Guided Setup Wizard to configure API keys? [Y/n]"
    $launchWizard = ($reply -eq "" -or $reply -match "^[Yy]$")
}
if ($launchWizard) {
    $wizardScript = Join-Path $PROJECT_ROOT "scripts\install\guided_setup.py"
    if (Test-Path -LiteralPath $wizardScript) {
        & $python $wizardScript
    } else {
        Write-Warn2 "Guided setup script not found."
    }
}

# -------- Summary --------------------------------------------------
Write-Host ""
Write-Host "  =============================================================" -ForegroundColor Green
Write-Host "    Installation Complete!"                                      -ForegroundColor Green
Write-Host "  =============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Location : $PROJECT_ROOT" -ForegroundColor Cyan
Write-Host "  Backups  : $BACKUP_DIR"   -ForegroundColor Cyan
Write-Host ""
if ($BACKUP_CREATED) {
    Write-Host "  Your existing user_data\ was preserved and backed up." -ForegroundColor Yellow
}
Write-Host "  Security : API keys are stored in the OS keychain when available." -ForegroundColor Yellow
Write-Host "             If keychain access is unavailable, setup falls back to user_data\.env." -ForegroundColor Yellow
Write-Host "             Never commit user_data\.env to version control."    -ForegroundColor Yellow
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor Cyan
Write-Host "    1. Double-click CairnIQ.bat on your Desktop"
Write-Host "    2. Or run:  CairnIQ.bat  from this directory"
Write-Host ""
Read-Host "Press Enter to close" | Out-Null
