# Scripts Directory

Utility scripts for running, installing, packaging, and maintaining CairnIQ.

## Directory Structure

```
scripts/
├── run_api.sh                  # Run the FastAPI server (dev, --reload)
├── cairniq_user.py             # Manage multi-user login accounts
├── cairniq_watchdog.py         # Runaway kill-switch + liveness supervisor (launchd)
├── install/                    # Installation & maintenance
│   ├── install.sh             # Main installer (with data preservation)
│   ├── guided_setup.py        # Interactive setup wizard
│   ├── setup_service.sh       # Install as a macOS launchd service
│   ├── import_config.sh       # Import shared configuration bundle
│   ├── restore_backup.sh      # Restore user data from backup
│   └── verify_data.py         # Data integrity verification
├── package/                    # Distribution packaging
│   ├── create_package.sh      # Create full application package
│   └── create_config_bundle.sh # Create shared config bundle
├── docker/                     # Docker configuration
│   ├── Dockerfile
│   └── .dockerignore
└── local/                      # Local-only, gitignored scratch scripts (see its README)
```

> **Launching the app.** Day-to-day you start CairnIQ with the root launchers, not
> these scripts: `CairnIQ.command` (production) or `start_demo.sh` (demo). See
> [docs/LAUNCHER_MODES.md](../docs/LAUNCHER_MODES.md). Use `run_api.sh` when you want
> a bare uvicorn dev server with auto-reload.

## Runtime Scripts

### Run the API

**Location**: `scripts/run_api.sh`

```bash
./scripts/run_api.sh
```

Starts uvicorn against `server:app` with `--reload`. Honors `CAIRNIQ_HOST`
(default `127.0.0.1`) and `PORT` (default `8000`). Requires `.venv` — run
`./install.sh` first.

### Manage Users

**Location**: `scripts/cairniq_user.py`

```bash
python scripts/cairniq_user.py add alice --profile alice --role admin
python scripts/cairniq_user.py list
python scripts/cairniq_user.py passwd alice
python scripts/cairniq_user.py remove olduser
```

Manages login accounts for multi-user setups. Only needed once you enable auth to
reach the server from another device — single-user local installs need none of this.

### Watchdog

**Location**: `scripts/cairniq_watchdog.py`

Runaway kill-switch **and** liveness supervisor, run periodically by launchd
(`com.cairniq.watchdog`) independently of the server.

- **Kill-switch:** enforces the LLM hard budget and detects server restart-storms;
  on breach it alerts and can disable the server LaunchAgent. See
  `agent/llm_budget.py` for the budget it watches.
- **Liveness:** if nothing is listening on the server port and the LaunchAgent is
  loaded but idle, it `launchctl kickstart`s the service. This covers the case
  where a server started outside launchd won the port race, launchd's own start
  hit the single-instance guard and exited 0, and `KeepAlive {SuccessfulExit:
  false}` left the job dormant — so the service otherwise stayed down for good.

Revival never fights the kill-switch: it is skipped entirely on a runaway breach
and while the watchdog itself disabled the job, needs two consecutive down checks,
and is capped at one kickstart per 10 minutes / 3 attempts. Set
`AIDLC_WATCHDOG_AUTOREVIVE=0` to alert without reviving.

## Installation Scripts

### Main Installer

**Location**: `scripts/install/install.sh`

```bash
./install.sh
```

**What it does**:
1. Creates a backup of existing `user_data/` (if found)
2. Migrates legacy data files into `user_data/`
3. Checks Python version (3.12+ required)
4. Creates/validates virtual environment
5. Installs dependencies
6. Runs data integrity verification
7. Offers to launch the **Guided Setup Wizard**

### Guided Setup Wizard

**Location**: `scripts/install/guided_setup.py`

```bash
python3 scripts/install/guided_setup.py
```

**What it does**:
- Asks if you have a shared config bundle to import
- Configures your identity and financial goals
- Sets up LLM provider (Bedrock, Anthropic, or OpenAI)
- Configures financial data API keys
- Links brokerage accounts (Alpaca or Questrade)

### Run as a Service

**Location**: `scripts/install/setup_service.sh`

```bash
./scripts/install/setup_service.sh
```

Installs CairnIQ as a user-level macOS launchd agent so it runs in the background
and restarts on login/reboot.

### Data Utilities

| Script | Purpose |
|---|---|
| `restore_backup.sh` | Interactive restore from timestamped backup |
| `verify_data.py` | Validates JSON, SQLite, and .env integrity |
| `import_config.sh` | Applies a shared config bundle (.zip) |

## Packaging Scripts

### Create Package

**Location**: `scripts/package/create_package.sh`

```bash
./scripts/package/create_package.sh [version]
```

Creates a clean distribution archive excluding all user data. Optionally offers to create a shared configuration bundle alongside.

### Create Config Bundle

**Location**: `scripts/package/create_config_bundle.sh`

```bash
./scripts/package/create_config_bundle.sh
```

Creates a lightweight `.zip` containing only shareable infrastructure settings (API keys, global rules) — never personal data.

## Docker

```bash
cd scripts/docker
docker build -t cairniq .
docker run -p 8000:8000 cairniq
```

## Local-Only Scripts

`scripts/local/` is a default-deny, gitignored zone for ad-hoc fix-ups and one-off
migrations — nothing there is committed except its own `README.md`. See
[scripts/local/README.md](local/README.md).

---

**Last Updated**: July 4, 2026
