#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# CairnIQ — Universal Installer (v2.0)
# ═══════════════════════════════════════════════════════════════
# Installs the application with built-in data preservation.
# Usage: ./scripts/install/install.sh
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${GREEN}✓${NC} $*"; }
warn()  { echo -e "${YELLOW}⚠${NC} $*"; }
error() { echo -e "${RED}✗${NC} $*"; exit 1; }
step()  { echo -e "${CYAN}▸${NC} $*"; }
is_noninteractive() { [[ "${CAIRNIQ_NONINTERACTIVE:-}" =~ ^(1|true|yes|y)$ ]]; }

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BACKUP_DIR="${PROJECT_ROOT}/backups"
BACKUP_CREATED=false

cd "$PROJECT_ROOT"

# Header
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  🏛️  CairnIQ — Secure Installation${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# 1. PRE-FLIGHT: Data Preservation & Shared Config
step "Data Protection: Checking for existing user data..."
if [[ -d "user_data" ]]; then
    mkdir -p "$BACKUP_DIR"
    BACKUP_FILE="${BACKUP_DIR}/user_data_$(date +%Y%m%d_%H%M%S).tar.gz"
    step "Existing user_data found. Creating mandatory backup..."
    tar -czf "$BACKUP_FILE" user_data/ 2>/dev/null || warn "Backup created with some warnings"
    BACKUP_CREATED=true
    info "Safe backup created: $(basename "$BACKUP_FILE")"
else
    info "No existing user data found."
fi

# 2. MIGRATION: Handle legacy file locations
step "Migration: Checking for legacy data files..."
LEGACY_FILES=("checkpoints.sqlite" "chat_history.json" "knowledge_graph.json" "user_memory.json" ".env" "my_portfolio.csv")
MIGRATED_COUNT=0
mkdir -p user_data

for file in "${LEGACY_FILES[@]}"; do
    if [[ -f "$file" ]] && [[ ! -L "$file" ]]; then
        step "Moving legacy file '$file' to user_data/..."
        mv "$file" user_data/
        MIGRATED_COUNT=$((MIGRATED_COUNT + 1))
    fi
done

if [[ $MIGRATED_COUNT -gt 0 ]]; then
    info "Migrated $MIGRATED_COUNT legacy files to user_data/"
else
    info "No legacy files found"
fi

# 3. ENVIRONMENT: Python Check
step "System: Checking Python version..."
PYTHON_BIN=""
PYTHON_VERSION=""
for candidate in python3.13 python3.12 python3; do
    if ! command -v "$candidate" &>/dev/null; then
        continue
    fi
    candidate_version=$("$candidate" --version 2>&1 | awk '{print $2}')
    candidate_major=$(echo "$candidate_version" | cut -d. -f1)
    candidate_minor=$(echo "$candidate_version" | cut -d. -f2)
    # Reject Python 3.14+ (and any future 4.x). Pydantic V1 - still pulled
    # in transitively via the pydantic.v1 compat shim used by parts of the
    # LangChain ecosystem - is not compatible with Python 3.14+. Currently
    # tested range is 3.12-3.13. 3.11 dropped: numpy>=2.5 has no 3.11 wheel.
    if [[ "$candidate_major" -gt 3 ]] || [[ "$candidate_major" -eq 3 && "$candidate_minor" -ge 14 ]]; then
        continue
    fi
    if [[ "$candidate_major" -eq 3 && "$candidate_minor" -ge 12 ]]; then
        PYTHON_BIN="$(command -v "$candidate")"
        PYTHON_VERSION="$candidate_version"
        break
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    found_python="$(python3 --version 2>&1 | awk '{print $2}' || true)"
    error "Python 3.12-3.13 required. Found: ${found_python:-not installed}. Python 3.11 is no longer supported (numpy 2.5+ dropped it) and 3.14+ is not yet supported (Pydantic V1 compat shim is incompatible). Install Python 3.12 or 3.13 and rerun ./install.sh."
fi
info "Python $PYTHON_VERSION found at $PYTHON_BIN"

# 4. PORT: Conflict Check
step "System: Checking port availability..."
PORT="${PORT:-8000}"
if lsof -i :"$PORT" &>/dev/null; then
    warn "Port $PORT is already in use. Installation can continue, but stop the existing server before launching CairnIQ."
else
    info "Port $PORT available"
fi

# 5. VIRTUAL ENV: Safe Setup
step "Environment: Setting up virtual environment..."
RECREATE_VENV=false
if [[ -d ".venv" ]]; then
    if [[ -f ".venv/bin/python" ]]; then
        info "Existing virtual environment detected"
        if is_noninteractive; then
            info "Non-interactive mode: keeping existing virtual environment"
        else
            echo -n "  Recreate virtual environment? (Recommended for major updates) [y/N]: "
            read -r reply
            if [[ "$reply" =~ ^[Yy]$ ]]; then
                RECREATE_VENV=true
            fi
        fi
    else
        warn "Broken virtual environment detected. Recreating..."
        RECREATE_VENV=true
    fi
else
    RECREATE_VENV=true
fi

if [[ "$RECREATE_VENV" == "true" ]]; then
    step "Creating new virtual environment..."
    rm -rf .venv
    "$PYTHON_BIN" -m venv .venv
    info "Virtual environment created"
fi

# Activate
source .venv/bin/activate

# 6. DEPENDENCIES: Installation
if [[ "${CAIRNIQ_SKIP_DEPENDENCY_INSTALL:-}" =~ ^(1|true|yes|y)$ ]]; then
    warn "Skipping dependency installation because CAIRNIQ_SKIP_DEPENDENCY_INSTALL is set."
elif [[ -f "requirements.txt" ]]; then
    step "Dependencies: Updating core packages..."
    pip install --upgrade pip setuptools wheel -q
    info "Pip upgraded"

    step "Dependencies: Installing application requirements (5-10 mins)..."
    pip install -r requirements.txt -q
    info "Application dependencies installed"

    if [[ -f "requirements-optional.txt" ]]; then
        step "Dependencies: Installing optional acceleration packages..."
        if pip install -r requirements-optional.txt -q; then
            info "Optional dependencies installed"
        else
            warn "Optional dependencies could not be installed. CairnIQ will continue with fallback retrieval."
        fi
    fi
else
    error "requirements.txt not found"
fi

# 6b. PRECOMPILE: Warm the bytecode cache so the FIRST server start isn't slowed
# by Python compiling the large dependency tree (DSPy, LangChain, etc.) to .pyc.
# Pure optimization — never fail the install over it.
step "Performance: Precompiling bytecode (one-time, speeds up first launch)..."
if python -m compileall -q -j 0 .venv/lib agent api tools lib 2>/dev/null; then
    info "Bytecode cache warmed"
else
    warn "Bytecode precompile skipped (non-fatal); first launch may be slightly slower."
fi

# 7. STRUCTURE: Directory Creation
step "Infrastructure: Ensuring directory structure..."
mkdir -p logs/{agent,chat_runtime,frontend,server,tools}
mkdir -p user_data/{cache,embeddings,profiles,daily_cache}
mkdir -p tmp
info "Directories verified"

# 8. CONFIG: Template Sync
step "Configuration: Synchronizing templates..."
if [[ ! -f "user_data/.env" ]]; then
    if [[ -f ".env.example" ]]; then
        cp .env.example user_data/.env
        info "Created user_data/.env from template"
    else
        warn ".env.example not found - you will need to create user_data/.env manually"
    fi
fi

if [[ ! -f "user_data/my_portfolio.csv" ]]; then
    if [[ -f "my_portfolio.example.csv" ]]; then
        cp my_portfolio.example.csv user_data/my_portfolio.csv
        info "Created user_data/my_portfolio.csv from template"
    fi
fi

if [[ ! -f "user_data/funnel_config.json" ]]; then
    if [[ -f "funnel_config.example.json" ]]; then
        cp funnel_config.example.json user_data/funnel_config.json
        info "Created user_data/funnel_config.json from template (tune the opportunity scanner there; see docs/technical/FUNNEL_CONFIG.md)"
    fi
fi

if [[ ! -f "user_data/chat_history.json" ]]; then
    printf '{"sessions":[]}\n' > user_data/chat_history.json
    info "Initialized user_data/chat_history.json"
fi

if [[ ! -f "user_data/user_memory.json" ]]; then
    cat > user_data/user_memory.json <<'EOF'
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
EOF
    info "Initialized user_data/user_memory.json"
fi

if [[ ! -f "user_data/knowledge_graph.json" ]]; then
    printf '{"directed":true,"multigraph":true,"graph":{},"nodes":[],"links":[]}\n' > user_data/knowledge_graph.json
    info "Initialized user_data/knowledge_graph.json"
fi

# 9. DESKTOP: Launcher Update
step "Desktop: Updating launcher..."
if [[ -f "CairnIQ.command" ]]; then
    chmod +x "CairnIQ.command"
    DESKTOP_LAUNCHER="$HOME/Desktop/CairnIQ.command"
    if [[ -d "$HOME/Desktop" ]]; then
        # Always (re)write the desktop launcher as a thin wrapper that execs the
        # in-repo one. Skipping when it already exists used to strand old copies
        # that ran server.py directly — bypassing launchd supervision entirely.
        if [[ -f "$DESKTOP_LAUNCHER" ]] && ! grep -q 'CAIRNIQ_PROJECT_DIR' "$DESKTOP_LAUNCHER"; then
            cp "$DESKTOP_LAUNCHER" "${DESKTOP_LAUNCHER}.bak"
            warn "Existing desktop launcher was not a wrapper; backed it up to ${DESKTOP_LAUNCHER}.bak"
        fi
        {
            printf '#!/usr/bin/env bash\n'
            printf 'export CAIRNIQ_PROJECT_DIR=%q\n' "$PROJECT_ROOT"
            printf 'exec %q\n' "$PROJECT_ROOT/CairnIQ.command"
        } > "$DESKTOP_LAUNCHER"
        chmod +x "$DESKTOP_LAUNCHER"
        info "Desktop launcher updated"
    else
        warn "Desktop directory not found; skipping desktop launcher"
    fi
fi

# 10. VERIFICATION: Integrity Check
step "Validation: Running data integrity suite..."
if [[ -f "scripts/install/verify_data.py" ]]; then
    python3 scripts/install/verify_data.py
else
    warn "Verification script missing. Skipping integrity check."
fi

# 11. GUIDED SETUP: Launch Wizard
echo ""
echo -e "${CYAN}▸${NC} Technical setup complete! Would you like to launch the ${BOLD}Guided Setup Wizard${NC}?"
if is_noninteractive; then
    launch_wizard="n"
    info "Non-interactive mode: skipping Guided Setup Wizard"
else
    echo -n "  to configure your API keys and profile now? [Y/n]: "
    read -r launch_wizard
fi
if [[ "$launch_wizard" =~ ^[Yy]$ ]] || [[ -z "${launch_wizard:-}" ]]; then
    if [[ -f "scripts/install/guided_setup.py" ]]; then
        python3 scripts/install/guided_setup.py
    else
        warn "Guided setup script missing."
    fi
fi

# Summary
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Installation & Update Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${CYAN}Location:${NC}     $PROJECT_ROOT"
echo -e "  ${CYAN}Backups:${NC}      $BACKUP_DIR"
echo ""
echo -e "${YELLOW}Security Notice:${NC}"
if [[ "$BACKUP_CREATED" == "true" ]]; then
    echo "  Your existing user_data/ was preserved. A fresh backup was"
    echo "  created in the backups/ folder as a safety precaution."
else
    echo "  No existing user_data/ was found, so no backup was needed."
fi
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Double-click 'CairnIQ.command' on your Desktop"
echo "  2. Or run: ./CairnIQ.command"
echo ""
echo -e "${CYAN}Tools:${NC}"
echo "  • Restore data: ./scripts/install/restore_backup.sh"
echo "  • Verify data:  python3 scripts/install/verify_data.py"
echo ""
