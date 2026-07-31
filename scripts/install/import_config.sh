#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# CairnIQ — Config Import Utility
# ═══════════════════════════════════════════════════════════════
# Applies a shared configuration bundle to the current machine.
# Usage: ./scripts/install/import_config.sh [bundle_path]
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${GREEN}✓${NC} $*"; }
warn()  { echo -e "${YELLOW}⚠${NC} $*"; }
error() { echo -e "${RED}✗${NC} $*"; exit 1; }
step()  { echo -e "${CYAN}▸${NC} $*"; }

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$PROJECT_ROOT"

# Check arguments
BUNDLE_PATH="${1:-}"
if [[ -z "$BUNDLE_PATH" ]]; then
    echo "Usage: ./scripts/install/import_config.sh path/to/config_bundle.zip"
    exit 1
fi

if [[ ! -f "$BUNDLE_PATH" ]]; then
    error "Bundle not found at $BUNDLE_PATH"
fi

# Header
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  CairnIQ — Importing Configuration${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Extract to temp
TEMP_DIR=$(mktemp -d)
step "Extracting bundle..."
unzip -q "$BUNDLE_PATH" -d "$TEMP_DIR"

# Find the bundle folder inside
BUNDLE_INNER_DIR=$(find "$TEMP_DIR" -maxdepth 1 -type d -name "cairniq_config_*" | head -n 1)

if [[ -z "$BUNDLE_INNER_DIR" ]]; then
    error "Invalid bundle format (inner folder missing)"
fi

# Check for .env
if [[ -f "${BUNDLE_INNER_DIR}/.env" ]]; then
    step "Applying environment settings (.env)..."
    mkdir -p user_data
    if [[ -f "user_data/.env" ]]; then
        # Create backup of current .env before merging/overwriting
        cp "user_data/.env" "user_data/.env.bak_$(date +%Y%m%d_%H%M%S)"
    fi
    cp "${BUNDLE_INNER_DIR}/.env" "user_data/.env"
    info "Environment settings applied"
fi

# Check for global_rules.json
if [[ -f "${BUNDLE_INNER_DIR}/global_rules.json" ]]; then
    step "Applying global rules..."
    cp "${BUNDLE_INNER_DIR}/global_rules.json" "user_data/"
    info "Global rules applied"
fi

# Verification
step "Verifying infrastructure..."
if [[ -f "scripts/install/verify_data.py" ]]; then
    python3 scripts/install/verify_data.py || warn "Configuration verification failed (integrity check)"
fi

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "  Import Complete! Your infrastructure is now synchronized."
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
