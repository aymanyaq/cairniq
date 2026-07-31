#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# CairnIQ — Config Bundle Creator
# ═══════════════════════════════════════════════════════════════
# Bundles infrastructure settings (API keys, AWS, Rules)
# while EXCLUDING all personal user data (Memory, Portfolio).
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

info() { echo -e "${GREEN}✓${NC} $*"; }
step() { echo -e "${CYAN}▸${NC} $*"; }

# Get version from argument or default
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BUNDLE_NAME="cairniq_config_${TIMESTAMP}"

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$PROJECT_ROOT"

# Create temp directory
TEMP_DIR=$(mktemp -d)
BUNDLE_DIR="${TEMP_DIR}/${BUNDLE_NAME}"
mkdir -p "$BUNDLE_DIR"

step "Preparing configuration bundle..."

# 1. Environment Variables (Most important)
if [[ -f "user_data/.env" ]]; then
    cp "user_data/.env" "$BUNDLE_DIR/"
    info "Included user_data/.env"
else
    step "user_data/.env not found, checking root..."
    if [[ -f ".env" ]]; then
        cp ".env" "$BUNDLE_DIR/"
        info "Included .env (root)"
    fi
fi

# 2. Global Rules
if [[ -f "user_data/global_rules.json" ]]; then
    cp "user_data/global_rules.json" "$BUNDLE_DIR/"
    info "Included global_rules.json"
fi

# 3. Create Manifest
cat > "${BUNDLE_DIR}/CONFIG_MANIFEST.txt" << EOF
═══════════════════════════════════════════════════════════════
CairnIQ — Configuration Bundle
═══════════════════════════════════════════════════════════════

Bundle ID:   ${BUNDLE_NAME}
Created:     $(date +"%Y-%m-%d %H:%M:%S")

This bundle contains SHARED INFRASTRUCTURE only:
- API Keys & LLM Settings (.env)
- Global Rules & Preferences (global_rules.json)

EXCLUDED (Private Data):
- Portfolio (my_portfolio.csv)
- Chat History (chat_history.json)
- Knowledge Graph (checkpoints.sqlite)
- User Memory (user_memory.json)

═══════════════════════════════════════════════════════════════
Usage:
Provide this bundle to the install.sh script on a new machine
or run ./scripts/install/import_config.sh cairniq_config.zip
═══════════════════════════════════════════════════════════════
EOF

# Create archive
step "Creating archive..."
cd "$TEMP_DIR"
zip -r -q "${PROJECT_ROOT}/${BUNDLE_NAME}.zip" "${BUNDLE_NAME}/"

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Config Bundle Created: ${BUNDLE_NAME}.zip${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
