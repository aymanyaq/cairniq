#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# CairnIQ — Backup Restoration Utility
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
BACKUP_DIR="${PROJECT_ROOT}/backups"

cd "$PROJECT_ROOT"

# Header
clear
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  CairnIQ — Data Restoration${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

if [[ ! -d "$BACKUP_DIR" ]]; then
    error "Backup directory not found at ${BACKUP_DIR}"
fi

# List backups
step "Available Backups:"
backups=($(ls -1 "${BACKUP_DIR}"/user_data_*.tar.gz 2>/dev/null | sort -r))

if [[ ${#backups[@]} -eq 0 ]]; then
    error "No backups found in ${BACKUP_DIR}"
fi

for i in "${!backups[@]}"; do
    echo "  [$i] $(basename "${backups[$i]}")"
done

echo ""
echo -n "Select a backup to restore [0-$((${#backups[@]} - 1))]: "
read -r choice

if [[ ! "$choice" =~ ^[0-9]+$ ]] || [[ "$choice" -lt 0 ]] || [[ "$choice" -ge ${#backups[@]} ]]; then
    error "Invalid selection"
fi

SELECTED_BACKUP="${backups[$choice]}"

echo ""
warn "CAUTION: This will overwrite your current user_data/ directory."
echo -n "Are you sure you want to proceed? [y/N]: "
read -r confirm

if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    info "Restoration cancelled"
    exit 0
fi

# Create safety backup of CURRENT data before restoring
step "Creating safety backup of current data..."
SAFETY_BACKUP="${BACKUP_DIR}/user_data_PRE_RESTORE_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$SAFETY_BACKUP" user_data/ 2>/dev/null || true
info "Safety backup created: $(basename "$SAFETY_BACKUP")"

# Restore
step "Restoring backup..."
rm -rf user_data/
mkdir -p user_data/
tar -xzf "$SELECTED_BACKUP" -C .
info "Restoration complete!"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "  Data restored from: $(basename "$SELECTED_BACKUP")"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
