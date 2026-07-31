#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# CairnIQ — Package Creator
# ═══════════════════════════════════════════════════════════════
# Creates a distributable package of the application
# Usage: ./scripts/package/create_package.sh [version]
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() { echo -e "${GREEN}✓${NC} $*"; }
step() { echo -e "${CYAN}▸${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }

# Get version from argument or default
VERSION="${1:-2.1.0}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
PACKAGE_NAME="cairniq-v${VERSION}-${TIMESTAMP}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$PROJECT_ROOT"

# Header
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Creating Package: ${PACKAGE_NAME}${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Create temp directory
TEMP_DIR=$(mktemp -d)
PACKAGE_DIR="${TEMP_DIR}/${PACKAGE_NAME}"

step "Creating package directory..."
mkdir -p "$PACKAGE_DIR"

# Copy files
step "Copying project files..."
rsync -a \
    --exclude='.venv' \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='.git' \
    --exclude='*.sqlite*' \
    --exclude='logs/' \
    --exclude='tmp/' \
    --exclude='backups/' \
    --exclude='user_data/' \
    --exclude='data/cache/' \
    --exclude='data/embeddings/' \
    --exclude='.env' \
    --exclude='.env.local' \
    --exclude='.env.*.local' \
    --exclude='*.env.backup' \
    --exclude='.env.lock' \
    --exclude='my_portfolio.csv' \
    --exclude='chat_history.json' \
    --exclude='user_memory.json' \
    --exclude='knowledge_graph.json' \
    --exclude='feedback.json' \
    --exclude='portfolio_history.csv' \
    --exclude='*.tar.gz' \
    --exclude='*.zip' \
    --exclude='tools/guru_feed.py' \
    --exclude='landing_page/' \
    "${PROJECT_ROOT}/" \
    "${PACKAGE_DIR}/"

info "Files copied"

# Create package info
step "Creating package info..."
cat > "${PACKAGE_DIR}/PACKAGE_INFO.txt" << EOF
═══════════════════════════════════════════════════════════════
CairnIQ
═══════════════════════════════════════════════════════════════

Version:        ${VERSION}
Package Date:   $(date +"%Y-%m-%d %H:%M:%S")
Package Name:   ${PACKAGE_NAME}

═══════════════════════════════════════════════════════════════
Installation Instructions
═══════════════════════════════════════════════════════════════

1. Extract this package:
   tar -xzf ${PACKAGE_NAME}.tar.gz
   cd ${PACKAGE_NAME}

2. Run the installer:
   ./scripts/install/install.sh

3. Follow the on-screen instructions

4. See docs/user-guide/INSTALLATION.md for detailed setup

═══════════════════════════════════════════════════════════════
System Requirements
═══════════════════════════════════════════════════════════════

- macOS 11.0+ (Apple Silicon or Intel)
- Python 3.12 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Internet connection

═══════════════════════════════════════════════════════════════
Quick Start
═══════════════════════════════════════════════════════════════

After installation:
1. Edit user_data/.env with your API keys
2. Edit user_data/my_portfolio.csv with your holdings
3. Double-click "CairnIQ.command" on Desktop
4. Open http://localhost:8000 in browser

═══════════════════════════════════════════════════════════════
Documentation
═══════════════════════════════════════════════════════════════

- README.md - Project overview
- docs/user-guide/INSTALLATION.md - Detailed setup
- docs/user-guide/USER_GUIDE.md - Usage instructions
- docs/user-guide/TROUBLESHOOTING.md - Problem solutions

═══════════════════════════════════════════════════════════════
Support
═══════════════════════════════════════════════════════════════

Check logs in logs/ directory
Run health check in the application
Review documentation in docs/

═══════════════════════════════════════════════════════════════
EOF

info "Package info created"

# Create archive
step "Creating tar.gz archive..."
cd "$TEMP_DIR"
tar -czf "${PROJECT_ROOT}/${PACKAGE_NAME}.tar.gz" "${PACKAGE_NAME}/"
info "Archive created: ${PACKAGE_NAME}.tar.gz"

# Create zip archive
step "Creating zip archive..."
zip -r -q "${PROJECT_ROOT}/${PACKAGE_NAME}.zip" "${PACKAGE_NAME}/"
info "Archive created: ${PACKAGE_NAME}.zip"

# Cleanup
step "Cleaning up..."
rm -rf "$TEMP_DIR"
info "Cleanup complete"

# 7. CONFIG BUNDLE (Optional)
echo ""
echo -n "Would you also like to create a Shared Configuration Bundle (for family/multi-machine setup)? [y/N]: "
read -r create_bundle
if [[ "$create_bundle" =~ ^[Yy]$ ]]; then
    if [[ -f "scripts/package/create_config_bundle.sh" ]]; then
        bash "scripts/package/create_config_bundle.sh"
    else
        warn "Config bundle script missing."
    fi
fi

# Summary
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Package Created Successfully!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "  Package Name:  ${PACKAGE_NAME}"
echo "  Version:       ${VERSION}"
echo ""
echo "  Files Created:"
echo "    • ${PACKAGE_NAME}.tar.gz"
echo "    • ${PACKAGE_NAME}.zip"
echo ""
echo "  Package Size:"
tar_size=$(du -h "${PROJECT_ROOT}/${PACKAGE_NAME}.tar.gz" | cut -f1)
zip_size=$(du -h "${PROJECT_ROOT}/${PACKAGE_NAME}.zip" | cut -f1)
echo "    • tar.gz: ${tar_size}"
echo "    • zip:    ${zip_size}"
echo ""
echo "  Distribution:"
echo "    • Share the .tar.gz or .zip file"
echo "    • Recipients run: ./scripts/install/install.sh"
echo ""
