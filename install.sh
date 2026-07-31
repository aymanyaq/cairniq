#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# CairnIQ — Main Installer
# ═══════════════════════════════════════════════════════════════
# This is a convenience wrapper that calls the actual installer
# Usage: ./install.sh
# ═══════════════════════════════════════════════════════════════

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Call the actual installer
"${SCRIPT_DIR}/scripts/install/install.sh" "$@"
