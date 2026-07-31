#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# CairnIQ — macOS launchd Service Setup
# ═══════════════════════════════════════════════════════════════
# Installs CairnIQ as a user-level launchd agent so it runs
# automatically in the background and restarts on login/reboot.
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}✓${NC} $*"; }
step()  { echo -e "${CYAN}▸${NC} $*"; }
warn()  { echo -e "${YELLOW}⚠${NC} $*"; }
error() { echo -e "${RED}✗${NC} $*"; exit 1; }

# Get script and project directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Ensure plist directory exists
LAUNCH_AGENTS_DIR="${HOME}/Library/LaunchAgents"
mkdir -p "$LAUNCH_AGENTS_DIR"

PLIST_PATH="${LAUNCH_AGENTS_DIR}/com.cairniq.server.plist"

step "Setting up background service..."

# Stop and unload any existing instance first
if launchctl list | grep -q "com.cairniq.server"; then
    step "Stopping existing background service..."
    launchctl unload "$PLIST_PATH" 2>/dev/null || true
fi

# Define paths
PYTHON_PATH="${PROJECT_ROOT}/.venv/bin/python"
SERVER_PATH="${PROJECT_ROOT}/server.py"
# NOTE: scripts/cairniq_watchdog.py parses logs/cairniq.stderr.log to detect a
# restart-storm. Keep these two paths in sync with it.
STDOUT_LOG="${PROJECT_ROOT}/logs/cairniq.stdout.log"
STDERR_LOG="${PROJECT_ROOT}/logs/cairniq.stderr.log"
VENV_DIR="${PROJECT_ROOT}/.venv"

# Verify virtual environment python exists
if [[ ! -x "$PYTHON_PATH" ]]; then
    error "Virtual environment python not found at $PYTHON_PATH. Please run ./install.sh first."
fi

mkdir -p "${PROJECT_ROOT}/logs"

# Create plist file
cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.cairniq.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PYTHON_PATH</string>
        <string>$SERVER_PATH</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$PROJECT_ROOT</string>
    <key>RunAtLoad</key>
    <true/>

    <!-- Restart ONLY on failure (non-zero / signal). A clean exit(0) from the
         single-instance guard in server.py means "another instance already
         owns the port" — do NOT respawn it, or it crash-loops and re-runs the
         heavy startup (ticker downloads, agent init) every cycle. The liveness
         probe in scripts/cairniq_watchdog.py covers the flip side: a job left
         dormant by that clean exit gets kickstarted once the port goes quiet. -->
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <!-- Never respawn faster than this many seconds, even on repeated crashes.
         Caps any residual loop to ~2/min instead of ~30/min. -->
    <key>ThrottleInterval</key>
    <integer>30</integer>

    <key>StandardOutPath</key>
    <string>$STDOUT_LOG</string>
    <key>StandardErrorPath</key>
    <string>$STDERR_LOG</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>${VENV_DIR}/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>VIRTUAL_ENV</key>
        <string>${VENV_DIR}</string>
    </dict>
</dict>
</plist>
EOF

chmod 644 "$PLIST_PATH"
info "Service configuration created at $PLIST_PATH"

# Load the service
step "Loading background service into launchd..."
launchctl load "$PLIST_PATH"

info "Service successfully loaded and started!"
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "  CairnIQ Background Service Status"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  • ${CYAN}Status:${NC}      Running in background, will auto-start on login"
echo -e "  • ${CYAN}Logs (stdout):${NC} tail -f logs/cairniq.stdout.log"
echo -e "  • ${CYAN}Logs (stderr):${NC} tail -f logs/cairniq.stderr.log"
echo ""
echo -e "${YELLOW}Useful Service Commands:${NC}"
echo "  - To STOP the service:   launchctl unload ~/Library/LaunchAgents/com.cairniq.server.plist"
echo "  - To START the service:  launchctl load ~/Library/LaunchAgents/com.cairniq.server.plist"
echo "  - To RESTART the service: launchctl unload ~/Library/LaunchAgents/com.cairniq.server.plist && launchctl load ~/Library/LaunchAgents/com.cairniq.server.plist"
echo ""
