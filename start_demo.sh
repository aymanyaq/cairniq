#!/bin/bash
echo "🚀 Starting CairnIQ in DEMO MODE..."
echo "🛡️ Live broker sync (Questrade/Alpaca) is disabled."

DEMO_PROFILE="${DEMO_PROFILE:-demo}"

if [[ ! "$DEMO_PROFILE" =~ ^[A-Za-z0-9_-]{1,80}$ ]]; then
    echo "Invalid DEMO_PROFILE '$DEMO_PROFILE'. Use letters, numbers, underscores, or hyphens."
    exit 1
fi

echo "📊 Using isolated demo profile: user_data/profiles/${DEMO_PROFILE}"
export DEMO_MODE=true
export CAIRNIQ_FORCE_DEMO=true
export DEMO_RESET="${DEMO_RESET:-true}"
export DEMO_PROFILE
export ACTIVE_PROFILE="$DEMO_PROFILE"
export QUESTRADE_ENABLED=false
export ALPACA_PAPER_MODE=true

.venv/bin/python server.py
