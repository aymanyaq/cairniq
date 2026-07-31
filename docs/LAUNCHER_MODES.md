# Launcher Modes: Production vs Demo

**Date:** May 6, 2026  
**Status:** ✅ CONFIGURED

---

## Overview

The system now has two distinct launch modes:

1. **Production Mode** - Your real account (Questrade, live data)
2. **Demo Mode** - Isolated sandbox with sample data

---

## Production Mode (Your Account)

### Desktop Launcher
**File:** `CairnIQ.command`

**What it does:**
- Launches with your real Questrade account
- Uses live market data
- Accesses your actual portfolio
- Saves to `user_data/` (your personal data)

**How to use:**
1. Double-click `CairnIQ.command` on your desktop
2. System starts in production mode automatically
3. Opens browser to http://localhost:8000

**It does not run the server itself.** On macOS the launcher asks launchd to
start `com.cairniq.server` (installing the LaunchAgent first if needed), waits for
the port, then opens the browser. That keeps exactly one supervised code path into
the server: a directly-run instance would win the port race, launchd's own start
would hit the single-instance guard in `server.py` and exit 0, and
`KeepAlive {SuccessfulExit: false}` would leave the job dormant — an unsupervised
server that stays down permanently once it dies.

Consequences worth knowing:
- **Closing the Terminal window does not stop the server.** Stop it with
  `launchctl bootout gui/$(id -u)/com.cairniq.server`.
- The server keeps running after logout and restarts on login.
- If `scripts/cairniq_watchdog.py` disabled the service after a runaway, the
  launcher shows the alert and asks before re-enabling it.
- On platforms without launchd (Linux), the launcher runs the server in the
  window as before — there is no supervision to forfeit.

**Environment:**
- `DEMO_MODE`: Not set (production)
- `QUESTRADE_ENABLED`: true (from your .env)
- `ACTIVE_PROFILE`: Your account name
- Data location: `user_data/`

---

## Demo Mode (Sandbox)

### Demo Launcher
**File:** `start_demo.sh`

**What it does:**
- Launches with isolated demo profile
- Uses sample portfolio data
- Disables live broker sync
- Saves to `user_data/profiles/demo/`

**How to use:**
```bash
./start_demo.sh
```

**Environment:**
- `DEMO_MODE`: true
- `CAIRNIQ_FORCE_DEMO`: true
- `QUESTRADE_ENABLED`: false
- `ACTIVE_PROFILE`: demo
- Data location: `user_data/profiles/demo/`

---

## Key Differences

| Feature | Production Mode | Demo Mode |
|---------|----------------|-----------|
| **Launcher** | `CairnIQ.command` | `start_demo.sh` |
| **Questrade** | ✅ Live sync | ❌ Disabled |
| **Portfolio** | Your real holdings | Sample data |
| **Market Data** | Live APIs | Live APIs |
| **Data Storage** | `user_data/` | `user_data/profiles/demo/` |
| **Risk** | Real money | No risk |

---

## How It Works

### Production Mode Detection

The system checks these environment variables:
```python
def is_demo_mode() -> bool:
    return os.environ.get("DEMO_MODE") == "true" or \
           os.environ.get("CAIRNIQ_FORCE_DEMO") == "true"
```

**Production launcher explicitly unsets these:**
```bash
unset DEMO_MODE
unset CAIRNIQ_FORCE_DEMO
unset DEMO_PROFILE
unset DEMO_RESET
```

**Demo launcher explicitly sets these:**
```bash
export DEMO_MODE=true
export CAIRNIQ_FORCE_DEMO=true
export DEMO_PROFILE=demo
export QUESTRADE_ENABLED=false
```

---

## Verification

### Check Current Mode

When the server starts, look for this line:
```
✓ Mode: Production (Your Account)
```

Or in the logs:
```bash
grep "DEMO_MODE\|ACTIVE_PROFILE" logs/server/server.jsonl | tail -1
```

**Production mode:**
```json
{"active_profile": "default", "demo_mode": false}
```

**Demo mode:**
```json
{"active_profile": "demo", "demo_mode": true}
```

### Check in UI

1. Open Settings page
2. Look at "Account Owner" field
3. **Production:** Shows your configured account owner
4. **Demo:** Shows "demo"

---

## Safety Features

### Production Mode Safeguards

1. **Explicit unset** - Desktop launcher clears all demo variables
2. **Environment file** - Loads `user_data/.env` with your credentials
3. **Profile isolation** - Your data stays in `user_data/`
4. **Questrade sync** - Only enabled when `QUESTRADE_ENABLED=true` in your .env

### Demo Mode Safeguards

1. **Isolated storage** - Demo data in separate `profiles/demo/` folder
2. **Broker disabled** - Cannot accidentally trade
3. **Explicit flags** - Multiple environment variables must be set
4. **Reset option** - Can reset demo data without affecting production

---

## Troubleshooting

### "I launched production but it's in demo mode"

**Check:**
1. Did you use `CairnIQ.command` (not `start_demo.sh`)?
2. Are demo variables set in your shell?
   ```bash
   echo $DEMO_MODE
   echo $CAIRNIQ_FORCE_DEMO
   ```
3. Restart terminal and try again

**Fix:**
```bash
unset DEMO_MODE
unset CAIRNIQ_FORCE_DEMO
# Then double-click CairnIQ.command
```

### "I launched demo but it's using my real account"

**This should not happen** - demo launcher explicitly disables Questrade.

**If it does:**
1. Stop the server immediately
2. Check `start_demo.sh` has these lines:
   ```bash
   export DEMO_MODE=true
   export QUESTRADE_ENABLED=false
   ```
3. Report as a bug

---

## File Locations

### Production Data
```
user_data/
├── .env                    # Your credentials
├── my_portfolio.csv        # Your holdings
├── memory/                 # Your chat history
└── trade_journal.csv       # Your trades
```

### Demo Data
```
user_data/profiles/demo/
├── portfolio.csv           # Sample holdings
├── memory/                 # Demo chat history
└── trade_journal.csv       # Demo trades
```

---

## Best Practices

### For Development/Testing
- Use **demo mode** (`start_demo.sh`)
- Test new features safely
- Reset demo data anytime: `DEMO_RESET=true ./start_demo.sh`

### For Real Trading/Analysis
- Use **production mode** (`CairnIQ.command`)
- Double-check you're in production before making decisions
- Verify "Account Owner" in settings

### For Sharing/Screenshots
- Use **demo mode** to avoid exposing real data
- Demo mode safe for screen sharing
- No real credentials or holdings visible

---

## Summary

**Desktop launcher (`CairnIQ.command`):**
- ✅ Your real account
- ✅ Live Questrade sync
- ✅ Real portfolio data
- ✅ Production mode

**Demo script (`start_demo.sh`):**
- ✅ Isolated sandbox
- ✅ Sample data
- ✅ No broker access
- ✅ Demo mode

**The desktop launcher now explicitly ensures production mode** by unsetting all demo variables before starting the server.
