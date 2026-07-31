# Troubleshooting Guide

Solutions to common issues with CairnIQ.

## Installation Issues

### Python Version Error

**Problem**: `Python 3.12+ required`

**Solution**:
```bash
# Check version
python3 --version

# Install from python.org if needed
# Or use Homebrew:
brew install python@3.12
```

### Dependency Build Failures (`lxml` / `pyarrow`)

**Problem**: `pip install -r requirements.txt` fails compiling `lxml`, or reports "No matching distribution found" for `pyarrow`.

**Solution**:
```bash
# lxml (>=6): a source build needs the XML dev headers
#   Debian/Ubuntu:
sudo apt-get install libxml2-dev libxslt1-dev
#   macOS (if the Xcode Command Line Tools aren't enough):
brew install libxml2 libxslt

# pyarrow (>=24) ships wheels for Python 3.11-3.13; numpy (>=2.5) is the
# tighter constraint and only ships for 3.12-3.13.
# "No matching distribution" almost always means an unsupported Python:
python3 --version   # must be 3.12 or 3.13
```

### Virtual Environment Creation Fails

**Problem**: `Failed to create virtual environment`

**Solution**:
```bash
# Remove existing venv
rm -rf .venv

# Ensure python3-venv is available
python3 -m venv --help

# Retry installation
./install.sh
```

### Permission Denied

**Problem**: `Permission denied: ./install.sh`

**Solution**:
```bash
chmod +x install.sh
chmod +x CairnIQ.command
./install.sh
```

### FAISS Installation Fails

**Problem**: `Failed to install faiss-cpu`

**Solution**:
```bash
source .venv/bin/activate

# Try CPU version
pip install faiss-cpu

# If that fails, system will use fallback (keyword search)
```

## Startup Issues

### Port 8000 Already in Use

**Problem**: `Address already in use: 8000`

**Solution**:
```bash
# Find and kill process
lsof -ti:8000 | xargs kill -9

# Or use different port
PORT=8001 python server.py
```

### Server Won't Start

**Problem**: Server crashes immediately

**Solution**:
1. Check logs:
   ```bash
   tail -50 logs/server/server.jsonl
   ```

2. Verify `.env` file exists:
   ```bash
   ls -la .env
   ```

3. Check Python environment:
   ```bash
   source .venv/bin/activate
   python --version
   ```

4. Reinstall dependencies:
   ```bash
   ./install.sh
   ```

### Browser Doesn't Open

**Problem**: Server starts but browser doesn't open

**Solution**:
- Manually open: http://localhost:8000
- Check if default browser is set
- Try different browser

## API & Authentication Issues

### AWS Bedrock Errors

**Problem**: `UnrecognizedClientException`, `NoCredentialsError`, or `Access denied`

The most common causes are (1) the keychain didn't hydrate AWS creds into `os.environ` at startup, or (2) a stale `AWS_SESSION_TOKEN` from a prior SSO session is being signed onto long-term IAM-user keys (AWS rejects this combination).

**Solution**:

1. Look at the server's startup output. You should see a line like:
   ```
   [secrets] hydrated 13 from keychain
   ```
   If you see `keychain unavailable` instead, the keyring backend isn't reachable — fall back to writing keys directly to `user_data/.env`.

2. Confirm the AWS keys are actually in the keychain:
   ```bash
   .venv/bin/python -c "
   import keyring
   for k in ('AWS_ACCESS_KEY_ID','AWS_SECRET_ACCESS_KEY'):
       v = keyring.get_password('cairniq', k)
       print(f'{k}: {\"present\" if v else \"MISSING\"} (len={len(v) if v else 0})')
   "
   ```

3. Sanity-check by hitting STS directly:
   ```bash
   .venv/bin/python -c "
   import os, boto3
   from tools.secrets_store import load_secrets_into_env, clear_incompatible_aws_session_token
   load_secrets_into_env()
   clear_incompatible_aws_session_token()
   print(boto3.client('sts', region_name='us-east-1').get_caller_identity())
   "
   ```

4. If STS authenticates but Bedrock still rejects: verify Bedrock model access is granted in the AWS Console for your account and region.

### Settings Page Shows Empty Fields

**Problem**: You open Settings and the API key fields look empty even though they were configured before. Clicking Save risks wiping them.

The app reads field values from `os.environ`, which is hydrated from the OS keychain at startup. If hydration silently fails (rare keychain unlock issue, no backend available, etc.), the form renders empty.

**Solution**:
1. **Do not click Save** while the fields are empty — earlier versions could have cleared the keychain. Current versions guard against this (an empty field is treated as no-op when a value is already stored), but it's still bad UX.
2. Restart the server and check the startup log for the `[secrets] hydrated N from keychain` line.
3. Open Keychain Access (macOS) or Credential Manager (Windows) and search for `cairniq` — confirm entries are present.
4. If entries are gone (deletion happened on a prior version): re-enter them in the Settings page or restore from a backup.

### API Rate Limits

**Problem**: `Rate limit exceeded` or `429 Too Many Requests`

**Solution**:
- Wait for the cooldown — CairnIQ marks the affected provider as rate-limited for 5 minutes and surfaces a clear note to the agent.
- Many analysis tools automatically fall back to a secondary data source (e.g., yfinance) while the primary is in cooldown — the answer will be best-effort with a noted caveat.
- Hitting limits repeatedly? Upgrade your plan in the provider's dashboard. Each provider's Terms of Service governs how its keys may be used.

### Invalid API Keys

**Problem**: `Invalid API key` or `401 Unauthorized`

**Solution**:
1. Regenerate the key in the provider's console
2. Update it via Settings → **Commit Infrastructure Sync** (writes to keychain)
3. Restart the application
4. Run a health check to verify

### Azure OpenAI / AI Foundry: `DeploymentNotFound`

**Problem**: `404 ... 'code': 'DeploymentNotFound', 'message': 'The API deployment <name> does not exist'`

This is **not** about regional availability. It means the request reached an Azure resource that has no deployment by that **exact name**. Two things must line up: the deployment *name* and the *resource (endpoint)* the request hits. On Azure, model id fields hold **deployment names** (e.g. `AIDLC_MODEL_ID_AZURE`, `AIDLC_SONNET_MODEL_ID_AZURE`), and each tier is routed independently. Common causes:

- The configured name doesn't match the deployment exactly (case-sensitive; watch for version-date suffixes shown in the model *catalog* vs the actual *deployment* name).
- A tier is pointed at the wrong resource — e.g. the fast tier's `AZURE_OPENAI_ENDPOINT_FAST` aims at a resource that only hosts **embeddings**, while the chat deployments live on the primary resource.

**Solution**:
1. List what's actually deployed on a resource (v1 surface):
   ```bash
   curl -s https://<resource>.services.ai.azure.com/openai/v1/models \
     -H "Authorization: Bearer $AZURE_OPENAI_API_KEY" | python3 -m json.tool | grep '"id"'
   ```
   Caveat: a *project* endpoint (`/api/projects/<p>/openai/v1`) may not support `/models`, and `/models` lists the **catalog** (deployable) not necessarily what's **deployed** — confirm by sending a real chat completion to the deployment name.
2. Verify the chat resource (`AZURE_OPENAI_ENDPOINT`) hosts your chat deployment names, and that the fast tier (`AZURE_OPENAI_ENDPOINT_FAST`, if set) does too. Embeddings live on their own resource (`AZURE_OPENAI_ENDPOINT_EMBEDDING`) and only host the embedding deployment — never point a chat tier there.
3. If a tier is misrouted, remove the `_FAST` override so it shares the primary chat resource, or repoint it at the resource that actually hosts the deployment. (Keys live in the keychain under service `cairniq`; a blank value in `.env` falls through to the keychain copy.)
4. Restart so the new env/keychain values load.

### Azure Throttling (429): you are token-limited, not request-limited

**Problem**: `429`, repeated `⏳ Stream throttled, waiting 4s/8s/16s...`, planner/synthesis fails after backoff.

Azure deployments enforce both a **TPM** (tokens/min) and **RPM** (requests/min) cap — whichever you hit first. For reasoning/synthesis workloads the **TPM cap is almost always the binding one**: a single synthesis call can be 15–20k+ *input* tokens, exceeding a default 20k-TPM deployment by itself. A client-side rate limiter cannot fix this — it paces *requests*, not *tokens*.

**Solution**:
1. Confirm the model and prompt size — usage is logged per call:
   ```bash
   grep '"TokenUsage"' logs/agent/agent.jsonl | tail -20
   ```
   Inspect `model_id`, `input_tokens`, `tokens_last_60s`, and `throttle_risk`. (Quota is **per model**, so the fast/planner and primary/synthesis tiers have independent ceilings.)
2. **Raise the deployment's capacity** — the real fix. Azure AI Foundry → deployment → **Edit → Tokens-per-Minute**. In the capacity-units dialog, **1 unit = 1,000 TPM** (enter `60` for 60k). For Standard ("Azure Direct") deployments this is a rate ceiling drawn from your subscription quota — you still pay per token *used*, not per unit allocated.
3. If you can't raise capacity: reduce tokens per call (less context into synthesis) and/or deploy the same model in a second region.
4. Optional code-side knobs (env vars, read live):
   - `LLM_MAX_RETRIES` (default `8`) — retry budget; the SDK honors `Retry-After`.
   - `LLM_MAX_RPS` (default `0` = off) — per-endpoint request ceiling. Only helps when you're RPM-bound.
   - `LLM_TPM_LIMIT` (default `20000`) — sets the `throttle_risk` threshold in the token logs.

## Performance Issues

### Slow Response Times

**Problem**: AI takes too long to respond

**Solution**:
1. Check internet speed
2. Run health check to identify slow tools
3. Reduce portfolio size
4. Use Concise mode instead of Deep mode
5. Check system resources:
   ```bash
   top -l 1 | grep python
   ```

### High Memory Usage

**Problem**: Application using too much RAM

**Solution**:
1. Restart the application
2. Clear old chat history (active profile only):
   ```bash
   rm user_data/chat_history.json
   ```
   Or, for a specific profile:
   ```bash
   rm user_data/profiles/<profile_name>/chat_history.json
   ```
3. Reduce portfolio size
4. Clear daily cache:
   ```bash
   rm -rf user_data/daily_cache/*
   ```

### Browser Lag

**Problem**: UI is slow or unresponsive

**Solution**:
- Clear browser cache
- Close other tabs
- Try different browser
- Disable browser extensions
- Restart browser

## Data Issues

### Portfolio Not Loading

**Problem**: Portfolio shows $0 or empty

**Solution**:
1. Check file exists:
   ```bash
   ls -la user_data/my_portfolio.csv
   ```

2. Verify format:
   ```bash
   head user_data/my_portfolio.csv
   ```

3. Check for errors:
   ```bash
   tail -20 logs/server/server.jsonl | grep portfolio
   ```

4. Validate CSV format (no extra commas, proper dates)

### Questrade Sync Failing

**Problem**: Questrade portfolio not syncing

**Solution**:
1. Verify refresh token in `.env`
2. Check token hasn't expired
3. Regenerate token in Questrade
4. Check logs:
   ```bash
   tail -50 logs/tools/tools.jsonl | grep questrade
   ```

### Knowledge Graph Fails to Load

**Problem**: `⚠️ Failed to load Knowledge Graph from .../knowledge_graph.json: Expecting property name enclosed in double quotes`

Almost always a **transient concurrent-write race** — one worker read the file while another was mid-write (truncated). Current versions write atomically (temp file + `os.replace`) and retry the decode on load, so this should self-heal and the file is usually valid moments later. If the message persists, the file is genuinely corrupt.

**Solution**:
1. Validate the file:
   ```bash
   .venv/bin/python -c "import json; json.load(open('user_data/knowledge_graph.json')); print('valid')"
   ```
2. If it's invalid, restore from a backup tarball in `backups/`, or move it aside to start fresh (it rebuilds from portfolio + memory):
   ```bash
   mv user_data/knowledge_graph.json user_data/knowledge_graph.json.corrupt
   ```
3. Restart the application.

### Stock Universe Screening & Sourcing

**Overview**: Opportunity scans and sector screens use a 100% dynamic market screener engine (powered by keyless TradingView, Yahoo Finance, FMP, SEC 13F filings, and intraday movers) covering all 11 GICS sectors across both US (NYSE/NASDAQ) and Canadian (TSX/TSX-V) markets.

**Note**: Legacy static stock universe files (`stock_universe.json`) have been retired. Candidate discovery is performed dynamically at runtime, eliminating static ticker list maintenance.

### Stale Market Data

**Problem**: Prices not updating

**Solution**:
1. Run health check
2. Verify API keys in Settings (or in the OS keychain)
3. Check API rate limits in the provider dashboard
4. Clear daily cache:
   ```bash
   rm -rf user_data/daily_cache/*
   ```
5. Restart application

**Note on "Real-time" labels**: a provider's `data_freshness` field describes the *source*, not the observation you're looking at. A quote served from cache can still carry a "Real-time" label hours after it was actually fetched. Alerts do not trust that label — they use the fetch timestamp recorded inside the payload — but chat answers may echo it. If a price looks wrong, the cache clear above is the fastest check.

Some providers also return end-of-day data on free key tiers (Alpha Vantage `GLOBAL_QUOTE` in particular). That is annotated rather than rejected — on a weekend, holiday, or pre-open the prior close *is* the right answer — so look for the staleness note attached to the quote.

## Alerts & Scheduler Issues

### No Alerts Are Firing

**Problem**: The Alerts inbox stays empty, or a level you're watching was crossed and nothing arrived.

Work through these in order — the first two account for most cases:

1. **The scheduler is off.** Background work is opt-in per profile. Turn on the **Scheduler** toggle in Settings, then restart the server. With it off, no brief, scan, or monitor runs at all.
2. **The first run is deliberately silent.** Both zero-LLM monitors record what they see on their first tick as a baseline, and alert only on a subsequent *change*. A level that was already true when monitoring started is not announced.
3. **It's outside market hours.** Watch conditions and the intraday sentinel only tick while the market is open.
4. **The data couldn't be proved fresh.** An alert will not fire on a quote older than 45 minutes or on a daily bar from an earlier session — the reading is skipped rather than acted on, so the crossing stays live for the next tick. Suppression counts are written into the run record; check `logs/tools/tools.jsonl` for the monitor's run entries.
5. **It already fired.** Watch conditions are terminal: a level alerts once and retires. Check `GET /api/watch_conditions` for what's still pending.

### The Scheduler Runs But Produces Nothing

Check `user_data/profiles/<profile>/scheduler_runs.json` — it holds the per-task cooldown and daily-marker registry. A task that shows a recent run but produced no output failed internally; its error will be in `logs/tools/tools.jsonl`.

## Advice & Risk Issues

### The Advisor Cites a Limit I Never Set

This was a real bug and is fixed: a "2% risk rule" and a "10% concentration cap" were hardcoded and cited back to users who had never set them. Risk limits now come only from `risk_constraints` in your profile. If you still see a cap quoted as yours, check `user_data/profiles/<profile>/user_memory.json` — and if it isn't there, that's a regression worth reporting.

### No Limits Are Being Enforced

Expected, if you haven't stated any. There are no house defaults: an unset limit enforces nothing. Set them via `POST /api/memory/risk_constraints` (`max_position_pct`, `max_fund_position_pct`, `max_sector_pct`, `max_risk_per_trade_pct`, `restricted_symbols`).

### Advice Gets Blocked or Rewritten

Advice-generating turns pass a review gate: deterministic grounding and compliance pre-checks, then an LLM judge. A draft that claims a number it can't source, references a holding you don't own, or breaches a cap you set gets capped and retried. Verdicts are recorded per profile in `risk_verdicts.jsonl` — read that file to see exactly which rule fired.

## Tool Issues

### Health Check Shows Failures

**Problem**: Multiple tools showing ❌ Failed

**Solution**:
1. Check specific error messages
2. Verify all API keys in `.env`
3. Test internet connection
4. Check API provider status pages
5. Review logs:
   ```bash
   tail -100 logs/tools/tools.jsonl
   ```

### Specific Tool Always Fails

**Problem**: One tool consistently fails

**Solution**:
1. Check tool-specific API key
2. Verify API quota not exceeded
3. Check provider status page
4. Review tool logs for error details
5. System will use fallback if available

### Opportunity Scanner Stuck

**Problem**: Scanner runs forever

**Solution**:
1. Click stop button
2. Wait 30 seconds
3. Restart application if needed
4. Try scanning single sector instead of all

## UI Issues

### Spinner Won't Stop

**Problem**: Loading spinner persists after completion

**Solution**:
- Click stop button
- Refresh browser page
- Clear browser cache
- Restart application

### Chat History Lost

**Problem**: Previous conversations disappeared

**Solution**:
- Check the active profile's history file exists: `user_data/chat_history.json` (default profile) or `user_data/profiles/<profile_name>/chat_history.json`
- Restore from the most recent backup tarball in `backups/`
- Confirm you haven't switched to a different profile via the profile cookie or `ACTIVE_PROFILE` env var

### Icons Not Showing

**Problem**: Custom icon not appearing

**Solution**:
1. Restart Finder:
   ```bash
   killall Finder
   ```
2. Clear icon cache:
   ```bash
   sudo rm -rf /Library/Caches/com.apple.iconservices.store
   ```
3. Reapply icon manually (see APPLY_ICON_INSTRUCTIONS.md)

## Network Issues

### Connection Timeout

**Problem**: `Connection timeout` errors

**Solution**:
- Check internet connection
- Verify firewall settings
- Try different network
- Check if APIs are accessible:
  ```bash
  curl -I https://www.alphavantage.co
  ```

### SSL Certificate Errors

**Problem**: `SSL certificate verify failed`

**Solution**:
```bash
# Update certificates
pip install --upgrade certifi

# Or temporarily disable verification (not recommended)
export PYTHONHTTPSVERIFY=0
```

**On macOS framework Python**, this can fail even with `certifi` installed: the interpreter reports no system CA file at all, so any raw `urllib` HTTPS call fails verification while `requests` (which carries its own bundle) succeeds. Check with:

```bash
python3 -c "import ssl; print(ssl.get_default_verify_paths().cafile)"
```

If that prints `None`, run the `Install Certificates.command` shipped in your Python's `/Applications/Python 3.x/` folder. This symptom is quiet rather than loud — it doesn't crash the app, it just empties whatever the failing call was fetching (it previously zeroed the scanner's ticker universe), so suspect it when a scan returns nothing on one machine and works on another.

## Database Issues

### Checkpoints Database Locked

**Problem**: `database is locked`

**Solution**:
```bash
# Stop application
# Remove lock files
rm checkpoints.sqlite-shm checkpoints.sqlite-wal

# Restart application
```

### Corrupted Database

**Problem**: Database errors on startup

**Solution**:
```bash
# Backup current database
cp checkpoints.sqlite checkpoints.sqlite.backup

# Remove and let it recreate
rm checkpoints.sqlite*

# Restart application
```

## Logging & Debugging

### Enable Debug Logging

```bash
# Edit .env
DEBUG=true

# Restart application
```

### View Real-Time Logs

```bash
# Server logs
tail -f logs/server/server.jsonl

# Tool logs
tail -f logs/tools/tools.jsonl

# Frontend logs
tail -f logs/frontend/frontend.jsonl
```

### Search Logs for Errors

```bash
# Find errors in last hour
grep -i error logs/server/server.jsonl | tail -50

# Find specific tool errors
grep "tool_name" logs/tools/tools.jsonl
```

## Getting More Help

### Run System Diagnostics

In the chat interface, type a plain-English request:
```
Run a full system health check
```

### Check System Status

```bash
# Check if server is running
lsof -i :8000

# Check Python process
ps aux | grep python

# Check disk space
df -h

# Check memory
vm_stat
```

### Collect Debug Information

Before reporting issues:

1. Run health check
2. Collect logs:
   ```bash
   tar -czf debug-logs.tar.gz logs/
   ```
3. Note error messages
4. Note steps to reproduce

### Reset to Clean State

If all else fails:

```bash
# Stop application
# Backup important data
cp user_data/my_portfolio.csv my_portfolio.backup.csv
cp user_data/.env .env.backup

# Clean install
rm -rf .venv logs/ backups/
./install.sh

# Restore data
cp my_portfolio.backup.csv user_data/my_portfolio.csv
cp .env.backup user_data/.env
```

## Common Error Messages

### "Model not found"
- Check `AIDLC_MODEL_ID` in `.env`
- Verify model access in AWS Bedrock console

### "DeploymentNotFound" (Azure)
- The deployment name doesn't match, or a tier is routed to a resource that doesn't host it
- See **Azure OpenAI / AI Foundry: `DeploymentNotFound`** above

### "Expecting property name enclosed in double quotes"
- A JSON data file (usually `knowledge_graph.json`) was read mid-write, or is corrupt
- See **Knowledge Graph Fails to Load** above

### "No module named 'X'"
- Reinstall dependencies: `./install.sh`

### "Connection refused"
- Server not running or wrong port
- Check `lsof -i :8000`

### "Out of memory"
- Restart application
- Reduce portfolio size
- Close other applications

### "Permission denied"
- Check file permissions: `chmod +x`
- Check directory permissions

---

**Still Having Issues?**

1. Check logs in `logs/` directory
2. Run health check in the application
3. Review error messages carefully
4. Try clean reinstall as last resort
