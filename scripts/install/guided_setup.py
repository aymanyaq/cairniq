#!/usr/bin/env python3
import json
import os
import sys

# Ensure UTF-8 output on Windows consoles (Python 3.7+).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Colors for terminal output
GREEN = "\033[0;32m"
CYAN = "\033[0;36m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
BOLD = "\033[1m"
NC = "\033[0m"

def info(msg): print(f"{GREEN}[OK]{NC} {msg}")
def step(msg): print(f"{CYAN} >> {NC} {msg}")
def warn(msg): print(f"{YELLOW}[!]{NC}  {msg}")
def error(msg): print(f"{RED}[X]{NC}  {msg}")
def header(msg):
    print(f"\n{CYAN}{BOLD}{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}{NC}\n")

def get_input(prompt, default=None):
    if default:
        val = input(f"{CYAN}?{NC} {prompt} [{default}]: ").strip()
        return val if val else default
    return input(f"{CYAN}?{NC} {prompt}: ").strip()

def _try_import_secrets_store():
    """Best-effort import of the in-tree secrets_store helper.

    The wizard runs from the repo root via `python scripts/install/guided_setup.py`
    or the install script — `tools.secrets_store` should be importable. If not
    (e.g. the user is running this from a weird CWD), we fall back to writing
    everything to .env so the install never bricks.
    """
    try:
        # Make sure repo root is on sys.path even when launched from elsewhere.
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from tools import secrets_store  # noqa: WPS433 — intentional lazy import
        return secrets_store
    except Exception as e:  # noqa: BLE001
        warn(f"Could not load OS-keychain helper ({e}); secrets will be written to .env instead.")
        return None


def save_env(settings):
    """Persist wizard answers.

    Sensitive credentials (API keys, brokerage tokens) are written to the OS
    keychain via `tools.secrets_store`. Non-sensitive configuration is written
    to `user_data/.env` so it remains human-readable.

    If the keychain isn't usable on this machine (uncommon on Mac/Windows,
    common on headless Linux), the secrets fall back to .env so the install
    still completes successfully.
    """
    env_path = os.path.join("user_data", ".env")

    secrets_store = _try_import_secrets_store()
    secret_keys = set(secrets_store.SECRET_KEYS) if secrets_store else set()

    # Split incoming settings into (config → .env) and (secret → keychain).
    config_updates: dict[str, str] = {}
    secret_updates: dict[str, str] = {}
    for k, v in settings.items():
        if k in secret_keys:
            secret_updates[k] = v
        else:
            config_updates[k] = v

    # 1. Try writing secrets to the keychain. Any that fail get pushed back
    #    into config_updates so they still end up in .env (never lose the value).
    stored_in_keychain: list[str] = []
    if secrets_store and secret_updates:
        for k, v in list(secret_updates.items()):
            if not v:
                continue
            if secrets_store.set_secret(k, v):
                stored_in_keychain.append(k)
            else:
                config_updates[k] = v
        # Record which secret names exist (blank values) in .env so the user
        # can see at a glance what's configured without exposing the bytes.
        for k in stored_in_keychain:
            config_updates.setdefault(k, "")
    elif secret_updates:
        # No keychain available — write secrets to .env as the fallback.
        config_updates.update(secret_updates)

    # 2. Persist config_updates to .env, preserving any pre-existing keys.
    existing: dict[str, str] = {}
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    k, v = line.split('=', 1)
                    existing[k.strip()] = v.strip()

    final = existing.copy()
    final.update(config_updates)

    with open(env_path, 'w', encoding="utf-8") as f:
        f.write("# CairnIQ - Configuration\n")
        f.write("# Created via Guided Setup Wizard\n")
        f.write("# Note: API keys are stored in your OS keychain when available.\n\n")
        for k, v in final.items():
            f.write(f"{k}={v}\n")
    info(f"Saved configuration to {env_path}")
    if stored_in_keychain:
        info(f"Stored {len(stored_in_keychain)} secret(s) in the OS keychain.")

def main():
    header("CAIRNIQ - GUIDED SETUP")

    print("Welcome! This wizard will help you configure CairnIQ.")
    print("Your data stays local and is never shared with us.\n")

    # 0. Import Config (New Step)
    step("Step 0: Existing Configuration")
    has_bundle = get_input("Do you have a shared config bundle (.zip) to import? (y/N)", "n")
    if has_bundle.lower() == 'y':
        bundle_path = get_input("Enter path to bundle .zip file")
        if os.path.exists(bundle_path):
            import subprocess
            print(f"{CYAN} >> {NC} Importing {bundle_path}...")
            try:
                # Call our standalone import utility
                subprocess.run(["bash", "scripts/install/import_config.sh", bundle_path], check=True)
                print(f"\n{GREEN}[OK]{NC} Shared configuration imported successfully.")
                print(f"{YELLOW} >> {NC} You can still go through the wizard to refine your settings.\n")
            except subprocess.CalledProcessError:
                error("Failed to import configuration bundle.")
        else:
            error(f"Bundle file not found at {bundle_path}")

    # 1. Identity
    step("Step 1: Identity & Profile")
    name = get_input("What is your name?")
    age = get_input("What is your age? (Optional)", "")
    goals = get_input("What is your primary financial goal?", "Personal Wealth Management")

    # Update user_memory.json
    mem_path = os.path.join("user_data", "user_memory.json")
    memory = {"user_profile": {}, "key_facts": [], "active_theses": [], "lessons_learned": []}
    if os.path.exists(mem_path):
        try:
            with open(mem_path, encoding="utf-8") as f: memory = json.load(f)
        except: pass

    memory["user_profile"]["name"] = name
    if age: memory["user_profile"]["age"] = age
    memory["user_profile"]["primary_goal"] = goals

    with open(mem_path, 'w', encoding="utf-8") as f:
        json.dump(memory, f, indent=2)
    info("Identity profile initialized")

    # 2. LLM Provider
    header("Section 2: AI Infrastructure")
    print("Choose your primary LLM Provider:")
    print("  [1] AWS Bedrock (Claude on AWS - Recommended)")
    print("  [2] Anthropic (Direct API)")
    print("  [3] OpenAI (Direct API)")
    print("  [4] Azure OpenAI / AI Foundry (gpt-5.x, DeepSeek, grok, Kimi)")
    choice = get_input("Select [1-4]", "1")

    env_updates = {}
    if choice == "1":
        env_updates["LLM_PROVIDER"] = "bedrock"
        env_updates["AWS_ACCESS_KEY_ID"] = get_input("AWS Access Key ID")
        env_updates["AWS_SECRET_ACCESS_KEY"] = get_input("AWS Secret Access Key")
        env_updates["AWS_REGION"] = get_input("AWS Region", "us-east-1")
        # Bedrock model: inference profile or full ARN.
        # We default to the "global." cross-region inference profiles for
        # Claude 4.6 — these route across multiple AWS regions automatically
        # and are the inference-profile IDs Anthropic publishes for 4.6.
        # If you prefer US-only or need to pin to a specific account,
        # you can paste a full ARN here instead.
        env_updates["AIDLC_MODEL_ID"] = get_input(
            "Bedrock Primary Model (inference profile or ARN)",
            "global.anthropic.claude-opus-4-8-v1"
        )
        env_updates["AIDLC_SONNET_MODEL_ID"] = get_input(
            "Bedrock Fast Model (used for risk checks / data gathering)",
            "global.anthropic.claude-sonnet-4-6"
        )
    elif choice == "2":
        env_updates["LLM_PROVIDER"] = "anthropic"
        env_updates["ANTHROPIC_API_KEY"] = get_input("Anthropic API Key")
        # Anthropic direct API: bare model ids (no us./global./bedrock prefix).
        env_updates["AIDLC_MODEL_ID"] = get_input(
            "Anthropic Primary Model",
            "claude-sonnet-4-6-20250929"
        )
        env_updates["AIDLC_SONNET_MODEL_ID"] = get_input(
            "Anthropic Fast Model (used for risk checks / data gathering)",
            "claude-haiku-4-6-20250929"
        )
    elif choice == "3":
        env_updates["LLM_PROVIDER"] = "openai"
        env_updates["OPENAI_API_KEY"] = get_input("OpenAI API Key")
        env_updates["AIDLC_MODEL_ID"] = get_input(
            "OpenAI Primary Model",
            "gpt-4o"
        )
        env_updates["AIDLC_SONNET_MODEL_ID"] = get_input(
            "OpenAI Fast Model (used for risk checks / data gathering)",
            "gpt-4o-mini"
        )
    elif choice == "4":
        env_updates["LLM_PROVIDER"] = "azure"
        print("\n  Endpoint must be the OpenAI-compatible v1 surface of your resource:")
        print("    https://<resource>.services.ai.azure.com/openai/v1")
        print("  (NOT the /api/projects/<project> Foundry URL — that fails with")
        print("   'API version not supported'.)\n")
        env_updates["AZURE_OPENAI_ENDPOINT"] = get_input(
            "Azure OpenAI Endpoint (…/openai/v1)"
        )
        env_updates["AZURE_OPENAI_API_KEY"] = get_input("Azure OpenAI API Key")
        # Model ids here are the DEPLOYMENT names you created in Azure AI Foundry.
        env_updates["AIDLC_MODEL_ID"] = get_input(
            "Azure Primary Deployment (reasoning)",
            "DeepSeek-V4-Pro"
        )
        env_updates["AIDLC_SONNET_MODEL_ID"] = get_input(
            "Azure Fast Deployment (used for risk checks / data gathering)",
            "gpt-5-mini"
        )

    # 3. Financial Data
    header("Section 3: Financial Data Access")
    print("To analyze real-time markets, we need data API keys.")
    print("All are optional — the system falls back to yfinance / web sources.\n")
    av_key = get_input("AlphaVantage API Key (for real-time quotes)", "")
    fmp_key = get_input("FMP API Key (for fundamentals/ratios)", "")

    if av_key: env_updates["ALPHA_VANTAGE_API_KEY"] = av_key
    if fmp_key: env_updates["FMP_API_KEY"] = fmp_key

    # Optional enhancement keys — only prompt if the user wants them, to keep
    # the common path short. These match the "Optional enhancements" and
    # "Search & News" blocks in .env.example.
    more_data = get_input("Add more optional data sources (macro/sentiment/options/search)? (y/N)", "n")
    if more_data.lower() == 'y':
        tavily_key  = get_input("Tavily API Key (AI-optimized news/web search)", "")
        fred_key    = get_input("FRED API Key (macro indicators: GDP, inflation, rates)", "")
        finnhub_key = get_input("Finnhub API Key (sentiment & analyst ratings)", "")
        polygon_key = get_input("Polygon API Key (options chains & advanced technicals)", "")
        if tavily_key:  env_updates["TAVILY_API_KEY"]  = tavily_key
        if fred_key:    env_updates["FRED_API_KEY"]    = fred_key
        if finnhub_key: env_updates["FINNHUB_API_KEY"] = finnhub_key
        if polygon_key: env_updates["POLYGON_API_KEY"] = polygon_key

    # 3b. Regional preferences — drive how figures/currency are presented.
    # Non-secret, so these land in user_data/.env (defaults match .env.example).
    header("Section 3b: Regional Preferences")
    env_updates["BASE_CURRENCY"] = get_input(
        "Base currency for valuations (e.g. USD, CAD, EUR, GBP)", "CAD"
    )
    env_updates["REGIONAL_LOCALE"] = get_input(
        "Regional locale (e.g. English (United States), English (Canada))",
        "English (Canada)"
    )

    # 4. Brokerage (Optional)
    header("Section 4: Brokerage Connectivity")
    connect_broker = get_input("Do you want to connect a live brokerage? (y/N)", "n")
    if connect_broker.lower() == 'y':
        print("Choose provider:")
        print("  [1] Alpaca (US/Global - Best for Paper Trading)")
        print("  [2] Questrade (Canada)")
        b_choice = get_input("Select [1-2]", "1")
        if b_choice == "1":
            env_updates["ALPACA_API_KEY"] = get_input("Alpaca API Key")
            env_updates["ALPACA_SECRET_KEY"] = get_input("Alpaca Secret Key")
            env_updates["ALPACA_PAPER_MODE"] = "true"
        elif b_choice == "2":
            env_updates["QUESTRADE_ENABLED"] = "true"
            env_updates["QUESTRADE_REFRESH_TOKEN"] = get_input("Questrade Refresh Token")

    # 5. Save and Finish
    header("Section 5: Finalizing")
    save_env(env_updates)

    print(f"\n{GREEN}{BOLD}** Setup Complete! **{NC}")
    print(f"You can now launch CairnIQ and start researching with {name}'s private intelligence console.")
    print("-" * 60)
    print(f"Run: {CYAN}./CairnIQ.command{NC}")
    print("-" * 60 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
        sys.exit(1)
