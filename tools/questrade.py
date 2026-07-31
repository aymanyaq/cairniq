import os
import time

import portalocker
import requests
from dotenv import load_dotenv

from agent.utils import safe_print
from tools.broker_credentials import (
    broker_lock_path,
    get_broker_secret,
    get_broker_setting,
    refresh_profile_state,
    set_broker_secret,
    set_broker_setting,
)
from tools.exception_logger import log_exceptions

# Load environment variables from user_data/
env_path = os.path.join(os.path.dirname(__file__), "..", "user_data", ".env")
load_dotenv(env_path)
REQUEST_TIMEOUT_SECONDS = 15


@log_exceptions()
def _safe_float(raw_value, default=0.0):
    """Coerce a value to float without crashing on empty or malformed input."""
    if raw_value is None:
        return float(default)

    text = str(raw_value).strip()
    if not text:
        return float(default)

    try:
        return float(text)
    except (TypeError, ValueError):
        return float(default)

class QuestradeClient:
    """Manages a SINGLE Questrade API connection (one token)."""
    def __init__(self, token_suffix="", refresh_token_override=None):
        self.suffix = token_suffix
        # Credentials resolve for the active profile: the default profile reads
        # the legacy global tokens; a named profile reads its own isolated set.
        self.refresh_token = refresh_token_override or get_broker_secret(f"QUESTRADE_REFRESH_TOKEN{token_suffix}")
        self.access_token = get_broker_setting(f"QUESTRADE_ACCESS_TOKEN{token_suffix}") or None
        self.api_server = get_broker_setting(f"QUESTRADE_API_SERVER{token_suffix}") or None
        self.token_expiry = _safe_float(get_broker_setting(f"QUESTRADE_TOKEN_EXPIRY{token_suffix}", "0"), 0)
        self.account_owner = get_broker_setting("QUESTRADE_ACCOUNT_OWNER", "User") or "User"

    def _save_tokens(self, access_token, api_server, refresh_token, expires_in):
        """Persist refreshed tokens for the active profile. Expected to be called
        from within a Lock by authenticate()."""
        s = self.suffix
        expires_at = time.time() + float(expires_in) - 60  # 1 minute buffer

        # We no longer take a lock here because this is always called from authenticate()
        # which already holds the lock. Double-locking causes deadlocks on some systems.
        # The refresh token is a secret (keychain); the rest is session state.
        set_broker_setting(f"QUESTRADE_ACCESS_TOKEN{s}", access_token)
        set_broker_setting(f"QUESTRADE_API_SERVER{s}", api_server)
        set_broker_secret(f"QUESTRADE_REFRESH_TOKEN{s}", refresh_token)
        set_broker_setting(f"QUESTRADE_TOKEN_EXPIRY{s}", str(expires_at))

        # Update local state too
        self.access_token = access_token
        self.api_server = api_server
        self.refresh_token = refresh_token
        self.token_expiry = expires_at

    def _format_request_error(self, action, exc):
        """Explain network failures in a way that separates token issues from API reachability."""
        if isinstance(exc, requests.exceptions.ConnectTimeout):
            api_host = self.api_server or "Questrade API host"
            return (
                f"Questrade {action}: API host {api_host} timed out. "
                "Token refresh may have succeeded, but the Questrade API server did not respond."
            )
        if isinstance(exc, requests.exceptions.RequestException):
            return f"Questrade {action}: {exc}"
        return str(exc)


    def authenticate(self, force_refresh=False):
        """Refresh the access token from Questrade with locking to prevent double-refresh."""
        if not self.refresh_token or not str(self.refresh_token).strip():
            return {"error": "Questrade Auth: Missing refresh token. Add a valid Questrade refresh token in Settings."}

        lock_path = broker_lock_path()

        with portalocker.Lock(lock_path, timeout=5):
            # 1. Re-read persisted state inside the lock to see if another worker
            #    (for this same profile) already refreshed the token.
            refresh_profile_state()
            s = self.suffix
            current_expiry = _safe_float(get_broker_setting(f"QUESTRADE_TOKEN_EXPIRY{s}", "0"), 0)

            if (not force_refresh) and time.time() < current_expiry:
                # Someone else already refreshed it! Use that.
                self.access_token = get_broker_setting(f"QUESTRADE_ACCESS_TOKEN{s}") or None
                self.api_server = get_broker_setting(f"QUESTRADE_API_SERVER{s}") or None
                self.refresh_token = get_broker_secret(f"QUESTRADE_REFRESH_TOKEN{s}") or None
                self.token_expiry = current_expiry
                # Only log once per instance to avoid console spam
                if not getattr(self, '_lock_sync_logged', False):
                    safe_print(f"ℹ️ Questrade Auth: Using fresh token already updated by another worker for {self.account_owner or 'Account' + self.suffix}")
                    self._lock_sync_logged = True
                return {"success": True, "source": "lock_sync"}

            # 2. If still expired, proceed with live refresh
            safe_print(f"🔄 Questrade Auth: Refreshing token for {self.account_owner or 'Account' + self.suffix}...")

            # OAuth2 standard requires sending credentials in the POST body to avoid 400 Bad Request
            url = "https://login.questrade.com/oauth2/token"
            data = {
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token.strip()
            }

            try:
                response = requests.post(url, data=data, timeout=REQUEST_TIMEOUT_SECONDS)
                if response.status_code == 200:
                    data = response.json()
                    self._save_tokens(
                        data['access_token'],
                        data['api_server'],
                        data['refresh_token'],
                        data['expires_in']
                    )
                    safe_print(f"✅ Questrade Auth: SUCCESS for {self.account_owner or 'Account' + self.suffix}")
                    return {"success": True, "source": "network"}
                else:
                    extra_hint = ""
                    if response.status_code == 400:
                        extra_hint = " Likely causes: expired/revoked refresh token, or a previously burned one-time token."
                    error_msg = f"Questrade Auth: FAILED for {self.account_owner or 'Account' + self.suffix} (Status {response.status_code}: {response.text}){extra_hint}"
                    safe_print(f"❌ {error_msg}")
                    # If it's a 400 but the body says 'invalid_grant', the token is dead.
                    # If it's a 400 and No Body, it might be a format error (Query vs Body).
                    return {"error": error_msg}
            except Exception as e:
                error_msg = f"Questrade Connection: ERROR for {self.account_owner or 'Account' + self.suffix} ({e})"
                safe_print(f"🚨 {error_msg}")
                return {"error": error_msg}

    def get_accounts(self):
        """Get list of accounts (with retry on 401)."""
        auth = self.authenticate()
        if "error" in auth: return auth

        url = f"{self.api_server}v1/accounts"
        headers = {"Authorization": f"Bearer {self.access_token}"}

        def _request():
            return requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)

        try:
            response = _request()
            # Check for invalid token (Questrade uses custom errors or 401)
            if response.status_code == 401 or (response.status_code != 200 and "invalid" in response.text.lower()):
                # Retry once with forced refresh
                safe_print(f"🔄 Token expired for {self.account_owner}, refreshing...")
                auth = self.authenticate(force_refresh=True)
                if "error" in auth: return {"error": auth["error"]}
                # Update headers with NEW token
                headers["Authorization"] = f"Bearer {self.access_token}"
                response = _request()

            if response.status_code == 200:
                return response.json()['accounts']
            else:
                return {"error": f"Failed to fetch accounts: {response.text}"}
        except Exception as e:
            return {"error": self._format_request_error("Accounts", e)}

    def get_positions(self, account_id):
        """Get positions regarding account (with retry)."""
        auth = self.authenticate()
        if "error" in auth: return auth

        url = f"{self.api_server}v1/accounts/{account_id}/positions"
        headers = {"Authorization": f"Bearer {self.access_token}"}

        def _request():
            return requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)

        try:
            response = _request()
            if response.status_code == 401 or (response.status_code != 200 and "invalid" in response.text.lower()):
                auth = self.authenticate(force_refresh=True)
                if "error" in auth: return {"error": auth["error"]}
                headers["Authorization"] = f"Bearer {self.access_token}"
                response = _request()

            if response.status_code == 200:
                return response.json()['positions']
            else:
                return {"error": f"Failed to fetch positions: {response.text}"}
        except Exception as e:
            return {"error": self._format_request_error("Positions", e)}

    def get_balances(self, account_id):
        """Get balances for a specific account."""
        auth = self.authenticate()
        if "error" in auth: return auth

        url = f"{self.api_server}v1/accounts/{account_id}/balances"
        headers = {"Authorization": f"Bearer {self.access_token}"}

        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Failed to fetch balances: {response.text}"}
        except Exception as e:
            return {"error": self._format_request_error("Balances", e)}

    def get_symbols(self, symbol_ids):
        """Fetch symbol details (currency, etc.) for a list of IDs."""
        auth = self.authenticate()
        if "error" in auth: return auth

        # Helper to chunk list into batches of 100 (API limit)
        def chunker(seq, size):
            return (seq[pos:pos + size] for pos in range(0, len(seq), size))

        all_symbols = []
        headers = {"Authorization": f"Bearer {self.access_token}"}

        try:
            for batch in chunker(symbol_ids, 100):
                ids_str = ",".join(map(str, batch))
                url = f"{self.api_server}v1/symbols?ids={ids_str}"
                response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)

                if response.status_code == 200:
                    data = response.json()
                    if 'symbols' in data:
                        all_symbols.extend(data['symbols'])
                else:
                    safe_print(f"⚠️ Failed to fetch symbols batch: {response.text}")

            return all_symbols
        except Exception as e:
            return {"error": self._format_request_error("Symbols", e)}

    def get_aggregated_holdings(self):
        """Fetch holdings AND CASH for this client."""
        accounts = self.get_accounts()
        if isinstance(accounts, dict) and "error" in accounts:
            return [{"error": f"{self.account_owner}: {accounts['error']}"}]

        holdings = []

        # 1. Collect all positions FIRST to do a batch symbol lookup
        all_positions = []
        symbol_ids_to_fetch = set()

        for acc in accounts:
            acc_type = acc['type']
            acc_id = acc['number']

            # Get Positions
            positions = self.get_positions(acc_id)
            if isinstance(positions, list):
                for p in positions:
                    if p['openQuantity'] > 0:
                        # Store context for later processing
                        p['_account_type'] = acc_type
                        p['_account_id'] = acc_id
                        all_positions.append(p)
                        if p.get('symbolId'):
                            symbol_ids_to_fetch.add(p['symbolId'])

            # Get Cash (process immediately as before)
            balances = self.get_balances(acc_id)
            if isinstance(balances, dict) and 'combinedBalances' in balances:
                 for bal in balances.get('perCurrencyBalances', []):
                     cash = bal.get('cash', 0)
                     if cash > 1.0: # Filter small dust
                         holdings.append({
                            "symbol": "CASH",
                            "shares": cash, # For cash, shares = amount
                            "purchase_price": 1.0,
                            "current_price": 1.0,
                            "market_value": cash,
                            "currency": bal.get('currency', 'CAD'),
                             "account": f"{acc_type} Questrade",
                            "account_id": f"***{str(acc_id)[-4:]}" if len(str(acc_id)) > 4 else "***",
                            "return_pct": 0.0
                         })

        # 2. Batch Fetch Symbol Details (Currency)
        symbol_map = {}
        if symbol_ids_to_fetch:
            symbols_data = self.get_symbols(list(symbol_ids_to_fetch))
            if isinstance(symbols_data, list):
                for s in symbols_data:
                    symbol_map[s['symbolId']] = s.get('currency', 'CAD')

        # 3. Process Positions with Correct Currency
        for p in all_positions:
            sid = p.get('symbolId')
            # Look up currency from our batch fetch, default to CAD only if missing
            # Note: "USD" vs "CAD" is standard in Questrade API
            curr = symbol_map.get(sid)

            # Fallback Heuristic if API failed to provide currency
            if not curr:
                # TSX/TSX-V symbols usually end in .TO or .VN
                if str(p['symbol']).endswith('.TO') or str(p['symbol']).endswith('.VN'):
                    curr = 'CAD'
                else:
                    # Most other symbols (NYSE, NASDAQ) are USD
                    curr = 'USD'

            # Verify if it's a USD account type but missing symbol info?
            # Trust the symbol metadata first.

            holdings.append({
                "symbol": p['symbol'],
                "shares": p['openQuantity'],
                "purchase_price": p['averageEntryPrice'],
                "current_price": p['currentPrice'],
                "market_value": p['currentMarketValue'],
                "currency": curr,
                "account": f"{p['_account_type']} Questrade",
                "account_id": f"***{str(p['_account_id'])[-4:]}" if len(str(p['_account_id'])) > 4 else "***",
                "return_pct": ((p['currentPrice'] - p['averageEntryPrice']) / p['averageEntryPrice'] * 100) if p['averageEntryPrice'] else 0
            })

        return holdings


class QuestradeAPI:
    """Wrapper to handle MULTIPLE Questrade clients/tokens transparently."""
    def __init__(self, refresh_token=None):
        self.clients = []
        self.enabled = get_broker_setting("QUESTRADE_ENABLED", "false").lower() == "true"

        if not self.enabled and not refresh_token:
            self.refresh_token = None
            self.access_token = None
            return

        # If a specific refresh_token is provided, use only that
        if refresh_token:
            self.clients.append(QuestradeClient(token_suffix="", refresh_token_override=refresh_token))
            self.refresh_token = refresh_token
            self.access_token = True  # Dummy for check
            return

        # 1. Main Account (No suffix) — resolved for the active profile.
        if get_broker_secret("QUESTRADE_REFRESH_TOKEN"):
            self.clients.append(QuestradeClient(token_suffix=""))

        self.refresh_token = True if self.clients else None # Backwards compatibility flag
        self.access_token = True # Dummy for check

    def get_all_holdings(self):
        """Aggregate holdings from ALL configured Questrade clients."""
        # "Switched off" and "asked and failed" are different answers and the
        # portfolio engine treats them differently — an error there suppresses the
        # Last-Known-Good snapshot and raises a sync alarm. Reporting a disabled
        # integration as an error made both of those permanent for anyone who
        # simply never linked Questrade, which is the default state.
        if not self.enabled:
            return {"holdings": [], "errors": [],
                    "notices": ["Questrade integration is disabled in Settings."]}

        if not self.clients:
            return {"holdings": [], "errors": [],
                    "notices": ["No Questrade tokens configured."]}

        all_data = []
        errors = []

        seen_account_ids = set()

        for client in self.clients:
            result = client.get_aggregated_holdings()
            # Check for list of positions vs error dict
            if isinstance(result, list):
                # Check for list of positions vs error dict
                if result and isinstance(result[0], dict) and "error" in result[0]:
                    errors.append(result[0]["error"])
                else:
                    # DEDUPLICATION:
                    # Filter out items belonging to accounts we've already seen
                    unique_items = []
                    for item in result:
                        acc_id = item.get("account_id")
                        if acc_id and acc_id in seen_account_ids:
                            continue # Skip duplicate account
                        unique_items.append(item)

                    # Update seen list with NEW accounts found
                    for item in unique_items:
                        if item.get("account_id"):
                            seen_account_ids.add(item.get("account_id"))

                    all_data.extend(unique_items)
            elif isinstance(result, dict) and "error" in result:
                errors.append(result["error"])

        return {
            "holdings": all_data,
            "errors": errors,
            "notices": []
        }

if __name__ == "__main__":
    qt = QuestradeAPI()
    print(f"Initialized {len(qt.clients)} clients.")

    print("\nFetching aggregated positions...")
    holdings = qt.get_all_holdings()
    if isinstance(holdings, list):
        print(f"Found {len(holdings)} positions total.")
        for h in holdings:
            print(f"{h.get('symbol', 'Unknown')} in {h.get('account', 'Unknown')}: {h.get('shares', 0)} shares ({h.get('currency', 'Unknown')})")
    else:
        print(f"Error: {holdings}")
