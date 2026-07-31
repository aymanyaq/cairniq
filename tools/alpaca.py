import os

import requests
from dotenv import load_dotenv

from tools.exception_logger import log_exceptions

# Load environment variables from user_data/
env_path = os.path.join(os.path.dirname(__file__), "..", "user_data", ".env")
load_dotenv(env_path)


def _safe_float(value, default: float = 0.0) -> float:
    """Coerce Alpaca string/number fields to float without crashing."""
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


class AlpacaAPI:
    """Manages connection to Alpaca Markets for portfolio data."""

    def __init__(self):
        # Resolved for the active profile: the default profile reads the legacy
        # global keys; a named profile reads its own isolated credentials.
        from tools.broker_credentials import get_broker_secret, get_broker_setting

        self.api_key = get_broker_secret("ALPACA_API_KEY")
        self.secret_key = get_broker_secret("ALPACA_SECRET_KEY")
        self.paper_mode = get_broker_setting("ALPACA_PAPER_MODE", "true").lower() == "true"

        if self.paper_mode:
            self.base_url = "https://paper-api.alpaca.markets/v2"
        else:
            self.base_url = "https://api.alpaca.markets/v2"

        self.headers = {
            "APCA-API-KEY-ID": self.api_key or "",
            "APCA-API-SECRET-KEY": self.secret_key or ""
        }

    def is_configured(self):
        return bool(self.api_key and self.secret_key)

    @log_exceptions()
    def _get(self, path: str) -> dict | list:
        """Single-request helper with consistent error envelopes and short retries
        on transient 5xx responses."""
        if not self.is_configured():
            return {"error": "Alpaca API keys not configured."}

        url = f"{self.base_url}{path}"
        last_error = None
        for attempt in range(2):  # 1 retry on transient failure
            try:
                response = requests.get(url, headers=self.headers, timeout=15)
            except requests.exceptions.Timeout:
                last_error = f"Alpaca timeout on {path} (attempt {attempt + 1}/2)"
                continue
            except requests.exceptions.RequestException as e:
                last_error = f"Alpaca network error on {path}: {e}"
                continue

            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError as e:
                    return {"error": f"Alpaca returned non-JSON response: {e}"}
            if 500 <= response.status_code < 600 and attempt == 0:
                last_error = f"Alpaca {response.status_code} on {path}: {response.text[:200]}"
                continue
            # Authoritative 4xx — surface key reasons clearly
            if response.status_code in (401, 403):
                mode = "paper" if self.paper_mode else "live"
                return {"error": (
                    f"Alpaca {response.status_code} on {path}: credentials rejected. "
                    f"Check ALPACA_API_KEY/SECRET match the {mode} environment "
                    f"(ALPACA_PAPER_MODE={'true' if self.paper_mode else 'false'})."
                )}
            return {"error": f"Alpaca API Error {response.status_code} on {path}: {response.text[:200]}"}

        return {"error": last_error or f"Alpaca unknown error on {path}"}

    def get_account(self):
        """Fetch general account information."""
        return self._get("/account")

    def get_positions(self):
        """Fetch open positions."""
        return self._get("/positions")

    def get_aggregated_holdings(self):
        """Fetch holdings and cash, formatted for the terminal's portfolio engine."""
        # Not configured is a settings state, not a failed sync — see the same
        # split in QuestradeAPI.get_all_holdings.
        if not self.is_configured():
            return {"holdings": [], "errors": [], "notices": ["Alpaca not configured"]}

        account = self.get_account()
        if isinstance(account, dict) and "error" in account:
            return {"holdings": [], "errors": [account["error"]]}
        if not isinstance(account, dict):
            return {"holdings": [], "errors": [f"Alpaca /account returned unexpected payload type: {type(account).__name__}"]}

        positions = self.get_positions()
        if isinstance(positions, dict) and "error" in positions:
            # Cash may still be valid even if positions failed — surface partial data.
            positions_list: list = []
            position_error: str | None = positions["error"]
        elif isinstance(positions, list):
            positions_list = positions
            position_error = None
        else:
            positions_list = []
            position_error = f"Alpaca /positions returned unexpected payload type: {type(positions).__name__}"

        holdings = []
        errors: list[str] = []
        if position_error:
            errors.append(position_error)

        acc_id = account.get("account_number") or account.get("id") or "ALPACA"
        acc_id_str = str(acc_id)
        masked_acc_id = f"***{acc_id_str[-4:]}" if len(acc_id_str) >= 4 else "***"
        account_label = "Alpaca " + ("Paper" if self.paper_mode else "Live")

        # 1. Add Cash
        cash = _safe_float(account.get("cash"))
        if cash > 0.01:
            holdings.append({
                "symbol": "CASH",
                "shares": cash,
                "purchase_price": 1.0,
                "current_price": 1.0,
                "market_value": cash,
                "currency": "USD",
                "account": account_label,
                "account_id": masked_acc_id,
                "return_pct": 0.0
            })

        # 2. Add Positions — skip rather than crash on malformed entries
        for idx, p in enumerate(positions_list):
            if not isinstance(p, dict):
                errors.append(f"Alpaca position #{idx} skipped: unexpected payload {type(p).__name__}")
                continue
            symbol = p.get("symbol")
            if not symbol:
                errors.append(f"Alpaca position #{idx} skipped: missing symbol")
                continue
            holdings.append({
                "symbol": symbol,
                "shares": _safe_float(p.get("qty")),
                "purchase_price": _safe_float(p.get("avg_entry_price")),
                "current_price": _safe_float(p.get("current_price")),
                "market_value": _safe_float(p.get("market_value")),
                "currency": "USD",
                "account": account_label,
                "account_id": masked_acc_id,
                "return_pct": _safe_float(p.get("unrealized_plpc")) * 100
            })

        return {
            "holdings": holdings,
            "errors": errors
        }

if __name__ == "__main__":
    alpaca = AlpacaAPI()
    print("Testing Alpaca Integration...")
    result = alpaca.get_aggregated_holdings()
    print(f"Found {len(result['holdings'])} items.")
    for h in result['holdings']:
        print(f"- {h['symbol']}: {h['shares']} @ {h['current_price']} ({h['market_value']} USD)")
