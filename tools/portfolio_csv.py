"""
Simple CSV Portfolio Reader
Reads your portfolio from a local CSV file - no cloud services needed!

Expected CSV format:
symbol,shares,purchase_price,account,currency
AAPL,100,150.00,Brokerage IRA,USD
NVDA,50,200.00,QuestTrade,USD
VTI,200,180.00,QuestTrade RSP,CAD
"""
import concurrent.futures
import csv
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

import yfinance as yf

from agent.utils import safe_print

# --- CACHING INFRASTRUCTURE ---
from tools.daily_cache import get_cached, set_cached
from tools.exception_logger import log_exceptions
from tools.json_store import write_json_atomic
from tools.user_profile import get_active_profile, get_data_path, is_demo_mode

# Global lock to prevent "Thundering Herd" (multiple parallel tool results at once)
_PORTFOLIO_LOCK = threading.Lock()

# How long a computed summary stays servable. ONE constant on purpose: the fast
# path and the post-lock re-check below used to disagree (300s vs 900s), and the
# stricter number sat on the path that always runs first, so nobody was ever
# served the 5-to-15-minute-old file the looser gate was there to allow — a lone
# visitor recomputed at 300s every time. Recomputing costs a full broker sync
# plus a quote per holding (~20s+ measured), which is the whole reason the cache
# exists, and a buy-and-hold book does not change meaningfully in 15 minutes.
SUMMARY_TTL_SECONDS = 900


class PortfolioPayload(list):
    """The holdings list, with the broker-sync metadata carried beside it.

    load_portfolio() used to append ``{"_sync_errors": [...]}`` as an extra entry
    to the list it returned, so a 29-position portfolio came back with 30 entries
    and every caller's count was wrong by exactly one. Wrong by one *whenever a
    broker had something to say* — and one always did, because a broker that is
    merely switched off reported that as an error, on every single load. A count
    that is wrong by a constant reads as stable, which is why this survived: the
    callers that iterate positions all happen to skip an entry with no symbol, so
    nothing ever crashed and nothing ever looked odd.

    Iterating this IS iterating positions. The metadata lives on attributes:

      sync_errors          a broker that was asked and failed — real, actionable,
                           and the trigger for the Last-Known-Good fallback.
      integration_notices  a broker that was never asked (disabled, no tokens).
                           Not a failure; it must not suppress the LKG snapshot
                           or raise an alarm, or the alarm means nothing.

    Caveat: this is a list subclass, so the attributes do not survive slicing,
    ``sorted()`` or ``list()``. Read them off the value load_portfolio() returned,
    or use split_portfolio_payload().
    """

    __slots__ = ("sync_errors", "integration_notices")

    def __init__(self, positions=(), sync_errors=None, integration_notices=None):
        super().__init__(positions)
        self.sync_errors: list[str] = list(sync_errors or [])
        self.integration_notices: list[str] = list(integration_notices or [])


def split_portfolio_payload(payload: Any) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """(positions, sync_errors, integration_notices) from whatever was handed back.

    Accepts the PortfolioPayload load_portfolio() returns AND the older plain
    list-with-an-inline-``_sync_errors``-dict shape, so a caller or test that
    hand-builds a holdings list still resolves its errors rather than silently
    reporting none. Anything without a "symbol" key is metadata, not a position.
    """
    if isinstance(payload, PortfolioPayload):
        return list(payload), list(payload.sync_errors), list(payload.integration_notices)

    positions: list[dict[str, Any]] = []
    sync_errors: list[str] = []
    notices: list[str] = []
    for item in payload or []:
        if not isinstance(item, dict):
            continue
        if "_sync_errors" in item:
            sync_errors.extend(item["_sync_errors"] or [])
            notices.extend(item.get("_integration_notices") or [])
            continue
        positions.append(item)
    return positions, sync_errors, notices


def _coerce_number(value: Any, default: float = 0.0) -> float:
    """Parse formatted currency/percent strings used in portfolio summaries."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r"[^0-9.\-]", "", str(value))
    if cleaned in {"", "-", "."}:
        return default
    try:
        return float(cleaned)
    except ValueError:
        return default

@log_exceptions()
def _lkg_path() -> str:
    """Path to the persistent 'Last Known Good' portfolio snapshot."""
    from tools.daily_cache import CACHE_DIR
    os.makedirs(CACHE_DIR, exist_ok=True)
    profile = get_active_profile()
    is_demo = is_demo_mode()
    file_name = f"{profile}_demo_portfolio_lkg.json" if is_demo else f"{profile}_portfolio_lkg.json"
    return os.path.join(CACHE_DIR, file_name)

@log_exceptions()
def save_lkg(summary: dict[str, Any]) -> None:
    """Save a successful portfolio snapshot to disk."""
    try:
        path = _lkg_path()
        # Add timestamp to the data
        data = summary.copy()
        data["_lkg_timestamp"] = datetime.now().isoformat()
        write_json_atomic(path, data, default=str)
        safe_print(f"📊 Portfolio LKG: Saved snapshot to {os.path.basename(path)}")
    except Exception as e:
        safe_print(f"⚠️ Failed to save LKG: {e}")

@log_exceptions()
def load_lkg() -> dict[str, Any] | None:
    """Load the most recent successful snapshot from disk."""
    try:
        path = _lkg_path()
        if os.path.exists(path):
            with open(path) as f:
                import json
                return json.load(f)
    except Exception as e:
        safe_print(f"⚠️ Failed to load LKG: {e}")
    return None


@log_exceptions()
def load_portfolio(file_path: str = None) -> PortfolioPayload:
    """
    Load portfolio from a CSV file AND live Questrade API if configured.
    Prioritizes live data for Questrade accounts, removing duplicates from CSV.

    Returns a PortfolioPayload: a list of positions and nothing else, with the
    broker-sync metadata on ``.sync_errors`` / ``.integration_notices``. On an
    unreadable portfolio file it returns ``{"error": ...}`` instead — callers
    that iterate must check for that dict first, or they will iterate its keys.
    """
    is_demo = is_demo_mode()

    if file_path is None:
        file_path = get_data_path("my_portfolio.csv")
        if is_demo and not os.path.exists(file_path):
            legacy_demo_path = get_data_path("demo_portfolio.csv")
            if os.path.exists(legacy_demo_path):
                file_path = legacy_demo_path

    holdings = []

    # 1. Load CSV (Base Layer)
    if os.path.exists(file_path):
        try:
            with open(file_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Skip comment lines
                    symbol = (row.get("symbol") or row.get("Symbol") or "").strip()
                    if symbol.startswith("#") or not symbol:
                        continue

                    # Get return percentage (for pension/cash accounts)
                    return_pct_str = (row.get("return_pct") or row.get("Return Pct") or "").strip()
                    return_pct = float(return_pct_str) if return_pct_str else None

                    # Flexible extraction
                    shares_str = row.get("shares") or row.get("Shares") or "0"
                    price_str = row.get("purchase_price") or row.get("Purchase Price") or "0"
                    current_price_str = row.get("current_price") or row.get("Current Price") or ""
                    market_value_str = row.get("market_value") or row.get("Market Value") or ""
                    account_str = row.get("account") or row.get("Account") or "Unknown"

                    # Identify if it's a Private Asset (not a tradable ticker)
                    asset_type_str = (row.get("Asset Type") or row.get("asset_type") or "Public").strip().lower()
                    is_private_asset = (asset_type_str == "private")

                    currency_str = row.get("currency") or row.get("Currency")
                    if not currency_str:
                        currency_str = "CAD" if (is_private_asset or ".TO" in symbol) else "USD"

                    try:
                        holding = {
                            "symbol": symbol.upper(),
                            "shares": float(shares_str),
                            "purchase_price": float(price_str),
                            "account": account_str,
                            "currency": currency_str,
                            "return_pct": return_pct,
                            "source": "Manual",
                            "is_private_asset": is_private_asset
                        }
                        if current_price_str:
                            holding["current_price"] = float(current_price_str)
                        if market_value_str:
                            holding["market_value"] = float(market_value_str)
                        holdings.append(holding)
                    except ValueError:
                        continue # Skip bad rows
        except Exception as e:
            return {"error": f"Failed to read portfolio file: {e}"}

    # 2. Add Questrade Holdings (Automatic multi-source)
    sync_errors = []
    # A broker that was never asked — switched off, or no tokens saved — is not a
    # failed sync. Filing it under sync_errors made every load look degraded, which
    # is worse than useless: it suppressed the LKG snapshot that exists to bridge a
    # REAL failure, so by the time one happened there was no snapshot to fall back to.
    integration_notices = []
    qt_holdings = []
    if not is_demo:
        try:
            from tools.questrade import QuestradeAPI
            qt = QuestradeAPI()
            qt_result = qt.get_all_holdings()

            qt_holdings_list = qt_result.get("holdings", [])
            sync_errors = list(qt_result.get("errors") or [])
            integration_notices.extend(qt_result.get("notices") or [])

            if qt_holdings_list:
                # Mark each live holding as "API" source
                for h in qt_holdings_list:
                    h["source"] = "API"
                    qt_holdings.append(h)

                # Filter out ONLY the Questrade accounts that we successfully fetched live.
                # If a token failed (sync_errors), we KEEP the old CSV data for those accounts
                # as a stale fallback, rather than showing $0.
                synced_accounts = set(h["account"] for h in qt_holdings)

                non_qt_holdings = []
                for h in holdings:
                    acc_name = h.get("account", "")
                    # Logic: If it's a Questrade account that we just synced, remove legacy CSV row.
                    # If it's NOT in the sync list (maybe its token failed or it's a manual account), keep it.
                    if "Questrade" in acc_name and acc_name in synced_accounts:
                        continue
                    non_qt_holdings.append(h)

                holdings = non_qt_holdings + qt_holdings
            elif qt.enabled and qt.clients and not sync_errors:
                # Questrade is configured but returned zero holdings with no errors.
                # This is likely a transient auth/API failure that was silently swallowed.
                # Flag it so the LKG fallback can bridge the gap instead of caching $0.
                sync_errors.append("Questrade returned 0 holdings (possible token/API issue)")
                safe_print("⚠️ Questrade sync returned empty holdings — flagging for LKG fallback")

        except Exception as e:
            sync_errors.append(f"Questrade Global Error: {str(e)}")

    # 3. Add Alpaca Holdings
    alpaca_holdings = []
    if not is_demo:
        try:
            from tools.alpaca import AlpacaAPI
            alpaca = AlpacaAPI()
            if alpaca.is_configured():
                alpaca_result = alpaca.get_aggregated_holdings()
                alpaca_holdings_list = alpaca_result.get("holdings", [])
                sync_errors.extend(alpaca_result.get("errors") or [])
                integration_notices.extend(alpaca_result.get("notices") or [])

                if alpaca_holdings_list:
                    for h in alpaca_holdings_list:
                        h["source"] = "API"
                        alpaca_holdings.append(h)

                    synced_alpaca_accounts = set(h["account"] for h in alpaca_holdings)

                    # Filter CSV layer
                    filtered_holdings = []
                    for h in holdings:
                        acc_name = h.get("account", "")
                        if "Alpaca" in acc_name and acc_name in synced_alpaca_accounts:
                            continue
                        filtered_holdings.append(h)

                    holdings = filtered_holdings + alpaca_holdings
        except Exception as e:
            sync_errors.append(f"Alpaca Global Error: {str(e)}")

    # Metadata rides beside the positions, not inside them. See PortfolioPayload.
    return PortfolioPayload(holdings, sync_errors, integration_notices)


@log_exceptions()
def get_portfolio_summary(force: bool = False) -> dict[str, Any]:
    """
    Get a complete portfolio summary with current values and gains/losses.
    Uses a 5-minute TTL cache unless force=True.
    Thread-safe with a global lock to prevent redundant heavy computations.
    """
    is_demo = is_demo_mode()
    cache_key = "demo_portfolio_summary" if is_demo else "portfolio_summary"

    # 1. Fast Path: Check cache WITHOUT lock for performance
    if not force:
        cached = get_cached(cache_key, ttl_seconds=SUMMARY_TTL_SECONDS)
        if cached:
            safe_print("💎 Portfolio Summary: Cache Hit")
            return cached

        # 2. Slow Path: Acquire lock to compute (or wait for first thread to compute)
    with _PORTFOLIO_LOCK:
        # Re-check cache inside lock (Double-Checked Locking pattern)
        if not force:
            cached = get_cached(cache_key, ttl_seconds=SUMMARY_TTL_SECONDS)
            if cached:
                safe_print("💎 Portfolio Summary: Cache Hit (Post-Lock)")
                return cached

        safe_print("📊 Portfolio Summary: Cache Miss, computing live...")
        try:
            # We no longer need the inner ThreadPoolExecutor because the outer graph
            # is already threaded and we have a lock.
            result = _compute_portfolio_summary()

            # --- LKG LOGIC ---
            # If the live computation succeeded with no sync errors, update the LKG
            if result and not result.get("error") and not result.get("sync_errors"):
                save_lkg(result)

                # --- KNOWLEDGE GRAPH SYNC ---
                # Sync to graph maximum once every 4 hours (14400 seconds) to prevent I/O spam
                if not get_cached("graph_sync_throttler", ttl_seconds=14400):
                    import threading
                    threading.Thread(target=sync_portfolio_to_graph, args=(result["holdings"],), daemon=True).start()
                    set_cached("graph_sync_throttler", True)
            elif result.get("sync_errors"):
                # If we have sync errors, try to fallback to LKG to bridge the gap
                lkg = load_lkg()
                if lkg:
                    safe_print("🔄 Portfolio Sync: Using Last Known Good (LKG) fallback due to sync errors.")
                    lkg["is_stale"] = True
                    lkg["sync_errors"] = result["sync_errors"]
                    lkg["lkg_total_cad"] = lkg.get("total_value_cad")
                    # Optionally merge - if LKG has 100k more than live, it's safer to use LKG.
                    result = lkg
                else:
                    safe_print("⚠️ Portfolio Sync: No LKG found to bridge sync errors.")

        except Exception as e:
            safe_print(f"🚨 Portfolio calculation FAILED: {e}")
            # Try to load LKG as a hard fallback if the entire computation crashed
            lkg = load_lkg()
            if lkg:
                safe_print("🔄 Portfolio Calculation: Hard Fallback to LKG.")
                result = lkg
                result["is_stale"] = True
                result["lkg_total_cad"] = lkg.get("total_value_cad")
            else:
                result = {"error": str(e)}

        if not result.get("error"):
            set_cached(cache_key, result)
        return result

def get_profile_base_currency() -> str:
    try:
        from tools.memory import get_profile_base_currency as _get_profile_base_currency
        return _get_profile_base_currency()
    except Exception:
        return os.environ.get("BASE_CURRENCY") or os.environ.get("CAIRNIQ_BASE_CURRENCY") or "USD"


def get_exchange_rate(from_currency: str, to_currency: str) -> float:
    from_currency = (from_currency or "USD").upper().strip()
    to_currency = (to_currency or "USD").upper().strip()
    if from_currency == to_currency:
        return 1.0

    # 1. First check USD/CAD special case for exact backward compatibility and test stability
    if {from_currency, to_currency} == {"USD", "CAD"}:
        usd_cad_rate = get_cached("usd_cad_rate", ttl_seconds=3600)
        if not (usd_cad_rate and isinstance(usd_cad_rate, (int, float))):
            usd_cad_rate = float(os.environ.get("USD_TO_CAD", "1.44"))
        if from_currency == "USD":
            return usd_cad_rate
        else:
            return 1.0 / usd_cad_rate

    # 2. Try loading from cache
    cache_key = f"rate_{from_currency}_{to_currency}".lower()
    rate = get_cached(cache_key, ttl_seconds=3600)
    if rate and isinstance(rate, (int, float)):
        return rate

    # 3. Fetch using yfinance
    ticker_name = f"{from_currency}{to_currency}=X".upper()
    # Try direct and inverse ticker mappings
    for symbol in [ticker_name, f"{to_currency}{from_currency}=X".upper()]:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d", timeout=20)
            if not hist.empty:
                val = float(hist["Close"].iloc[-1])
                if symbol != ticker_name:
                    val = 1.0 / val
                set_cached(cache_key, val)
                return val
        except Exception:
            pass

    # Default fallback
    return 1.0



@log_exceptions()
def _compute_portfolio_summary() -> dict[str, Any]:
    """Internal implementation of portfolio summary calculation."""
    all_holdings_data = load_portfolio()

    # The error shape is a dict, so it has to be caught BEFORE anything iterates
    # it — iterating a dict walks its keys, and every one of them is a string.
    if isinstance(all_holdings_data, dict) and "error" in all_holdings_data:
        return all_holdings_data

    holdings, sync_errors, integration_notices = split_portfolio_payload(all_holdings_data)

    if not holdings:
        return {"error": "No holdings found in portfolio file"}

    portfolio_data = []
    total_invested = 0
    total_current_value = 0
    accounts = {}

    # 1. Parallel Fetching for Market Data
    is_demo_seeded = is_demo_mode()
    symbols_to_fetch = []
    # Broker rows arrive pre-priced, so they skip the quote path entirely and would
    # never learn the day's move: the same ticker held manually shows a direction while
    # the synced copy shows nothing. Quote these too — for the move only, their price
    # still comes from the broker — so both copies agree.
    day_move_only_symbols = set()
    for holding in holdings:
        if not isinstance(holding, dict) or "symbol" not in holding:
            continue
        symbol = holding["symbol"]
        purchase_price = holding.get("purchase_price", 0)

        # Identify if it's a Private Asset (not a tradable ticker)
        is_manual_asset = holding.get("is_private_asset", False)

        # Decide if we need to fetch live data. A pinned CSV price is deliberately NOT a
        # reason to skip the quote: it is a fallback for instruments the market cannot
        # price, not an override. Skipping made a pinned price permanent — the row kept
        # reporting it long after the market moved, and never showed a day move.
        # The demo profile is the exception: its prices are seeded so the sample
        # portfolio is deterministic and works offline, so there a pin is the answer.
        if not (holding.get("market_value") is not None or
                purchase_price == 1.0 or
                is_manual_asset or
                symbol == "CASH" or
                (is_demo_seeded and holding.get("current_price"))):
            symbols_to_fetch.append(symbol)
        elif (holding.get("source") == "API" and not is_manual_asset
              and symbol != "CASH" and purchase_price != 1.0):
            day_move_only_symbols.add(symbol)

    # Quote the broker-only tickers alongside the rest. Their price_map entry is simply
    # never read — those rows take an earlier branch below — so this adds a direction
    # without letting a quote override a broker's own price. get_stock_data is cached,
    # so repeat computes inside the cache window cost nothing.
    fetch_list = list(dict.fromkeys(symbols_to_fetch + sorted(day_move_only_symbols)))

    price_map = {}
    # Day move for the symbols we quote live. Kept separate from price_map so a
    # provider that has a price but no usable previous close (Polygon serves the prior
    # session's close; a yfinance quote can fall back to it) simply has no entry here,
    # rather than a 0.00% that would read as "unchanged today".
    day_change_map = {}
    if fetch_list:
        from agent.utils import get_st_aware_func
        from tools.market_data import get_stock_data
        executor = ThreadPoolExecutor(max_workers=min(len(fetch_list), 5))
        try:
            future_to_symbol = {executor.submit(get_st_aware_func(get_stock_data), sym): sym for sym in fetch_list}
            for future in concurrent.futures.as_completed(future_to_symbol, timeout=180):
                symbol = future_to_symbol[future]
                try:
                    data = future.result(timeout=40) # Individual tool timeout
                    price_str = data.get("current_price", "0").replace("$", "").replace(",", "")
                    price_map[symbol] = float(price_str) if "N/A" not in price_str else None
                    day_change = data.get("day_change_pct")
                    if isinstance(day_change, (int, float)):
                        day_change_map[symbol] = float(day_change)
                except Exception:
                    price_map[symbol] = None
        except concurrent.futures.TimeoutError:
             safe_print("⚠️ Portfolio Sync: Price fetching timed out for some symbols. Using fallbacks.")
             # Mark timed-out symbols as None so downstream code uses purchase_price fallback
             for future, sym in future_to_symbol.items():
                 if sym not in price_map:
                     price_map[sym] = None

        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    # 1.5 Load LKG for price fallbacks if any symbols still have None price
    lkg_data = {}
    if any(p is None for p in price_map.values()) or not price_map:
        try:
            lkg_path = get_data_path(f"{get_active_profile()}_portfolio_lkg.json")
            if os.path.exists(lkg_path):
                with open(lkg_path) as f:
                    lkg_snapshot = json.load(f)
                    for h in lkg_snapshot.get("holdings", []):
                        if h.get("symbol") and h.get("current_price"):
                            p_str = str(h["current_price"]).replace("$", "").replace(",", "")
                            try:
                                lkg_data[h["symbol"]] = float(p_str)
                            except: pass
        except Exception: pass

    # 2. Main Processing Loop
    for holding in holdings:
        # Safety check for malformed entries
        if not isinstance(holding, dict) or "symbol" not in holding:
            continue

        symbol = holding["symbol"]
        shares = holding.get("shares", 0)
        purchase_price = holding.get("purchase_price", 0)
        account = holding.get("account", "Unknown")
        currency = holding.get("currency", "USD")
        stored_return = holding.get("return_pct")
        # A stated total — the holder typed the value straight off a statement,
        # because the instrument carries no quotable price and they hold no entry
        # price for it. A group pension states units, a return and a total, and
        # nothing else; that is the whole of what its holder can enter. Where it is
        # present it IS the mark, and cost basis is derived backwards out of it —
        # the reverse of every other row in this loop.
        # Restricted to manual rows on purpose: market_value is overloaded. A broker
        # sets it as a "came pre-valued, don't fetch" marker on a row that DOES have a
        # real quote and a real day move, so reading a broker's copy as a hand-typed
        # statement total re-derives its price and strips its direction.
        stated_total = holding.get("market_value")
        is_manually_valued = (stated_total is not None and shares > 0
                              and holding.get("source") != "API")

        # Identify if it's a Private Asset (not a tradable ticker)
        is_manual_asset = holding.get("is_private_asset", False)
        # A unit-priced row (purchase_price == 1.0) is a synthetic holding — pension
        # units, a cash balance — not a $1.00 equity. The price branch below already
        # pins such a row to 1.0, so it must be classified the same way here or its
        # typed return_pct is silently dropped and the row reports a flat 0%.
        # A row carrying its own total is synthetic for the same reason — nothing
        # quoted it — so it must not be returned as though a market had.
        is_cash_or_pension = (symbol == "CASH" or is_manual_asset
                              or purchase_price == 1.0 or is_manually_valued)

        # Stays None unless the price below came from a live quote this run. A broker
        # row, a manual CSV price, a Last-Known-Good snapshot and a purchase-price
        # fallback all have no "today" to move against.
        day_change_pct = None

        # A stated total is authoritative, so the unit mark is implied by it rather
        # than the other way round. Kept at full precision: the 2dp display string is
        # a separate field, and valuing off a rounded unit price loses real money on a
        # large unit count (the demo pension lost $295 to exactly that).
        if is_manually_valued:
             current_price = stated_total / shares
        # A broker's market_value is a "came pre-valued, don't fetch" marker rather than
        # an authoritative figure: every producer (Questrade, Alpaca) sets it to
        # shares × current_price, and the value below is derived from the price rather
        # than read from it. An instrument whose market value is not that plain product
        # — an option, via its contract multiplier — would therefore be understated.
        elif holding.get("market_value") is not None:
             current_price = holding.get("current_price", 1.0)
        elif purchase_price == 1.0 or is_manual_asset or symbol == "CASH":
             current_price = 1.0
        elif holding.get("current_price") and holding.get("current_price") > 0:
             # A live quote is newer than anything pinned in the CSV, so it wins and
             # brings its day move with it. The pin is what we fall back to when the
             # market has no price to give — an untradable fund, a failed lookup — which
             # is the case it exists for.
             live_price = None if is_demo_seeded else price_map.get(symbol)
             if live_price:
                 current_price = live_price
                 day_change_pct = day_change_map.get(symbol)
             else:
                 current_price = holding["current_price"]
             # Recalculate is_cash_or_pension if price > 1.0 (some funds use shares as dollars)
             if current_price > 1.0 and not is_manual_asset:
                 is_cash_or_pension = False
        else:
            # Check price_map -> LKG fallback -> purchase_price
            current_price = price_map.get(symbol)
            if current_price is not None:
                 # Claim a day move only for the live-quoted price itself. The fallbacks
                 # below are older prices, and pairing them with today's move would date
                 # the two halves of the row differently.
                 day_change_pct = day_change_map.get(symbol)
            if current_price is None:
                 current_price = lkg_data.get(symbol)

            if current_price is None:
                 current_price = purchase_price

            if current_price is None or (isinstance(current_price, str) and "N/A" in current_price):
                 current_price = 0.0

            # Feature: Reverse-engineer purchase price if user just provided return_pct
            if purchase_price == 0 and stored_return is not None and current_price and current_price > 0:
                purchase_price = current_price / (1 + (stored_return / 100))

            # If we found a price > 1.0 and it's not a known manual asset, it's likely a tradable stock/ETF
            if current_price > 1.0 and not is_manual_asset:
                is_cash_or_pension = False

        # A broker row took one of the pre-priced branches above, so it has a genuinely
        # current price but no direction — the same ticker held manually would show a
        # move while the synced copy showed nothing. That price is today's, so today's
        # market move belongs beside it. A manually pinned CSV price is deliberately
        # excluded: that mark is of no particular date, and pairing it with today's move
        # would date the two halves of the row differently.
        if (day_change_pct is None and holding.get("source") == "API"
                and not is_cash_or_pension and symbol != "CASH"):
            day_change_pct = day_change_map.get(symbol)

        # Units on record and nothing to value them with: no entry price, no stated
        # total, no live quote, no last-known-good and no pinned mark. Every other
        # row in this loop has at least one of those; this one is genuinely
        # underspecified, and the two ways of papering over it are both worse than
        # saying so. Valuing it at zero drops a real position out of every
        # allocation, risk and return figure while reporting a confident +0.0%
        # beside it — a fabricated flat on a row we can see we cannot price. Falling
        # back to the $1.00 synthetic unit the pension path uses would guess a number
        # on the holder's behalf and then present the guess as their statement.
        # So it is carried as unvalued and kept out of the totals, which is the one
        # answer that does not invent anything.
        is_unvalued = (shares > 0 and not is_manually_valued
                       and not (current_price and current_price > 0)
                       and not (purchase_price and purchase_price > 0))

        # Calculate values
        if is_unvalued:
            # Both stay 0.0 for the arithmetic below, but nothing downstream reads
            # them: this row is excluded from every total and its emitted value
            # fields are None, so the zero never reaches a caller as a fact.
            cost_basis = 0.0
            current_value = 0.0
        elif is_manually_valued:
            # The stated total IS the current value, and cost basis is read backwards
            # out of the stated return. This is the only branch that does not need an
            # entry price: assuming one (the $1.00 unit the branch below falls back to)
            # would value a 1,240-unit pension at $1,240 and report the statement's own
            # total as wrong. Guarded at -100%, where the division is undefined.
            current_value = stated_total
            if stored_return is not None and stored_return > -100:
                cost_basis = current_value / (1 + (stored_return / 100))
                purchase_price = cost_basis / shares
            else:
                cost_basis = shares * purchase_price
        else:
            # For pensions/cash, if purchase_price is 0, assume shares = invested amount.
            cost_basis = shares * (purchase_price if purchase_price > 0 else 1.0 if is_cash_or_pension else 0)

            if is_cash_or_pension and stored_return is not None:
                 # Apply the return percentage to the cost basis to find the true current market value
                 current_value = cost_basis * (1 + (stored_return / 100))
                 current_price = current_value / shares if shares > 0 else 1.0
            else:
                 current_value = shares * current_price

        current_value - cost_basis

        # For cash/pension, use stored return percentage if available
        if is_unvalued:
            # No basis, so no return. A typed return_pct is deliberately NOT reported
            # here: a percentage of an unknown amount is still unknown, and echoing it
            # as this row's gain would dress the missing entry price up as a result.
            gain_loss_pct = None
            status = "⚠️ Unvalued — needs an entry price or a total"
        elif is_cash_or_pension:
            if stored_return is not None:
                gain_loss_pct = stored_return
                status = f"💰 +{stored_return}% return" if stored_return > 0 else "💰 Cash"
            elif is_manually_valued and cost_basis > 0:
                # Total stated but no return typed. The return is computable against the
                # entry price this row does carry, so compute it — reporting the 0% below
                # would state a flat year as fact on a row we can see is not flat.
                gain_loss_pct = (current_value - cost_basis) / cost_basis * 100
                status = f"💰 {gain_loss_pct:+.1f}% return" if gain_loss_pct else "💰 Cash"
            else:
                gain_loss_pct = 0
                status = "💰 Cash/Pension"
        else:
            gain_loss_pct = ((current_price - purchase_price) / purchase_price * 100) if purchase_price > 0 else 0
            # Determine status emoji
            if gain_loss_pct > 20:
                status = "🟢🟢 BIG WINNER"
            elif gain_loss_pct > 0:
                status = "🟢 up"
            elif gain_loss_pct > -20:
                status = "🔴 down"
            else:
                status = "🔴🔴 BIG LOSER"

        if not is_unvalued:
            total_invested += cost_basis
            total_current_value += current_value

        # Track by account with currency. The account is registered even for an
        # unvalued row — the holding is real and its custody node should still appear
        # — but it contributes nothing to that account's figures.
        if account not in accounts:
            accounts[account] = {"invested": 0, "current": 0, "currencies": set()}
        if not is_unvalued:
            accounts[account]["invested"] += cost_basis
            accounts[account]["current"] += current_value
        accounts[account]["currencies"].add(currency)

        # Resolve name: check KG first, then fall back to yfinance/FMP/Polygon or symbol name
        name = symbol
        try:
            from tools.graph_memory import graph_memory
            graph_memory.load()
            kg_node = graph_memory.graph.nodes.get(symbol.upper())
        except Exception:
            kg_node = None

        if kg_node and kg_node.get("name"):
            name = kg_node["name"]
        else:
            try:
                from tools.market_data import get_stock_data
                stock_info = get_stock_data(symbol)
                name = stock_info.get("name") or stock_info.get("description") or symbol
            except Exception:
                name = symbol

        portfolio_data.append({
            "symbol": symbol,
            "name": name,
            "shares": shares,
            # An unvalued row has no entry price and no mark, and the agent prompts
            # render these strings verbatim — "$0.00" there reads as a real price of
            # zero rather than as an absent one. The _raw twins stay 0.0: they are the
            # numeric shape callers expect, and the editor keys its "this row is
            # hand-valued" affordance off purchase_price_raw == 0, which is how the
            # holder gets the Total Value input they need to fix this.
            "purchase_price": "—" if is_unvalued else f"${purchase_price:.2f}",
            "purchase_price_raw": purchase_price,
            "current_price": "—" if is_unvalued else f"${current_price:.2f}",
            # Numeric twins of the formatted strings above. Consumers that need to
            # compare or compute (thesis upside, UI colouring) must not re-parse
            # "$1,234.56" back into a float.
            "current_price_raw": current_price,
            # None where no live quote backed this row — the UI must render that as
            # "no direction known", never as flat.
            "day_change_pct": day_change_pct,
            "gain_loss": "—" if is_unvalued else f"{gain_loss_pct:+.1f}%",
            "gain_loss_pct": gain_loss_pct,
            "status": status,
            # This row holds units the engine cannot price. Consumers must skip it
            # rather than read its value fields as zero — those are None, not 0.0,
            # so an unguarded sum raises here instead of quietly understating a total.
            "is_unvalued": is_unvalued,
            "account": account,
            "currency": currency,
            "is_cash_or_pension": is_cash_or_pension,
            "is_private_asset": holding.get("is_private_asset", False),
            "return_pct": holding.get("return_pct"),
            # The total as TYPED, or None where this row's value was computed from a
            # price. The editor needs the distinction to decide whether the Total Value
            # cell is an input the holder owns or a figure the engine derived.
            "stated_total": stated_total if is_manually_valued else None,
            "source": holding.get("source", "Manual")
        })


    # --- CURRENCY NORMALIZATION ---
    base_currency = get_profile_base_currency()
    usd_cad_rate = get_exchange_rate("USD", "CAD")

    total_invested_base = 0.0
    total_current_value_base = 0.0
    total_invested_usd = 0.0
    total_current_value_usd = 0.0

    portfolio_data_enriched = []

    for item in portfolio_data:
        # A row with nothing to value it by carries None across all four value
        # fields and adds nothing to the totals. None rather than 0.0 on purpose:
        # a zero would sum silently and report the portfolio as complete when a
        # position is missing from it, which is the bug this branch exists to end.
        if item.get("is_unvalued"):
            item['value_native'] = None
            item['value_base'] = None
            item['value_usd'] = None
            item['value_cad'] = None
            portfolio_data_enriched.append(item)
            continue

        # Value off the raw floats, never the display strings: those are rounded to
        # 2dp, and a synthetic unit price (e.g. a pension at 1.055) truncates to 1.05
        # and silently understates the position.
        inv = item['purchase_price_raw'] * item['shares']
        curr = item['current_price_raw'] * item['shares']

        # Determine exchange rates dynamically
        item_curr = item.get('currency', 'USD')
        rate_to_base = get_exchange_rate(item_curr, base_currency)
        rate_to_usd = get_exchange_rate(item_curr, 'USD')
        rate_to_cad = get_exchange_rate(item_curr, 'CAD')

        inv_base = inv * rate_to_base
        curr_base = curr * rate_to_base

        total_invested_base += inv_base
        total_current_value_base += curr_base
        total_invested_usd += inv * rate_to_usd
        total_current_value_usd += curr * rate_to_usd

        # Add enriched data (with base currency and backward compatibility keys)
        # value_native is the position in its own currency, so it reconciles with the
        # entry and last prices shown beside it; the converted twins below do not.
        item['value_native'] = curr
        item['value_base'] = curr_base
        item['value_usd'] = curr * rate_to_usd
        item['value_cad'] = curr * rate_to_cad
        portfolio_data_enriched.append(item)

    # Sort by gain/loss percentage (exclude cash/pension, and rows with no return to
    # sort by — an unvalued row's gain_loss is "—", which is not a number and has no
    # place in a winners/losers ranking).
    tradable = [h for h in portfolio_data
                if not h.get("is_cash_or_pension") and not h.get("is_unvalued")]
    tradable.sort(key=lambda x: float(x["gain_loss"].replace("%", "").replace("+", "")), reverse=True)

    total_gain_loss_base = total_current_value_base - total_invested_base
    total_gain_loss_pct = (total_gain_loss_base / total_invested_base * 100) if total_invested_base > 0 else 0

    # Account summaries with currency and CASH BREAKDOWN
    account_summaries = []

    # Pre-calculate cash per account
    account_cash = {}
    CASH_ETFS = ["CASH.TO", "PSA.TO", "HISA.TO", "BIL", "SGOV", "CASH"]

    for item in portfolio_data_enriched:
        acc = item['account']
        sym = item['symbol']
        val = item['value_base']

        if acc not in account_cash:
            account_cash[acc] = 0.0

        # value_base is None on an unvalued row, so the membership test is not enough
        # on its own — a hand-entered CASH row with no amount would raise here.
        if sym in CASH_ETFS and val is not None:
            account_cash[acc] += val

    for acc, vals in accounts.items():
        # Standardize account-level totals in base currency. Unvalued rows are
        # excluded throughout: an account holding one reports the total of what is
        # known, and the notice on the summary says which position is missing from it.
        acc_valued = [i for i in portfolio_data_enriched
                      if i['account'] == acc and not i.get('is_unvalued')]
        acc_total_base = sum(i['value_base'] for i in acc_valued)
        acc_cash_base = sum(i['value_base'] for i in acc_valued
                            if i['symbol'] == "CASH" or i['symbol'] in CASH_ETFS)

        # Calculate cost basis in base currency for this account
        acc_cost_base = 0.0
        for i in acc_valued:
            inv_native = i['purchase_price_raw'] * i['shares']
            acc_cost_base += inv_native * get_exchange_rate(i.get('currency', 'USD'), base_currency)

        acc_gain_pct = ((acc_total_base - acc_cost_base) / acc_cost_base * 100) if acc_cost_base > 0 else 0
        currencies = "/".join(sorted(vals["currencies"]))

        # Get holdings in this account (for explicit listing)
        acc_holdings = [
            f"{i['symbol']}: {i['gain_loss']}"
            for i in portfolio_data_enriched
            if i['account'] == acc and not i.get('is_cash_or_pension')
        ]

        account_summaries.append({
            "account": acc,
            "currency": currencies,
            "total_value": f"${acc_total_base:,.2f} {base_currency}",
            "invested_value": f"${acc_cost_base:,.2f} {base_currency}",
            "cash_value": f"${acc_cash_base:,.2f} {base_currency}",
            "return": f"{acc_gain_pct:+.1f}%",
            "holdings": acc_holdings
        })

    # Top winners and losers (tradable only)
    winners = [h for h in tradable if float(h["gain_loss"].replace("%", "").replace("+", "")) > 0][:5]
    losers = [h for h in tradable if float(h["gain_loss"].replace("%", "").replace("+", "")) < 0][-5:]

    # Separate Liquid Cash vs. Locked Pension vs. Investments
    liquid_cash_base = 0.0
    locked_pension_base = 0.0
    cash_etfs_base = 0.0

    # Symbols considered "Liquid Cash" but traded as ETFs (Price != 1.0)
    CASH_ETFS = ["CASH.TO", "PSA.TO", "HISA.TO", "BIL", "SGOV"]

    for item in portfolio_data_enriched:
        sym = item['symbol']
        val_base = item['value_base']

        # Nothing to add to any of the three buckets, and every branch below sums.
        # A pension row is the likeliest thing to arrive unvalued, and it is exactly
        # the bucket ("locked_pension_base") a None would raise in.
        if val_base is None:
            continue

        # 1. True Cash (Currency)
        if sym == "CASH":
            liquid_cash_base += val_base

        # 2. Cash Equivalent ETFs (Liquid)
        elif sym in CASH_ETFS:
            cash_etfs_base += val_base

        # 3. Locked/Pension Assets
        # Identify if it's a Private Asset (not a tradable ticker)
        is_manual_asset = item.get("is_private_asset", False)
        if is_manual_asset or "PENSION" in item['account'].upper() or "LOCKED" in item['account'].upper():
            locked_pension_base += val_base

    # Calculate totals
    total_liquid_base = liquid_cash_base + cash_etfs_base

    # Positions the engine holds units for but cannot price. Named individually
    # rather than counted: "1 holding excluded" tells the holder something is wrong
    # without telling them which row to go and fix.
    unvalued_holdings = [
        {
            "symbol": h["symbol"],
            "account": h["account"],
            "shares": h["shares"],
            "reason": "no entry price and no stated total value",
        }
        for h in portfolio_data_enriched if h.get("is_unvalued")
    ]
    # Deliberately NOT folded into integration_notices. That field means one specific
    # thing — a broker nobody asked — and this means another: a holding nobody can
    # price. Overloading one field with two meanings is what made a broker's
    # market_value read as a hand-typed statement total, and consumers label
    # integration_notices "not synced (never asked)", which would misdescribe this.
    unvalued_notice = ""
    if unvalued_holdings:
        unvalued_notice = (
            "{} excluded from every total — {} states units but neither an entry price "
            "nor a total value, so there is no basis to value {}. Enter either one in "
            "the portfolio editor.".format(
                ", ".join(f"{h['symbol']} ({h['account']})" for h in unvalued_holdings),
                "it" if len(unvalued_holdings) == 1 else "they",
                "it" if len(unvalued_holdings) == 1 else "them",
            )
        )

    summary = {
        "holdings": portfolio_data_enriched,
        "base_currency": base_currency,
        "total_value_base": total_current_value_base,
        "total_value_cad": total_current_value_base * get_exchange_rate(base_currency, "CAD"),
        "total_value_usd": total_current_value_base * get_exchange_rate(base_currency, "USD"),
        "total_invested_cad": total_invested_base * get_exchange_rate(base_currency, "CAD"),
        "total_invested_usd": total_invested_base * get_exchange_rate(base_currency, "USD"),
        "total_gain_loss_cad": total_gain_loss_base * get_exchange_rate(base_currency, "CAD"),
        "total_gain_loss_usd": total_gain_loss_base * get_exchange_rate(base_currency, "USD"),
        "last_sync_time": datetime.now().isoformat(),
        "top_winners": [f"{w['symbol']}: {w['gain_loss']}" for w in winners],
        "top_losers": [f"{l['symbol']}: {l['gain_loss']}" for l in losers],
        "liquidity": {
            "total_liquid_cash": f"${total_liquid_base:,.2f} {base_currency}",
            "pure_cash": f"${liquid_cash_base:,.2f} {base_currency}",
            "cash_equivalents": f"${cash_etfs_base:,.2f} {base_currency}",
            "locked_pension_value": f"${locked_pension_base:,.2f} {base_currency}",
            "note": "Liquid Cash includes CASH currency and Money Market ETFs (e.g. CASH.TO)"
        },
        "accounts": account_summaries,
        "summary": {
            "total_invested": f"${total_invested_base:,.2f} {base_currency}",
            "current_value": f"${total_current_value_base:,.2f} {base_currency}",
            "total_gain_loss": f"${total_gain_loss_base:+,.2f} {base_currency}",
            "total_return": f"{total_gain_loss_pct:+.1f}%",
            "number_of_positions": len(portfolio_data),
            "verdict": "🟢 Your portfolio is UP overall!" if total_gain_loss_base > 0 else "🔴 Your portfolio is DOWN overall",
            "exchange_rate_used": f"1 USD = {get_exchange_rate('USD', base_currency):.2f} {base_currency}"
        },
        # Raw Data for UI/Agents
        "percent_return": total_gain_loss_pct,
        "usd_cad_rate": usd_cad_rate,
        "sync_errors": sync_errors,
        # Brokers that were never asked. Recorded rather than dropped: "we did not
        # check Questrade" and "we checked and it was fine" are different answers,
        # and only one of them means the totals below are complete.
        "integration_notices": integration_notices,
        # Holdings deliberately left out of every total above because there is no
        # basis to value them. Empty on a complete portfolio, so a caller can treat
        # a non-empty list as "this total understates the book, by these positions".
        "unvalued_holdings": unvalued_holdings,
        "unvalued_notice": unvalued_notice,
    }

    return summary

@log_exceptions()
def sync_portfolio_to_graph(portfolio_data_enriched: list[dict[str, Any]] | None = None) -> None:
    """
    Sync portfolio holdings to the Knowledge Graph.
    Separated from the main data loop to allow background execution.
    """
    try:
        from tools.graph_memory import graph_memory

        if portfolio_data_enriched is None:
            # Dynamically compute summary to resolve current holdings list
            summary = get_portfolio_summary(force=True)
            portfolio_data_enriched = summary.get("holdings", [])

        graph_memory.load()
        for h in portfolio_data_enriched:
            sym = h["symbol"].upper()
            acc = h["account"]
            sector = h.get("sector") or "Unknown"

            # Holdings dicts no longer carry a `sector` field, so this used to fall
            # through to "Unknown" for every name and write zero IN_SECTOR edges —
            # leaving the graph with a single legacy tag and making downstream nodes
            # report a bogus single-sector concentration. Resolve the sector via the
            # canonical cached classifier (static universe → KG → yfinance) so the
            # whole book is correctly tagged again.
            if sector == "Unknown":
                try:
                    from tools.opportunity_scanner import _get_sector_for_ticker
                    resolved = _get_sector_for_ticker(sym)
                    if resolved and resolved not in ("Unknown", "Private/Manual Holding"):
                        sector = resolved
                except Exception:
                    pass

            # Add nodes and relationships
            graph_memory.add_entity(sym, "Stock", {"sector": sector, "owned": True})
            graph_memory.add_relationship("Portfolio", sym, "OWNS", {"account": acc})

            # Add sector relationship if known
            if sector and sector != "Unknown":
                graph_memory.add_entity(sector, "Sector")
                graph_memory.add_relationship(sym, sector, "IN_SECTOR")

        graph_memory.prune_orphans()
        graph_memory.save()
        safe_print("🕸️ Knowledge Graph: Portfolio Sync Complete")

    except Exception as e:
        safe_print(f"⚠️ Graph sync failed: {e}")


@log_exceptions()
def get_tradeable_symbols() -> list[str]:
    """
    Get a list of tradeable symbols from the portfolio,
    excluding cash, pension wrappers, and other non-tradeable entries.
    Useful for risk tools that require historical price data.
    """
    holdings = load_portfolio()
    if isinstance(holdings, dict) and "error" in holdings:
        return []

    tradeable = []
    for h in holdings:
        if not isinstance(h, dict) or "symbol" not in h:
            continue

        symbol = h["symbol"].upper()
        purchase_price = _coerce_number(h.get("purchase_price"))
        # Identify if it's a Private Asset (not a tradable ticker)
        is_manual_asset = h.get("is_private_asset", False)

        # Filter rules:
        # 1. Not a manual asset (pension/private fund)
        # 2. Not CASH (literal cash)
        # 3. purchase_price != 1.0 (marker for placeholders where shares = dollar amount).
        #    NOTE: this must be an equality check, not `> 1.0` — a real holding whose
        #    cost basis was never entered (purchase_price == 0, relying on the
        #    return_pct reverse-engineering fallback in _compute_portfolio_summary)
        #    is still tradeable and must not be swept into this exclusion.
        # 4. symbol exists
        if (symbol and
            not is_manual_asset and
            symbol != "CASH" and
            purchase_price != 1.0):
            tradeable.append(symbol)

    return list(set(tradeable)) # Unique symbols


@log_exceptions()
def get_portfolio_decision_context(
    symbols: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """
    Build a compact, verification-oriented portfolio snapshot for trade decisions.

    This is intentionally separate from the UI summary: agents need an exact source
    of truth for "do I hold this?" plus magnitude fields for trim/sell sizing.
    """
    summary = get_portfolio_summary(force=force)
    if not isinstance(summary, dict) or summary.get("error"):
        return {
            "error": summary.get("error", "Portfolio summary unavailable") if isinstance(summary, dict) else "Portfolio summary unavailable",
            "owned_symbols": [],
            "holdings": [],
        }

    total_value_cad = _coerce_number(summary.get("total_value_cad"))
    total_value_usd = _coerce_number(summary.get("total_value_usd"))
    # base_currency/total_value_base may be absent from an older cached/LKG snapshot
    # saved before these fields existed — fall back to CAD (the historical default).
    base_currency = str(summary.get("base_currency") or "CAD").upper()
    total_value_base = _coerce_number(summary.get("total_value_base"))
    if not total_value_base:
        total_value_base = total_value_cad if base_currency == "CAD" else total_value_usd
    holdings = []

    for raw in summary.get("holdings", []):
        if not isinstance(raw, dict) or not raw.get("symbol"):
            continue

        # A holding the engine could not value. _coerce_number would turn its None
        # into 0.0 here and hand an agent a $0.00 position as verified fact — the
        # exact fabrication the summary went to the trouble of avoiding. Carry the
        # absence through instead, and let the flag say why.
        is_unvalued = bool(raw.get("is_unvalued"))
        if is_unvalued:
            value_cad = value_usd = value_base = None
            allocation_pct = None
        else:
            value_cad = _coerce_number(raw.get("value_cad"))
            value_usd = _coerce_number(raw.get("value_usd"))
            value_base = _coerce_number(raw.get("value_base"))
            if not value_base:
                value_base = value_cad if base_currency == "CAD" else value_usd
            allocation_pct = (value_cad / total_value_cad * 100) if total_value_cad > 0 else None
        symbol = str(raw.get("symbol", "")).upper()

        holdings.append({
            "symbol": symbol,
            "account": raw.get("account", "Unknown"),
            "source": raw.get("source", "Unknown"),
            "shares": raw.get("shares"),
            "current_price": raw.get("current_price"),
            "purchase_price": raw.get("purchase_price"),
            "gain_loss": raw.get("gain_loss"),
            "currency": raw.get("currency", "USD"),
            "value_base": round(value_base, 2) if value_base is not None else None,
            "value_cad": round(value_cad, 2) if value_cad is not None else None,
            "value_usd": round(value_usd, 2) if value_usd is not None else None,
            "allocation_pct": round(allocation_pct, 2) if allocation_pct is not None else None,
            "is_cash_or_pension": bool(raw.get("is_cash_or_pension")),
            "is_unvalued": is_unvalued,
        })

    owned_symbols = sorted({h["symbol"] for h in holdings})
    requested_symbols = []
    if symbols:
        for symbol in symbols:
            clean_symbol = str(symbol or "").strip().upper()
            if not clean_symbol:
                continue
            matches = [h for h in holdings if h["symbol"] == clean_symbol]
            requested_symbols.append({
                "symbol": clean_symbol,
                "owned": bool(matches),
                "matches": matches,
            })

    return {
        "profile": get_active_profile(),
        "as_of": summary.get("last_sync_time"),
        "is_stale": bool(summary.get("is_stale")),
        "sync_errors": summary.get("sync_errors", []),
        "base_currency": base_currency,
        "total_value_base": round(total_value_base, 2),
        "total_value_cad": round(total_value_cad, 2),
        "total_value_usd": round(total_value_usd, 2),
        "summary": summary.get("summary", {}),
        "owned_symbols": owned_symbols,
        "holdings": holdings,
        "requested_symbols": requested_symbols,
        # Carried through so the judge knows the total above is the sum of what could
        # be valued, not of what is held. Without it, an advisor that correctly says
        # "your workplace pension is not included" reads as contradicting verified data.
        "unvalued_holdings": summary.get("unvalued_holdings", []),
        "unvalued_notice": summary.get("unvalued_notice", ""),
        "verification_note": (
            "Use only owned_symbols as the user's verified current holdings. "
            "If a ticker is absent, do not recommend trimming/selling it as a current position."
        ),
    }


if __name__ == "__main__":
    safe_print(get_portfolio_summary())
