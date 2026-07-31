"""
Trade Journal / Active Thesis Management
Zero-Effort Auto-Sync Thesis System: Tracks active investment theses and
automatically reconciles/archives them when portfolio holdings change.
"""
import json
import os
from datetime import date, datetime
from typing import Any

from tools.exception_logger import log_exceptions
from tools.json_store import write_json_atomic
from tools.user_profile import get_data_path

_LEGACY_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_LEGACY_JOURNAL_FILE = os.path.join(_LEGACY_DATA_DIR, "trade_journal.json")


def _journal_file() -> str:
    return get_data_path("trade_journal.json")


@log_exceptions()
def _ensure_data_dir():
    os.makedirs(os.path.dirname(_journal_file()), exist_ok=True)


@log_exceptions()
def _load_journal() -> list[dict[str, Any]]:
    for path in (_journal_file(), _LEGACY_JOURNAL_FILE):
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []


@log_exceptions()
def _save_journal(entries: list[dict[str, Any]]):
    _ensure_data_dir()
    write_json_atomic(_journal_file(), entries)


@log_exceptions()
def _sync_to_memory(entry: dict[str, Any]):
    """Sync an active thesis entry to user_memory.json's active_theses."""
    try:
        from tools.memory import add_active_thesis, get_active_theses
        existing = get_active_theses()
        sym = entry.get("symbol", "").upper()
        # Avoid duplicate open thesis for same symbol in memory
        for t in existing:
            if t.get("symbol", "").upper() == sym and str(t.get("id")) == str(entry.get("id")):
                return
        add_active_thesis({
            "id": str(entry.get("id")),
            "symbol": sym,
            "action": entry.get("action", "BUY"),
            "target_price": entry.get("target_price"),
            "stop_loss": entry.get("stop_loss"),
            "conditions": entry.get("thesis"),
            "conviction": entry.get("conviction", "Medium"),
            "time_horizon": entry.get("time_horizon", "Medium Term"),
            "created_at": entry.get("date", datetime.now().isoformat())
        })
    except Exception:
        pass


@log_exceptions()
def _remove_from_memory(entry_id: Any):
    """Remove a thesis from memory active_theses when closed or deleted."""
    try:
        from tools.memory import delete_active_thesis
        delete_active_thesis(str(entry_id))
    except Exception:
        pass


@log_exceptions()
def log_trade(
    symbol: str,
    action: str,
    price: float = 0.0,
    quantity: float = 0.0,
    thesis: str = "",
    time_horizon: str = "Medium Term",
    conviction: str = "Medium",
    target_price: float | None = None,
    stop_loss: float | None = None
) -> str:
    """
    Log a new trade decision / thesis entry.
    action: BUY, SELL, HOLD, ADD, TRIM
    """
    entries = _load_journal()
    symbol_clean = symbol.upper().strip()

    if not thesis:
        thesis = f"Strategic {action.upper()} position in {symbol_clean} tracked against benchmark."

    entry = {
        "id": len(entries) + 1,
        "date": date.today().isoformat(),
        "status": "OPEN" if action.upper() in ["BUY", "ADD", "HOLD"] else "CLOSED",
        "symbol": symbol_clean,
        "action": action.upper(),
        "price": price,
        "quantity": quantity,
        "thesis": thesis,
        "target_price": target_price,
        "stop_loss": stop_loss,
        "time_horizon": time_horizon,
        "conviction": conviction,
        "outcome": None,
        "exit_date": None,
        "exit_price": None,
        "lessons_learned": None
    }

    entries.append(entry)
    _save_journal(entries)

    if entry["status"] == "OPEN":
        _sync_to_memory(entry)

    return f"✅ Logged {action} for {quantity} of {symbol_clean} at ${price:.2f}"


@log_exceptions()
def close_trade(symbol: str, exit_price: float, outcome: str = "Closed", lessons_learned: str = "") -> str:
    """
    Close an active trade/thesis entry and record the outcome and lesson learned.
    """
    entries = _load_journal()
    updated = False
    symbol_clean = symbol.upper().strip()

    for entry in reversed(entries):
        if entry.get("symbol", "").upper() == symbol_clean and entry.get("status") == "OPEN":
            entry["status"] = "CLOSED"
            entry["exit_date"] = date.today().isoformat()
            entry["exit_price"] = exit_price

            entry_price = entry.get("price", 0.0)
            if entry_price and entry_price > 0 and exit_price:
                ret_pct = ((exit_price - entry_price) / entry_price) * 100
                outcome_str = f"{outcome} ({ret_pct:+.1f}%)"
            else:
                outcome_str = outcome

            entry["outcome"] = outcome_str
            entry["lessons_learned"] = lessons_learned or "Position closed per strategic allocation update."
            updated = True
            _remove_from_memory(entry.get("id"))
            break

    if updated:
        _save_journal(entries)
        return f"✅ Closed active thesis for {symbol_clean}. Outcome: {outcome}."
    else:
        return f"⚠️ No open thesis found for {symbol_clean}."


@log_exceptions()
def get_active_trades() -> list[dict[str, Any]]:
    """Return all active open theses."""
    entries = _load_journal()
    return [e for e in entries if e.get("status") == "OPEN"]


@log_exceptions()
def get_trade_history(symbol: str | None = None) -> list[dict[str, Any]]:
    """Return full trade/thesis history, optionally filtered by symbol."""
    entries = _load_journal()
    if symbol:
        return [e for e in entries if e.get("symbol", "").upper() == symbol.upper().strip()]
    return entries


@log_exceptions()
def delete_trade(trade_id: int | str) -> str:
    """Delete a trade/thesis entry by ID."""
    entries = _load_journal()
    initial_len = len(entries)

    str_id = str(trade_id)
    entries = [e for e in entries if str(e.get("id")) != str_id]

    if len(entries) < initial_len:
        _save_journal(entries)
        _remove_from_memory(str_id)
        return f"✅ Deleted trade ID {trade_id}."
    return f"⚠️ Trade ID {trade_id} not found."


@log_exceptions()
def reconcile_with_holdings(holdings: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Auto-Sync Reconciliation: Checks active open theses against current portfolio holdings.
    If a symbol with an open thesis is no longer held (shares == 0 or missing),
    the thesis is automatically archived as closed.

    Reconciles nothing when the holdings could not be read, or read as zero
    positions — the result then carries ``reconciled: False`` and a
    ``skipped_reason`` instead of an archive count.
    """
    # This function's only action is destructive — it closes theses — and it infers
    # the closure from an ABSENCE. So a holdings list that arrived broken must stop
    # it, not drive it: an error envelope, or zero readable positions, would archive
    # every open thesis at once and stamp them "exited during portfolio sync".
    # Refusing costs a stale open thesis until the next run; proceeding rewrites the
    # journal. (A genuinely emptied portfolio also lands here, and is also refused —
    # from here it is indistinguishable from a failed load, and the safe reading of
    # an ambiguous signal is the one that does not delete anything.)
    if isinstance(holdings, dict) or not isinstance(holdings, list):
        return {"reconciled": False, "auto_archived_count": 0, "auto_archived_symbols": [],
                "skipped_reason": "Portfolio could not be read; theses left untouched."}

    held_symbols = set()
    for item in holdings:
        if not isinstance(item, dict):
            continue
        sym = str(item.get("symbol") or "").upper().strip()
        try:
            shares = float(str(item.get("shares", 0)).replace(",", "") or 0)
        except (TypeError, ValueError):
            continue
        if sym and shares > 0 and sym != "CASH":
            held_symbols.add(sym)

    if not held_symbols:
        return {"reconciled": False, "auto_archived_count": 0, "auto_archived_symbols": [],
                "skipped_reason": "No positions read from the portfolio; theses left untouched."}

    entries = _load_journal()

    archived_count = 0
    auto_archived_symbols = []

    for entry in entries:
        if entry.get("status") == "OPEN":
            sym = entry.get("symbol", "").upper().strip()
            # If position is no longer in portfolio holdings, auto-archive it
            if sym not in held_symbols and sym not in ["CASH", "USD"]:
                entry["status"] = "CLOSED"
                entry["exit_date"] = date.today().isoformat()
                entry["outcome"] = "Auto-Archived (Portfolio Exit)"
                entry["lessons_learned"] = "Position was exited during portfolio sync."
                _remove_from_memory(entry.get("id"))
                archived_count += 1
                auto_archived_symbols.append(sym)

    if archived_count > 0:
        _save_journal(entries)

    return {
        "reconciled": True,
        "auto_archived_count": archived_count,
        "auto_archived_symbols": auto_archived_symbols
    }


if __name__ == "__main__":
    print(log_trade("AAPL", "BUY", 150.00, 100, "Strong earnings growth", "Long Term", "High"))
    print(close_trade("AAPL", 180.00, "Profit", "Held through earnings as planned"))
    print(json.dumps(get_trade_history(), indent=2))
