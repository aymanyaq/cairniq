"""
API Router for Trade Journal & Active Thesis Management.
Exposes REST endpoints for the Zero-Effort Auto-Sync Thesis System.
"""
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent.logger import log_to_component
from tools.portfolio_csv import load_portfolio
from tools.trade_journal import (
    close_trade,
    delete_trade,
    get_active_trades,
    get_trade_history,
    log_trade,
    reconcile_with_holdings,
)

router = APIRouter()


class LogThesisRequest(BaseModel):
    symbol: str
    action: str = "BUY"
    price: float = 0.0
    quantity: float = 0.0
    thesis: str = ""
    time_horizon: str = "Medium Term"
    conviction: str = "Medium"
    target_price: float | None = None
    stop_loss: float | None = None


class CloseThesisRequest(BaseModel):
    symbol: str
    exit_price: float
    outcome: str = "Closed"
    lessons_learned: str = ""


@router.get("/api/journal")
def list_journal(symbol: str | None = None):
    """Retrieve full active theses and historical trade decisions."""
    history = get_trade_history(symbol=symbol)
    active = [e for e in history if e.get("status") == "OPEN"]
    closed = [e for e in history if e.get("status") == "CLOSED"]
    return JSONResponse({
        "active_theses": active,
        "archived_history": closed,
        "total_count": len(history)
    })


@router.post("/api/journal")
def create_thesis(req: LogThesisRequest):
    """Create or log a new investment thesis/trade."""
    if not req.symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")

    result = log_trade(
        symbol=req.symbol,
        action=req.action,
        price=req.price,
        quantity=req.quantity,
        thesis=req.thesis,
        time_horizon=req.time_horizon,
        conviction=req.conviction,
        target_price=req.target_price,
        stop_loss=req.stop_loss
    )
    log_to_component("server", "Journal", f"Logged thesis for {req.symbol}", level=logging.INFO)
    return JSONResponse({"status": "success", "message": result})


@router.post("/api/journal/close")
def close_thesis(req: CloseThesisRequest):
    """Close an active thesis and record exit price/lesson."""
    if not req.symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")

    result = close_trade(
        symbol=req.symbol,
        exit_price=req.exit_price,
        outcome=req.outcome,
        lessons_learned=req.lessons_learned
    )
    log_to_component("server", "Journal", f"Closed thesis for {req.symbol}", level=logging.INFO)
    return JSONResponse({"status": "success", "message": result})


@router.delete("/api/journal/{trade_id}")
def remove_thesis(trade_id: str):
    """Remove a thesis or trade entry by ID."""
    result = delete_trade(trade_id)
    return JSONResponse({"status": "success", "message": result})


@router.post("/api/journal/reconcile")
def auto_reconcile_journal():
    """Auto-sync active theses against live portfolio holdings."""
    holdings = load_portfolio()
    if isinstance(holdings, dict) and "error" in holdings:
        return JSONResponse({"status": "error", "message": holdings["error"]}, status_code=500)

    res = reconcile_with_holdings(holdings)
    return JSONResponse(res)
