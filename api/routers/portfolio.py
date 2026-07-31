import csv
import io
import logging
import math
import os
from datetime import datetime, timedelta

from fastapi import APIRouter, Body, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from agent.logger import log_to_component
from tools.daily_cache import _get_today, get_or_compute

router = APIRouter()


def sanitize_for_json(obj):
    """
    Recursively sanitize data for JSON serialization by replacing NaN and Infinity with None.

    This prevents JSON encoding errors when numerical calculations produce invalid values.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    return obj

@router.get("/api/portfolio/download-template")
def download_portfolio_template():
    """Generate and download a sample portfolio CSV template."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["symbol", "shares", "purchase_price", "account", "currency", "return_pct"])
    writer.writerow(["AAPL", "10", "150.00", "Brokerage IRA", "USD", ""])
    writer.writerow(["NVDA", "5", "400.00", "Robinhood", "USD", ""])
    writer.writerow(["TD.TO", "100", "85.00", "Questrade TFSA", "CAD", ""])
    writer.writerow(["CASH", "5000", "1.00", "Savings Account", "CAD", ""])

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=portfolio_template.csv"}
    )

@router.post("/api/portfolio/upload")
def upload_portfolio_csv(file: UploadFile = File(...)):
    """Upload a CSV and save it as the active portfolio."""
    import shutil

    from tools.user_profile import get_data_path

    if not file.filename.endswith('.csv'):
        return JSONResponse({"error": "Only CSV files are allowed."}, status_code=400)

    try:
        csv_path = get_data_path("my_portfolio.csv")
        with open(csv_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        from tools.portfolio_csv import get_portfolio_summary, load_portfolio
        from tools.trade_journal import reconcile_with_holdings
        get_portfolio_summary(force=True)
        reconcile_res = reconcile_with_holdings(load_portfolio())

        msg = "Portfolio uploaded successfully."
        if reconcile_res.get("auto_archived_count", 0) > 0:
            msg += f" Auto-archived {reconcile_res['auto_archived_count']} exited thesis position(s)."
        elif reconcile_res.get("skipped_reason"):
            # The upload itself succeeded; say plainly that the journal was not
            # touched rather than letting silence imply it was reconciled.
            msg += f" Thesis reconciliation skipped: {reconcile_res['skipped_reason']}"

        return {"status": "success", "message": msg, "reconciliation": reconcile_res}
    except Exception as e:
        log_to_component("server", "Portfolio", f"Error uploading portfolio: {e}", level=logging.ERROR)
        return JSONResponse({"error": "Failed to upload portfolio"}, status_code=500)

@router.post("/api/portfolio/parse-statement")
def parse_statement(payload: dict = Body(...)):
    """Draft holding rows from pasted statement text, for accounts that cannot sync.

    A READ of the user's text and nothing more. This endpoint writes no file,
    clears no cache and touches no store — it hands drafts back to the editor,
    where they land as unsaved inputs the user reviews and then saves through the
    ordinary /api/portfolio/save path. The model is therefore never the last
    thing between an extraction and the ledger; a human always is.

    Matching a draft against what is already in the table is deliberately NOT
    done here. The editor's rows are the live truth — they include unsaved edits
    and rows added since the page loaded — and a server-side match against the
    saved CSV would disagree with what the user is looking at.
    """
    from tools.statement_parser import parse_statement_text

    text = payload.get("text") or ""
    result = parse_statement_text(
        text,
        default_account=payload.get("default_account") or "",
        default_currency=payload.get("default_currency") or "",
    )

    # The pasted text is a private financial document; only its size is logged.
    log_to_component(
        "server", "Portfolio",
        f"Statement parse: {result['reason']}",
        data={"chars": len(str(text)), "rows": result["row_count"], "dropped": len(result["dropped"])},
    )
    return result


def _compute_benchmark_data():
    """Compute portfolio vs SPY benchmark comparison (cached daily)."""
    import yfinance as yf

    from tools.portfolio_tracker import get_portfolio_history, snapshot_portfolio

    log_to_component("server", "Benchmark", f"Starting computation for logical date {_get_today()}...")

    # Ensure today's snapshot exists
    try:
        start_snap = datetime.now()
        snapshot_portfolio()
        log_to_component("server", "Benchmark", f"Portfolio snapshot took {(datetime.now() - start_snap).total_seconds():.2f}s")
    except Exception as e:
        log_to_component("server", "Benchmark", f"Snapshot failed: {e}", level=logging.ERROR)
        pass

    # Get portfolio history (all valid days)
    start_hist = datetime.now()
    portfolio_df = get_portfolio_history("all")
    log_to_component("server", "Benchmark", f"History fetch took {(datetime.now() - start_hist).total_seconds():.2f}s ({len(portfolio_df)} points)")

    portfolio_points = []

    if not portfolio_df.empty:
        # Start return to normalize
        start_ret = float(portfolio_df.iloc[0].get("percent_return", 0))

        for _, row in portfolio_df.iterrows():
            total_val = float(row.get("total_value_usd", 0))
            abs_ret = float(row.get("percent_return", 0))
            norm_ret = abs_ret - start_ret

            portfolio_points.append({
                "date": str(row["date"])[:10],
                "value": total_val,
                "norm_return_pct": norm_ret,
                "return_pct": abs_ret
            })

    # Get SPY benchmark history matching dates
    spy_points = []
    spy_return = "N/A"

    if len(portfolio_points) > 0:
        try:
            start_date_str = portfolio_points[0]["date"]
            log_to_component("server", "Benchmark", f"Fetching SPY benchmark starting from {start_date_str}...")
            # Fetch from one day before just to be safe with timezone boundaries
            start_dt = datetime.strptime(start_date_str, "%Y-%m-%d") - timedelta(days=1)

            logging.getLogger("yfinance").setLevel(logging.CRITICAL)

            ticker = yf.Ticker("SPY")
            spy_df = ticker.history(start=start_dt.strftime("%Y-%m-%d"))

            if spy_df is not None and not spy_df.empty:
                # Find the closest SPY price to our true start date
                spy_start_val = spy_df.iloc[0]["Close"]
                spy_end_val = spy_df.iloc[-1]["Close"]

                # Make dictionary mapped by date string "YYYY-MM-DD"
                spy_dict = {}
                for idx, row in spy_df.iterrows():
                    date_str = str(idx.date())
                    norm_spy = ((row["Close"] / spy_start_val) - 1.0) * 100.0
                    spy_dict[date_str] = norm_spy

                # Match SPY points to the exact dates in the portfolio
                last_spy = 0.0
                for pt in portfolio_points:
                    date_str = pt["date"]
                    if date_str in spy_dict:
                        last_spy = spy_dict[date_str]
                    spy_points.append({
                        "date": date_str,
                        "norm_return_pct": last_spy
                    })

                # Set overall SPY return over this exact customized period
                spy_return = f"{((spy_end_val / spy_start_val) - 1.0) * 100.0:+.1f}%"
                log_to_component("server", "Benchmark", f"SPY comparison matched ({len(spy_points)} points).")

        except Exception as e:
            log_to_component("server", "Benchmark", f"FAILED to fetch matching SPY history: {e}", level=logging.ERROR)
            # Ensure spy_points is at least as long as portfolio_points with 0s to avoid UI crashing
            if not spy_points:
                spy_points = [{"date": pt["date"], "norm_return_pct": 0.0} for pt in portfolio_points]

    # Get absolute portfolio return
    portfolio_return = "0.0%"
    if portfolio_points:
        portfolio_return = f"{portfolio_points[-1].get('return_pct', 0):+.1f}%"

        # Calculate real portfolio growth over this tracked time tracking period
        track_period_growth = portfolio_points[-1].get("norm_return_pct", 0)
        portfolio_return = f"{track_period_growth:+.1f}%"

    log_to_component("server", "Benchmark", f"Computation complete. Port: {portfolio_return}, SPY: {spy_return}")
    return {
        "portfolio_points": portfolio_points,
        "spy_points": spy_points,
        "portfolio_return": portfolio_return,
        "spy_return": spy_return,
        "generated_at": datetime.now().isoformat()
    }

@router.get("/api/benchmark")
def get_benchmark():
    """Get portfolio vs benchmark data (daily cached)."""
    data = get_or_compute("benchmark", _compute_benchmark_data)
    sanitized_data = sanitize_for_json(data)
    return JSONResponse(sanitized_data)


def _read_manual_price_columns(csv_path: str) -> dict:
    """Map (symbol, account) -> the manually-set price fields already in the CSV.

    Current Price and Market Value are CSV-only fields: load_portfolio honours them,
    but the editor renders price read-only and never sends them back. They are
    recovered from the file rather than from the payload because the summary feeding
    the page cannot tell a manual price from a live quote — echoing a quote back
    would pin the row to it forever, since _compute_portfolio_summary skips the live
    fetch for any row that already carries a current_price.
    """
    manual_prices = {}
    if not os.path.exists(csv_path):
        return manual_prices

    try:
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                symbol = (row.get("symbol") or row.get("Symbol") or "").strip().upper()
                if not symbol or symbol.startswith("#"):
                    continue
                current_price = (row.get("current_price") or row.get("Current Price") or "").strip()
                market_value = (row.get("market_value") or row.get("Market Value") or "").strip()
                if not current_price and not market_value:
                    continue
                account = (row.get("account") or row.get("Account") or "Unknown").strip()
                # Keyed on symbol+account: renaming either identifies a different
                # holding, whose old price should not follow it. Repeated lots collapse
                # onto one entry — one symbol in one account has a single price anyway.
                manual_prices[(symbol, account)] = {
                    "Current Price": current_price,
                    "Market Value": market_value,
                }
    except Exception as e:
        # A save is worth more than the preserved prices; carry on with an empty map.
        log_to_component("server", "Portfolio", f"Could not read existing prices from {csv_path}: {e}", level=logging.WARNING)

    return manual_prices


@router.post("/api/portfolio/save")
async def save_portfolio_data(data: list = Body(...)):
    """Save edited portfolio holdings to my_portfolio.csv."""
    import glob

    from tools.cache import clear_cache
    from tools.daily_cache import CACHE_DIR, _safe_cache_part, get_active_profile
    from tools.user_profile import get_data_path

    csv_path = get_data_path("my_portfolio.csv")

    try:
        manual_prices = _read_manual_price_columns(csv_path)

        rows = []
        for h in data:
            symbol = h.get("symbol", "").strip().upper()
            if not symbol: continue
            shares = float(h.get("shares") or 0.0)
            purchase_price = float(h.get("purchase_price") or 0.0)
            account = h.get("account", "Unknown").strip()
            currency = h.get("currency", "USD").strip().upper()
            return_pct = h.get("return_pct")
            if return_pct is not None and str(return_pct).strip() != "":
                try:
                    return_pct = float(return_pct)
                except ValueError:
                    return_pct = ""
            else:
                return_pct = ""
            asset_type = h.get("asset_type", "Public").strip()
            preserved = manual_prices.get((symbol, account), {})

            # A holder-stated total, for the rows that have no quotable price: a group
            # pension states units, a return and a total and nothing else. It is taken
            # from the payload ONLY for those rows, because a stated total suppresses
            # the live quote — accepting one on a market-priced ticker would freeze it
            # at today's number forever. The gate mirrors the editor's, which is what
            # decides whether the input is rendered at all.
            is_manually_valued = (
                asset_type.lower() == "private"
                or return_pct != ""
                or purchase_price in (0.0, 1.0)
            )
            stated_total = h.get("market_value")
            if is_manually_valued and stated_total is not None:
                # Empty string is meaningful: the holder cleared the field, and the row
                # reverts to being valued off units × price like any other.
                market_value = str(stated_total).strip()
            else:
                market_value = preserved.get("Market Value", "")

            rows.append({
                "Symbol": symbol,
                "Shares": shares,
                "Purchase Price": purchase_price,
                "Current Price": preserved.get("Current Price", ""),
                "Market Value": market_value,
                "Account": account,
                "Currency": currency,
                "Return Pct": return_pct,
                "Asset Type": asset_type,
            })

        # Carry a manual-price column only when some row fills it, so a portfolio that
        # is entirely live-quoted keeps its plain shape instead of gaining dead columns.
        columns = ["Symbol", "Shares", "Purchase Price"]
        if any(r["Current Price"] for r in rows):
            columns.append("Current Price")
        if any(r["Market Value"] for r in rows):
            columns.append("Market Value")
        columns += ["Account", "Currency", "Return Pct", "Asset Type"]

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        # 1. Clear session cache
        clear_cache()

        # 2. Clear daily cache files for the active profile
        profile_prefix = _safe_cache_part(get_active_profile())
        for filepath in glob.glob(os.path.join(CACHE_DIR, f"{profile_prefix}_*.json")):
            try:
                os.remove(filepath)
            except Exception:
                pass

        # Also sync portfolio changes to the graph database
        from tools.portfolio_csv import sync_portfolio_to_graph
        try:
            sync_portfolio_to_graph()
        except Exception:
            pass

        return {"status": "success", "message": "Portfolio saved successfully."}
    except Exception as e:
        log_to_component("server", "Portfolio", f"Error saving portfolio: {e}", level=logging.ERROR)
        return JSONResponse({"error": "Failed to save portfolio"}, status_code=500)


@router.get("/api/portfolio/asset-location")
def get_asset_location_analysis():
    """Returns portfolio asset location efficiency scores, leakages, and swap recommendations."""
    from tools.asset_location import analyze_asset_location
    res = analyze_asset_location()
    return sanitize_for_json(res)


@router.get("/api/portfolio/reconciliation")
def get_portfolio_reconciliation_endpoint():
    """Returns per-account position changes observed since the previous snapshot (4.10a).

    Read `status` before `changes`: `no_data` and `accruing` mean there is nothing
    to compare yet, which is a different claim from an unchanged portfolio. Every
    change carries `cause: "unclassified"` and must not be rendered as a trade or
    a cash flow.
    """
    from tools.portfolio_reconciliation import get_reconciliation
    return sanitize_for_json(get_reconciliation())


@router.get("/api/tax/policy-coverage")
def get_tax_policy_coverage_endpoint():
    """What the loss-deferral policy engine covers, per jurisdiction (4.7).

    `advice_ready` is false until a tax professional has reviewed a module
    against a specific `policy_version`. Until then these results may be used
    defensively — to stop and ask — and must not be quoted as tax treatment.
    """
    from tools.tax_policy import coverage_matrix

    return sanitize_for_json(coverage_matrix())


@router.get("/api/tax/dispositions")
def get_tax_dispositions_endpoint(lookback_days: int = 400):
    """Dated dispositions the app can prove, from the journal and stated causes (4.7).

    `no_data` is a statement about the RECORD, not about the portfolio. An
    unclassified position decrease is excluded on purpose: it might be a sale, a
    transfer, a fee or a corporate action.
    """
    from tools.tax_policy import scan_dispositions

    return sanitize_for_json(scan_dispositions(lookback_days=lookback_days))


@router.get("/api/tax/precheck-rebuy")
def precheck_rebuy_endpoint(symbol: str, account: str, proposed_date: str = None):
    """Whether a proposed buy clears the wash-sale / superficial-loss window (4.7, 3.8 P2).

    Read `evidence_complete` with `allowed`. On an empty transaction record a
    pass means the engine failed to OBJECT, not that it cleared the trade.
    """
    from tools.tax_policy import precheck_rebuy

    return sanitize_for_json(precheck_rebuy(symbol, account, proposed_date))


@router.get("/api/portfolio/attribution")
def get_attribution_endpoint(window_days: int = 365):
    """Benchmark-relative, flow-adjusted portfolio attribution (4.10).

    Read `status` before any number. Four of its five values are refusals with a
    named unblocker: `no_history`, `insufficient_coverage` (read
    `coverage.coverage_pct`, never `coverage.span_days`), `flows_incomplete` and
    `flow_date_unvalued`. Only `measured` carries `twr_pct` and `alpha_pct`.
    """
    from tools.attribution import get_attribution_report

    return sanitize_for_json(get_attribution_report(window_days=window_days))


@router.get("/api/portfolio/coverage")
def get_history_coverage_endpoint(window_days: int = 365):
    """Rows-out-of-days for the valuation series — the figure the 365-day gate misses.

    `goal_projection._history_span_days` measures the distance between two
    endpoints and is blind to holes between them. This reports both, side by
    side, so the difference is visible rather than inferred.
    """
    from tools.attribution import _load_history, coverage

    series = _load_history()
    return sanitize_for_json(coverage([d for d, _ in series], window_days=window_days))


@router.get("/api/portfolio/rate-sensitivity")
def get_rate_sensitivity_endpoint():
    """Fixed-income duration, convexity and the ±100bp shock table (4.8).

    Read `status` before any number. `no_fixed_income` means every holding was
    classified and none is a bond — a measured zero. `undetermined` means some
    holding could NOT be classified, so the zero is not measured and must not be
    read as one. `yields_missing` means bonds were found and no yield is on file,
    so duration is withheld rather than guessed from the curve.
    """
    from tools.bond_analytics import portfolio_rate_sensitivity

    return sanitize_for_json(portfolio_rate_sensitivity())


@router.get("/api/fixed-income/shock-table")
def get_shock_table_endpoint(coupon_pct: float, ytm_pct: float, years: float,
                             face: float = 100.0, frequency: int = 2):
    """Price, duration, convexity and a parallel-shift shock table for ONE bond (4.8).

    Rates are in PERCENT here (4.5 for 4.5%) because that is what a user types;
    the engine works in decimals. Every shock row carries the exact reprice
    alongside the duration-only and duration+convexity estimates, so the reader
    can see how much of the move the linear number misses.
    """
    from tools.bond_analytics import shock_table

    return sanitize_for_json(shock_table(
        coupon_rate=coupon_pct / 100.0,
        ytm=ytm_pct / 100.0,
        years=years,
        face=face,
        frequency=frequency,
    ))


@router.get("/api/fixed-income/ladder-sensitivity")
def get_ladder_sensitivity_endpoint(amount: float = 100000.0,
                                    investment_type: str = "GIC",
                                    currency: str = "CAD"):
    """Duration and convexity of the 5-year ladder `construct_bond_ladder` builds (4.8).

    `marked_to_market` is the field to read alongside the numbers: a
    non-redeemable GIC has no secondary market, so its shock rows are the
    opportunity cost of being locked in, not a loss anyone can realise.
    """
    from tools.bond_analytics import ladder_rate_sensitivity

    return sanitize_for_json(ladder_rate_sensitivity(amount, investment_type, currency))


@router.get("/api/catalysts/scoreboard")
def get_catalyst_scoreboard_endpoint():
    """Resolution outcomes for recorded catalyst predictions (1.3).

    Read `overall.reportable` before `overall.hit_rate`: below 20 scored calls the
    rate is `null` on purpose, and the counts beside it are still real. The
    `by_confidence` block is what the item exists for — it is the only evidence
    that can justify or move the extractor's authored 0.5 and 0.8 thresholds.
    """
    from tools.catalyst_resolution import scoreboard

    return sanitize_for_json(scoreboard())


@router.get("/api/portfolio/classification-options")
def get_classification_options_endpoint():
    """The causes a human may assign to an observed change, with what each means.

    Served rather than hardcoded in the template so the entry screen and the
    engine can never drift apart — a cause the UI offers but `classify_change`
    rejects would be a dead button, and one the UI omits is a cause nobody can
    give.
    """
    from tools.portfolio_classification import CAUSES, EXTERNAL_FLOW_CAUSES

    return sanitize_for_json({
        "causes": [
            {"value": key, "label": label, "description": description,
             "is_external_flow": key in EXTERNAL_FLOW_CAUSES}
            for key, (label, description) in CAUSES.items()
        ],
    })


@router.post("/api/portfolio/classify")
async def classify_change_endpoint(payload: dict = Body(...)):
    """Record a human's stated cause for ONE observed position change (4.10a).

    This is the only writer for that store, and the store has no other author:
    nothing in the codebase infers a cause. Send the change exactly as the
    reconciliation endpoint returned it — the identifying fields AND the share
    values, because the values are fingerprinted so a later snapshot rewrite
    cannot silently re-point an old answer at new numbers.

    `cause: "unclassified"` retracts a previous statement, and is appended rather
    than deleted so the ledger still shows the change of mind.

    **`amount_base` is optional and is what 4.10 is waiting for.** This store
    records quantities — shares for a security, currency units for cash — and a
    time-weighted return needs the flow in money, in base currency, on its own
    date. Send it with an external inflow or outflow and the attribution engine
    can remove the flow; omit it and the flow is reported UNPRICED and the return
    is withheld. The sign is derived from the cause, so it does not matter whether
    a withdrawal is entered positive or negative.
    """
    from tools.portfolio_classification import classify_change

    change = payload.get("change")
    if not isinstance(change, dict):
        return JSONResponse({"ok": False, "error": "missing `change` object"},
                            status_code=400)

    result = classify_change(
        change,
        cause=payload.get("cause", ""),
        note=payload.get("note", ""),
        classified_by=payload.get("classified_by", "user"),
        amount_base=payload.get("amount_base"),
        base_currency=payload.get("base_currency", ""),
    )
    if not result.get("ok"):
        return JSONResponse(sanitize_for_json(result), status_code=400)
    return sanitize_for_json(result)


@router.get("/api/portfolio/classification-pending")
def get_classification_pending_endpoint(consumer: str = "4.10"):
    """The changes `consumer` is actually blocked on, and why.

    Deliberately demand-driven. 4.10a is not a data-entry chore: a delta nobody
    is waiting on should stay unclassified forever, so this asks about a change
    only when something downstream cannot proceed without it.
    """
    from tools.portfolio_classification import pending_for
    from tools.portfolio_reconciliation import get_reconciliation

    recon = get_reconciliation(limit=10_000)
    if recon.get("status") != "ready":
        return sanitize_for_json({
            "consumer": consumer,
            "blocked": False,
            "pending_count": 0,
            "pending": [],
            "status": recon.get("status"),
            "note": recon.get("note"),
        })
    return sanitize_for_json(pending_for(consumer, recon.get("changes") or []))


@router.get("/api/portfolio/event-radar")
def get_event_radar_endpoint():
    """Returns the merged holdings event radar (earnings, ex-dividend, FOMC dates)."""
    from tools.event_radar import build_event_radar_cached
    res = build_event_radar_cached()
    return sanitize_for_json(res)


@router.get("/api/portfolio/fund-flows")
def get_fund_flows_endpoint(symbol: str = None):
    """Returns flow series for ETF funds held by the active profile."""
    from tools.fund_flows import collect_active_profile_fund_universe, get_flow_series
    if symbol:
        res = get_flow_series(symbol)
        return sanitize_for_json(res)

    universe = collect_active_profile_fund_universe()
    funds = universe.get("funds", [])
    series_map = {}
    for f in funds:
        series_map[f] = get_flow_series(f)
    return sanitize_for_json({
        "universe": universe,
        "fund_series": series_map,
    })
