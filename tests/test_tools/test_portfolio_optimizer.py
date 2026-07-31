"""Constrained optimizer + drift-band rebalancing tests (Advisor Roadmap 4.4) — offline.

Two layers:

  - `optimize_weights` is pure math and tested unmocked (deterministic diagonal
    covariance problems with known closed-form answers).
  - The orchestrators have their collaborators mocked at the seams:
    `_decision_context` (holdings), `_optimizer_returns` (price history),
    `_ips_constraints` (the user's stated caps), `_symbol_sector_profile`
    (fund classification + sector vector), and `_playbook` (the stored
    rebalance_drift_pct). What's exercised here is 4.4's own composition:
    universe construction, restricted/held-constant handling, the nothing-
    invented refusals (no band, no target, no tax rate), and the tax-exposure
    math on the sells.
"""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import tools.ips_precheck as ips
import tools.portfolio_optimizer as po

# ---------------------------------------------------------------------------
# Pure core: optimize_weights on problems with known answers
# ---------------------------------------------------------------------------

def _diag(vols):
    return np.diag(np.array(vols, dtype=float) ** 2)


def test_min_vol_is_inverse_variance_on_diagonal():
    symbols = ["A", "B", "C"]
    mu = np.array([0.08, 0.08, 0.08])
    cov = _diag([0.10, 0.20, 0.30])
    out = po.optimize_weights(mu, cov, symbols, "min_vol")
    assert out["available"]
    # Closed form: w ∝ 1/var → (0.7347, 0.1837, 0.0816)
    assert out["weights"]["A"] == pytest.approx(0.7347, abs=0.02)
    assert out["weights"]["B"] == pytest.approx(0.1837, abs=0.02)
    assert out["weights"]["C"] == pytest.approx(0.0816, abs=0.02)


def test_position_cap_binds_min_vol():
    symbols = ["A", "B"]
    mu = np.array([0.08, 0.08])
    cov = _diag([0.10, 0.30])  # unconstrained min-vol would be ~90% A
    out = po.optimize_weights(mu, cov, symbols, "min_vol", max_weights={"A": 0.5})
    assert out["available"]
    assert out["weights"]["A"] == pytest.approx(0.5, abs=0.01)
    assert "A" in out["binding"]["position_caps"]


def test_max_sharpe_tilts_toward_higher_mean():
    symbols = ["A", "B", "C"]
    mu = np.array([0.05, 0.10, 0.15])
    cov = _diag([0.20, 0.20, 0.20])  # equal vol, zero correlation → w ∝ mu
    out = po.optimize_weights(mu, cov, symbols, "max_sharpe")
    assert out["available"]
    assert out["weights"]["A"] == pytest.approx(1 / 6, abs=0.03)
    assert out["weights"]["B"] == pytest.approx(2 / 6, abs=0.03)
    assert out["weights"]["C"] == pytest.approx(3 / 6, abs=0.03)
    assert out["sharpe"] == pytest.approx(0.9357, abs=0.02)


def test_max_sharpe_respects_cap():
    symbols = ["A", "B", "C"]
    mu = np.array([0.05, 0.10, 0.15])
    cov = _diag([0.20, 0.20, 0.20])
    out = po.optimize_weights(mu, cov, symbols, "max_sharpe", max_weights={"C": 0.30})
    assert out["available"]
    assert out["weights"]["C"] <= 0.30 + 1e-6


def test_max_sharpe_survives_a_min_vol_corner_that_drops_a_name():
    """Regression: `max_sharpe` crashed on any book where min-vol zeroes a name.

    Found on a live book, where it surfaced as a `solver error` whose entire
    message was a ticker — which is what made it read as bad data rather than
    as a bug.

    The mechanism: max_sharpe re-solves from the min-vol corner as a second
    start, and read that corner's weights with `alt["weights"][s]`. But the
    returned dict is built with `if wi > 0`, so a name the corner zeroed out is
    ABSENT — and the direct index raised KeyError, which the broad handler around
    the solve turned into a "solver error" naming the missing symbol.

    **Why 25 existing tests never reached it:** every one of them builds
    covariance with `_diag(...)`. On a diagonal covariance with positive means,
    min-vol is inverse-variance and gives EVERY asset a strictly positive weight,
    so no corner solution is possible and the missing key can never occur. It
    takes a correlated matrix — i.e. a real book — to produce one, which is why
    this shipped and stayed broken.
    """
    symbols = ["A", "B", "C"]
    mu = np.array([0.06, 0.07, 0.05])
    # C is both the highest-vol name and heavily correlated with the other two,
    # so min-vol drops it entirely.
    cov = np.array([
        [0.0100, 0.0010, 0.0090],
        [0.0010, 0.0120, 0.0095],
        [0.0090, 0.0095, 0.2000],
    ])

    corner = po.optimize_weights(mu, cov, symbols, "min_vol")
    assert corner["available"]
    assert "C" not in corner["weights"], "fixture no longer produces a corner solution"

    out = po.optimize_weights(mu, cov, symbols, "max_sharpe")
    assert out["available"], out.get("reason")
    # The failure mode was a reason string containing a ticker.
    assert "solver error" not in str(out.get("reason", ""))
    assert sum(out["weights"].values()) == pytest.approx(1.0, abs=1e-6)


def test_a_zeroed_name_is_absent_from_weights_and_that_is_the_contract():
    """Pins the property the bug depended on, so a caller cannot assume presence."""
    symbols = ["A", "B", "C"]
    mu = np.array([0.06, 0.07, 0.05])
    cov = np.array([
        [0.0100, 0.0010, 0.0090],
        [0.0010, 0.0120, 0.0095],
        [0.0090, 0.0095, 0.2000],
    ])
    weights = po.optimize_weights(mu, cov, symbols, "min_vol")["weights"]
    assert set(weights) < set(symbols), "a zero-weight name must be omitted, not zero-valued"
    assert all(w > 0 for w in weights.values())


def test_target_vol_binds_and_beats_min_vol_return():
    symbols = ["A", "B", "C"]
    mu = np.array([0.12, 0.08, 0.06])
    cov = _diag([0.10, 0.20, 0.30])
    mv = po.optimize_weights(mu, cov, symbols, "min_vol")
    assert mv["available"]
    out = po.optimize_weights(mu, cov, symbols, "target_vol", target_vol=0.09)
    assert out["available"]
    assert out["stats"]["volatility"] <= 0.09 + 1e-4
    assert out["stats"]["expected_return"] >= mv["stats"]["expected_return"] - 1e-6
    assert out["binding"]["target_vol"]


def test_target_vol_below_achievable_minimum_is_reported():
    symbols = ["A", "B", "C"]
    mu = np.array([0.12, 0.08, 0.06])
    cov = _diag([0.10, 0.20, 0.30])  # min-vol vol ≈ 8.6%
    out = po.optimize_weights(mu, cov, symbols, "target_vol", target_vol=0.05)
    assert not out["available"]
    assert "below the achievable minimum" in out["reason"]


def test_infeasible_caps_are_reported_not_garbage():
    symbols = ["A", "B"]
    mu = np.array([0.08, 0.08])
    cov = _diag([0.10, 0.20])
    out = po.optimize_weights(
        mu, cov, symbols, "min_vol", max_weights={"A": 0.30, "B": 0.30}
    )
    assert not out["available"]
    assert "infeasible" in out["reason"]


def test_sector_cap_binds_through_decomposition():
    symbols = ["A", "B", "C"]
    mu = np.array([0.10, 0.10, 0.08])
    cov = _diag([0.20, 0.20, 0.20])  # unconstrained min-vol: equal thirds
    exposure = {"A": {"Tech": 1.0}, "B": {"Tech": 1.0}, "C": {"Bonds": 1.0}}
    out = po.optimize_weights(
        mu, cov, symbols, "min_vol",
        sector_exposure=exposure, max_sector_weight=0.60,
    )
    assert out["available"]
    assert out["weights"]["C"] == pytest.approx(0.40, abs=0.01)  # Tech (A+B) capped at 60%
    assert "Tech" in out["binding"]["sectors"]


def test_sector_cap_that_cannot_reach_full_investment_says_so():
    """The cap applies to EVERY sector, so two sectors at 40% cannot fill 100%.

    That is genuine infeasibility, not solver noise — and it has to read as
    such, the same way jointly-infeasible position caps do.
    """
    symbols = ["A", "B", "C"]
    mu = np.array([0.10, 0.10, 0.08])
    cov = _diag([0.20, 0.20, 0.20])
    exposure = {"A": {"Tech": 1.0}, "B": {"Tech": 1.0}, "C": {"Bonds": 1.0}}
    out = po.optimize_weights(
        mu, cov, symbols, "min_vol",
        sector_exposure=exposure, max_sector_weight=0.40,
    )
    assert not out["available"]
    assert "jointly infeasible" in out["reason"]
    assert "80.0%" in out["reason"]  # 2 sectors x 40%


def test_unclassified_name_absorbs_the_sector_cap_remainder():
    """An Unknown-sector name is not capped, so it can fill what sectors can't."""
    symbols = ["A", "B", "C"]
    mu = np.array([0.10, 0.10, 0.08])
    cov = _diag([0.20, 0.20, 0.20])
    exposure = {"A": {"Tech": 1.0}, "B": {"Tech": 1.0}, "C": {"Unknown": 1.0}}
    out = po.optimize_weights(
        mu, cov, symbols, "min_vol",
        sector_exposure=exposure, max_sector_weight=0.40,
    )
    assert out["available"]
    assert out["weights"]["C"] == pytest.approx(0.60, abs=0.01)
    assert "C" in out["sector_unclassified"]


def test_unknown_sector_is_not_capped_and_is_reported():
    symbols = ["A", "B"]
    mu = np.array([0.08, 0.08])
    cov = _diag([0.20, 0.20])
    exposure = {"A": {"Unknown": 1.0}, "B": {"Tech": 1.0}}
    out = po.optimize_weights(
        mu, cov, symbols, "min_vol",
        sector_exposure=exposure, max_sector_weight=0.10,
    )
    assert out["available"]
    # B alone cannot exceed the cap; A is unclassified and unconstrained.
    assert out["weights"]["B"] <= 0.10 + 1e-4
    assert "A" in out["sector_unclassified"]


def test_bad_objective_is_reported():
    out = po.optimize_weights(np.array([0.1]), np.array([[0.04]]), ["A"], "yolo")
    assert not out["available"]


# ---------------------------------------------------------------------------
# Orchestration: optimize_portfolio (seams mocked)
# ---------------------------------------------------------------------------

def _returns_frame(symbols, n=260, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-01", periods=n)
    df = pd.DataFrame({s: rng.normal(0.0005, 0.012, n) for s in symbols}, index=idx)
    df.attrs["fx"] = {"base_currency": "USD", "converted": [], "unavailable": []}
    return df


def _ips(**over):
    base = {
        "enabled": True,
        "max_position_pct": None,
        "max_fund_position_pct": None,
        "max_sector_pct": None,
        "max_risk_per_trade_pct": None,
        "restricted_symbols": [],
    }
    base.update(over)
    return base


def _optimize_ctx():
    return {
        "total_value_base": 200_000.0,
        "base_currency": "USD",
        "as_of": "2026-07-25",
        "is_stale": False,
        "holdings": [
            {"symbol": "AAPL", "value_base": 60_000.0, "is_cash_or_pension": False},
            {"symbol": "MSFT", "value_base": 40_000.0, "is_cash_or_pension": False},
            {"symbol": "CASH", "value_base": 100_000.0, "is_cash_or_pension": True},
        ],
    }


def test_optimize_portfolio_happy_path_respects_stated_cap():
    with patch.object(po, "_decision_context", return_value=_optimize_ctx()), \
         patch.object(po, "_optimizer_returns",
                      return_value=(_returns_frame(["AAPL", "MSFT"]), ["AAPL", "MSFT"])), \
         patch.object(po, "_ips_constraints", return_value=_ips(max_position_pct=70.0)), \
         patch.object(po, "_symbol_sector_profile",
                      return_value=(False, {"Technology": 1.0})):
        out = po.optimize_portfolio(objective="min_vol")

    assert out["available"]
    assert sum(out["optimized_weights_pct"].values()) == pytest.approx(100.0, abs=0.1)
    assert all(w <= 70.0 + 1e-6 for w in out["optimized_weights_pct"].values())
    assert out["constraints_applied"]["max_position_pct"] == 70.0
    assert out["estimation"]["basis"] == "historical_mean"
    assert "not a forecast" in out["estimation"]["caveat"]
    assert set(out["universe"]) == {"AAPL", "MSFT"}  # cash excluded


def test_optimize_portfolio_restricted_name_is_held_constant_not_sold():
    with patch.object(po, "_decision_context", return_value=_optimize_ctx()), \
         patch.object(po, "_optimizer_returns",
                      return_value=(_returns_frame(["AAPL"]), ["AAPL"])), \
         patch.object(po, "_ips_constraints", return_value=_ips(restricted_symbols=["MSFT"])), \
         patch.object(po, "_symbol_sector_profile",
                      return_value=(False, {"Technology": 1.0})):
        out = po.optimize_portfolio(objective="min_vol")

    assert not out["available"]
    assert "restricted" in out["reason"]
    assert "MSFT" in out["restricted_held_constant"]


def test_optimize_portfolio_reports_context_failure_without_raising():
    with patch.object(po, "_decision_context", return_value={"error": "no portfolio file"}):
        out = po.optimize_portfolio(objective="min_vol")
    assert not out["available"]
    assert "no portfolio file" in out["reason"]


def test_optimize_portfolio_refuses_thin_history():
    thin = _returns_frame(["AAPL", "MSFT"], n=30)
    with patch.object(po, "_decision_context", return_value=_optimize_ctx()), \
         patch.object(po, "_optimizer_returns", return_value=(thin, ["AAPL", "MSFT"])), \
         patch.object(po, "_ips_constraints", return_value=_ips()), \
         patch.object(po, "_symbol_sector_profile",
                      return_value=(False, {"Technology": 1.0})):
        out = po.optimize_portfolio(objective="min_vol")
    assert not out["available"]
    assert "floor" in out["reason"]
    assert out["observations"] == 30


# ---------------------------------------------------------------------------
# Execution readiness
#
# An unbounded solve is a correct answer to "what is optimal with no limits" and
# a misleading one to "what should I hold" when nobody ever asked the user for
# limits: the weights come back concentrated exactly where the maths says, with
# no cap in sight and nothing saying the profile is missing one.
# ---------------------------------------------------------------------------

def _ack(*axes):
    return {"acknowledged_at": "2026-07-28T09:00:00", "axes": list(axes)}


def test_an_unbounded_solve_reports_itself_not_execution_ready():
    with patch.object(po, "_decision_context", return_value=_optimize_ctx()), \
         patch.object(po, "_optimizer_returns",
                      return_value=(_returns_frame(["AAPL", "MSFT"]), ["AAPL", "MSFT"])), \
         patch.object(po, "_ips_constraints", return_value=_ips()), \
         patch.object(po, "_symbol_sector_profile",
                      return_value=(False, {"Technology": 1.0})):
        out = po.optimize_portfolio(objective="min_vol")

    # It still solves and still returns the allocation — this reports, it does
    # not block. Refusing here would turn "unstated means unconstrained" into a
    # house default by another name.
    assert out["available"]
    assert out["optimized_weights_pct"]
    assert out["execution_ready"] is False
    assert "NOT EXECUTION-READY" in out["execution_readiness_note"]


def test_a_profile_that_chose_no_limits_gets_no_note():
    with patch.object(po, "_decision_context", return_value=_optimize_ctx()), \
         patch.object(po, "_optimizer_returns",
                      return_value=(_returns_frame(["AAPL", "MSFT"]), ["AAPL", "MSFT"])), \
         patch.object(po, "_ips_constraints",
                      return_value=_ips(unconstrained_ack=_ack(
                          "max_position_pct", "max_fund_position_pct",
                          "max_sector_pct", "max_risk_per_trade_pct"))), \
         patch.object(po, "_symbol_sector_profile",
                      return_value=(False, {"Technology": 1.0})):
        out = po.optimize_portfolio(objective="min_vol")

    assert out["execution_ready"] is True
    assert "execution_readiness_note" not in out


def test_the_optimizer_and_the_precheck_answer_from_the_same_seam():
    """One profile state must not produce two different verdicts about whether
    its proposals can be acted on."""
    constraints = _ips(max_position_pct=70.0)

    assert (po._execution_readiness(constraints)
            == ips.execution_readiness(constraints))


# ---------------------------------------------------------------------------
# Orchestration: check_rebalance_drift (seams mocked)
# ---------------------------------------------------------------------------

def _drift_ctx():
    return {
        "total_value_base": 100_000.0,
        "base_currency": "USD",
        "as_of": "2026-07-25",
        "is_stale": False,
        "holdings": [
            {"symbol": "AAPL", "account": "Brokerage", "value_base": 58_000.0,
             "purchase_price": 100.0, "current_price": 150.0, "is_cash_or_pension": False},
            {"symbol": "AAPL", "account": "TFSA", "value_base": 12_000.0,
             "purchase_price": 100.0, "current_price": 150.0, "is_cash_or_pension": False},
            {"symbol": "MSFT", "account": "Brokerage", "value_base": 30_000.0,
             "purchase_price": 300.0, "current_price": 320.0, "is_cash_or_pension": False},
        ],
    }


def test_drift_without_playbook_band_is_unavailable_and_names_2_8():
    with patch.object(po, "_playbook", return_value=None), \
         patch.object(po, "_decision_context", return_value=_drift_ctx()):
        out = po.check_rebalance_drift(target_allocation={"AAPL": 50, "MSFT": 50})

    assert not out["available"]
    assert "rebalance_drift_pct" in out["reason"]
    assert "2.8" in out["inert_dependency"]


def test_drift_without_target_is_unavailable():
    with patch.object(po, "_playbook", return_value={"rebalance_drift_pct": 5.0}), \
         patch.object(po, "_decision_context", return_value=_drift_ctx()):
        out = po.check_rebalance_drift()
    assert not out["available"]
    assert "never invented" in out["reason"]


def test_drift_breach_math_and_tax_exposure():
    # Current: AAPL 70%, MSFT 30%. Target 50/50 → sell 20k AAPL, buy 20k MSFT.
    with patch.object(po, "_playbook", return_value={"rebalance_drift_pct": 5.0}), \
         patch.object(po, "_decision_context", return_value=_drift_ctx()):
        out = po.check_rebalance_drift(target_allocation={"AAPL": 50, "MSFT": 50})

    assert out["available"]
    assert out["breached"]
    assert out["max_abs_drift_pct"] == pytest.approx(20.0)
    assert out["band_pct"] == 5.0
    assert out["target_basis"] == "explicit"

    trades = {t["symbol"]: t for t in out["trades_to_target"]}
    assert trades["AAPL"]["side"] == "SELL"
    assert trades["AAPL"]["amount_base"] == pytest.approx(20_000.0)
    assert trades["MSFT"]["side"] == "BUY"
    assert trades["MSFT"]["amount_base"] == pytest.approx(20_000.0)
    assert out["turnover"]["one_way_turnover_pct"] == pytest.approx(20.0)

    # Tax: only the taxable Brokerage row realizes gain; the TFSA row is
    # sheltered. Sale 20k at gain fraction (150-100)/150 = 1/3 → 6,666.67.
    tax = out["tax"]
    assert tax["taxable_realized_gain_base"] == pytest.approx(6_666.67, abs=0.01)
    assert tax["per_symbol"]["AAPL"] == pytest.approx(6_666.67, abs=0.01)
    assert tax["tax_bill"] is None
    assert "marginal" in tax["tax_bill_withheld_reason"]


def test_drift_within_band_has_no_trades():
    with patch.object(po, "_playbook", return_value={"rebalance_drift_pct": 25.0}), \
         patch.object(po, "_decision_context", return_value=_drift_ctx()):
        out = po.check_rebalance_drift(target_allocation={"AAPL": 75, "MSFT": 25})

    assert out["available"]
    assert not out["breached"]
    assert out["max_abs_drift_pct"] == pytest.approx(5.0)
    assert "Within band" in out["verdict"]
    assert "tax" not in out or out["tax"].get("taxable_realized_gain_base") == 0.0


def test_drift_target_from_optimizer_carries_held_constant_names():
    # Optimizer covers only AAPL/MSFT; restricted "IJR" must stay at current
    # weight instead of drifting to a full-sell target of 0.
    opt_result = {
        "available": True,
        "optimized_weights_pct": {"AAPL": 60.0, "MSFT": 40.0},
        "held_constant_pct": {"IJR": 10.0},
    }
    ctx = _drift_ctx()
    ctx["holdings"].append(
        {"symbol": "IJR", "account": "IRA", "value_base": 11_111.11,
         "purchase_price": 50.0, "current_price": 55.0, "is_cash_or_pension": False}
    )
    # tradeable total now 111,111.11; IJR current weight = 10%
    with patch.object(po, "_playbook", return_value={"rebalance_drift_pct": 5.0}), \
         patch.object(po, "_decision_context", return_value=ctx), \
         patch.object(po, "optimize_portfolio", return_value=opt_result):
        out = po.check_rebalance_drift(objective="min_vol")

    assert out["available"]
    assert out["target_basis"] == "optimizer:min_vol"
    # IJR held at 10%; optimizer weights scale over the remaining 90%.
    assert out["target_weights_pct"]["IJR"] == pytest.approx(10.0)
    assert out["target_weights_pct"]["AAPL"] == pytest.approx(54.0)
    assert sum(out["target_weights_pct"].values()) == pytest.approx(100.0, abs=0.1)


def test_drift_never_raises_on_context_error():
    with patch.object(po, "_playbook", return_value={"rebalance_drift_pct": 5.0}), \
         patch.object(po, "_decision_context", return_value={"error": "disk gone"}):
        out = po.check_rebalance_drift(target_allocation={"AAPL": 100})
    assert not out["available"]
    assert "disk gone" in out["reason"]
