"""Regression tests for advisor-calibration hit scoring in tools/memory.py.

Guards the fix for the defect where HOLD recommendations were scored with SELL
semantics: a "hit" was recorded only when the held name *underperformed* SPY, so
every correct "keep holding a market-beater" call was logged as a miss and the
corrupted hit-rate was injected into every subsequent prompt as the track record.
"""

from tools.memory import _is_long_bias_action


def test_hold_is_long_biased_like_buy():
    """HOLD is a keep call — it must score on the long side (hit == beat SPY),
    the same as BUY/ADD, not the sell side."""
    assert _is_long_bias_action("HOLD") is True
    assert _is_long_bias_action("BUY") is True
    assert _is_long_bias_action("ADD") is True


def test_sell_and_trim_are_reduce_biased():
    assert _is_long_bias_action("SELL") is False
    assert _is_long_bias_action("TRIM") is False


def test_action_classification_is_case_and_whitespace_insensitive():
    assert _is_long_bias_action(" hold ") is True
    assert _is_long_bias_action("Buy") is True


def test_unknown_or_empty_action_defaults_to_reduce_semantics():
    """Parity with the prior behaviour: anything not explicitly long-biased falls
    to the reduce branch (hit == alpha < 0) rather than raising."""
    assert _is_long_bias_action("") is False
    assert _is_long_bias_action(None) is False  # type: ignore[arg-type]
    assert _is_long_bias_action("WATCH") is False


def test_hit_semantics_match_classification():
    """The rule the read sites apply: long-bias hits on alpha>0, reduce hits on
    alpha<0. A HOLD on a name that beat SPY (alpha>0) is now a hit, not a miss."""
    def is_hit(action: str, alpha: float) -> bool:
        long_bias = _is_long_bias_action(action)
        return (alpha > 0) if long_bias else (alpha < 0)

    # HOLD on a market-beater: previously (SELL semantics) this was a miss.
    assert is_hit("HOLD", alpha=3.0) is True
    assert is_hit("HOLD", alpha=-3.0) is False
    # SELL that dropped vs SPY is still a hit.
    assert is_hit("SELL", alpha=-3.0) is True
    assert is_hit("SELL", alpha=3.0) is False
