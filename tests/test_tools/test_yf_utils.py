from unittest.mock import MagicMock, patch

import pytest

from tools.yf_utils import get_history_safe, get_info_safe, safe_yf_call


def test_safe_yf_call_success():
    mock_func = MagicMock(return_value="Success")
    res = safe_yf_call(mock_func)
    assert res == "Success"
    assert mock_func.call_count == 1

def test_safe_yf_call_retry_on_closed_file():
    # Fail twice with "closed file", succeed on third
    # Use ValueError to match the catch block
    mock_func = MagicMock(side_effect=[
        ValueError("I/O operation on closed file"),
        ValueError("closed file"),
        "Success"
    ])

    with patch("time.sleep") as mock_sleep:
        res = safe_yf_call(mock_func, max_retries=3, initial_delay=0.1)
        assert res == "Success"
        assert mock_func.call_count == 3
        assert mock_sleep.call_count == 2

def test_safe_yf_call_retry_on_401_crumb():
    # Fail with crumb error
    mock_func = MagicMock(side_effect=[
        ValueError("401 Unauthorized: Crumb not found"),
        "Success"
    ])

    with patch("time.sleep") as mock_sleep:
        res = safe_yf_call(mock_func, max_retries=2, initial_delay=0.1)
        assert res == "Success"
        assert mock_func.call_count == 2
        # Crumb error uses initial_delay * (attempt + 2)
        mock_sleep.assert_called_with(0.1 * 2)

def test_safe_yf_call_exhaust_retries():
    # Use an error that triggers the catch blocks BUT reaches the 'raise'
    # By setting max_retries to 1, it should raise immediately
    mock_func = MagicMock(side_effect=ValueError("closed file"))

    with pytest.raises(ValueError, match="closed file"):
        safe_yf_call(mock_func, max_retries=1)

@patch("yfinance.Ticker")
@patch("tools.yf_utils.safe_yf_call")
def test_wrappers(mock_safe_call, mock_ticker_class):
    mock_safe_call.return_value = "Result"

    get_history_safe("AAPL", period="1mo")
    mock_safe_call.assert_called()

    get_info_safe("AAPL")
    assert mock_safe_call.call_count == 2
