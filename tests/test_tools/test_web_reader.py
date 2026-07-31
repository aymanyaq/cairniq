"""SSRF-guard tests for tools/web_reader.py.

read_web_page fetches URLs that originate from untrusted page/news content,
so it must refuse to touch anything that isn't a public http(s) host — and it
must re-check every redirect hop, not just the first URL.
"""
from unittest.mock import MagicMock, patch

from tools.web_reader import _validate_url, read_web_page


def test_blocks_non_http_schemes():
    assert "Blocked URL scheme" in _validate_url("file:///etc/passwd")
    assert "Blocked URL scheme" in _validate_url("ftp://example.com/data")
    assert "Blocked URL scheme" in _validate_url("not-a-url")


def test_blocks_private_and_loopback_hosts():
    assert "non-public address" in _validate_url("http://127.0.0.1:8000/api")
    assert "non-public address" in _validate_url("http://localhost/admin")
    assert "non-public address" in _validate_url("http://192.168.1.1/")
    assert "non-public address" in _validate_url("http://10.0.0.5/")
    # Cloud metadata endpoint (link-local)
    assert "non-public address" in _validate_url("http://169.254.169.254/latest/meta-data")


def test_allows_public_hosts():
    with patch("tools.web_reader.socket.getaddrinfo") as mock_resolve:
        mock_resolve.return_value = [(2, 1, 6, "", ("93.184.216.34", 0))]
        assert _validate_url("https://example.com/article") is None


def test_read_web_page_refuses_blocked_url_without_fetching():
    with patch("tools.web_reader.requests.get") as mock_get:
        result = read_web_page("http://192.168.1.1/router")
        assert "non-public address" in result["error"]
        mock_get.assert_not_called()


def test_redirect_to_private_address_is_blocked():
    redirect = MagicMock()
    redirect.is_redirect = True
    redirect.is_permanent_redirect = False
    redirect.headers = {"Location": "http://127.0.0.1:8000/internal"}

    def resolve(host, port):
        if host == "evil.example.com":
            return [(2, 1, 6, "", ("93.184.216.34", 0))]
        return [(2, 1, 6, "", (host, 0))]  # IP-literal hosts resolve to themselves

    with patch("tools.web_reader.socket.getaddrinfo", side_effect=resolve), \
         patch("tools.web_reader.requests.get", return_value=redirect) as mock_get:
        result = read_web_page("https://evil.example.com/article")
        assert "non-public address" in result["error"]
        assert mock_get.call_count == 1  # the private hop was never fetched


def test_redirect_loop_is_bounded():
    redirect = MagicMock()
    redirect.is_redirect = True
    redirect.is_permanent_redirect = False
    redirect.headers = {"Location": "https://example.com/loop"}

    with patch("tools.web_reader.socket.getaddrinfo") as mock_resolve, \
         patch("tools.web_reader.requests.get", return_value=redirect):
        mock_resolve.return_value = [(2, 1, 6, "", ("93.184.216.34", 0))]
        result = read_web_page("https://example.com/loop")
        assert "Too many redirects" in result["error"]
