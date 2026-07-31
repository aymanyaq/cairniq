"""Offline-by-default network guard for the test suite.

The suite used to make ~520 live HTTPS calls to Yahoo Finance on every run: 53s of
the 105s wall time was a unit test sitting on a socket. None of it was deliberate —
it leaked in through incidental helpers (detect_sector_rotation inside
get_user_context, price lookups inside a summary render) that no one intended to
exercise. The cost was not only speed: a Yahoo hiccup, a plane, or a coffee-shop
captive portal turned unrelated assertions red, and the "unit" tests silently
depended on whatever the market happened to be doing that morning.

So the contract is: tests are offline unless they say otherwise. Any attempt to open
a real socket raises OfflineNetworkError, which names the test and tells you how to
fix it. Loopback stays open — TestClient, and any local fixture server, are not the
network we are guarding against.

Two escape hatches, in order of preference:

  1. Stub the call. Almost always correct — the test did not want live prices.
  2. @pytest.mark.allow_network — for tests whose subject IS the transport. These do
     not run in CI's default lane; see the `network` marker in pytest.ini.

The guard is installed ONCE at import and toggled by a module-level flag, rather than
patched per test: 1088 patch/unpatch cycles cost more than the check they perform.
"""

import socket

# noqa S104: these are hosts a test is allowed to CONNECT to, not an address
# anything binds. 0.0.0.0 resolves to the local machine on connect, so a fixture
# server that advertises itself that way must not trip the guard.
_ALLOWED_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "0.0.0.0", ""})  # noqa: S104

# Flipped by the autouse fixture in conftest for @pytest.mark.allow_network tests.
_enabled = True
_current_test = "<import time>"


class OfflineNetworkError(RuntimeError):
    """Raised when a test reaches for the real network."""


def _explain(target: str) -> OfflineNetworkError:
    return OfflineNetworkError(
        f"Blocked live network call to {target!r} from {_current_test}.\n"
        "The test suite runs offline. Either stub the call (usually what you want — "
        "the test probably did not mean to fetch live prices), or mark the test with "
        "@pytest.mark.allow_network if the transport itself is the subject."
    )


def _host_of(address) -> str:
    if isinstance(address, (tuple, list)) and address:
        return str(address[0])
    return str(address)


def _is_allowed(address) -> bool:
    return not _enabled or _host_of(address) in _ALLOWED_HOSTS


def set_enabled(enabled: bool) -> None:
    global _enabled
    _enabled = enabled


def set_current_test(nodeid: str) -> None:
    global _current_test
    _current_test = nodeid


def install() -> None:
    """Patch every transport the app can actually reach the internet through."""
    _orig_connect = socket.socket.connect
    _orig_create = socket.create_connection

    def guarded_connect(self, address):
        if not _is_allowed(address):
            raise _explain(_host_of(address))
        return _orig_connect(self, address)

    def guarded_create_connection(address, *args, **kwargs):
        if not _is_allowed(address):
            raise _explain(_host_of(address))
        return _orig_create(address, *args, **kwargs)

    socket.socket.connect = guarded_connect
    socket.create_connection = guarded_create_connection

    # yfinance talks through curl_cffi, which is libcurl in C — it never touches a
    # Python socket, so the patches above cannot see it. This is where the 512 calls
    # were coming from, and blocking it is the whole point of this module.
    try:
        from curl_cffi import curl as curl_cffi_curl
    except ImportError:  # pragma: no cover — curl_cffi ships with yfinance
        return

    _orig_perform = curl_cffi_curl.Curl.perform

    def guarded_perform(self, *args, **kwargs):
        if _enabled:
            raise _explain("curl_cffi (yfinance)")
        return _orig_perform(self, *args, **kwargs)

    curl_cffi_curl.Curl.perform = guarded_perform
