import ipaddress
import os as _os
import socket
import ssl as _ssl
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from tools.exception_logger import log_exceptions

# Resolve system CA bundle (Homebrew OpenSSL) — certifi's bundle is incomplete on macOS
_CA_BUNDLE = _ssl.get_default_verify_paths().cafile
if not (_CA_BUNDLE and _os.path.isfile(_CA_BUNDLE)):
    _CA_BUNDLE = True  # fall back to requests default
from typing import Any

_MAX_REDIRECTS = 5


def _validate_url(url: str) -> str | None:
    """Return an error message if the URL must not be fetched, else None.

    SSRF guard: the agent feeds this tool URLs taken from untrusted page/news
    content, so a prompt-injected page could otherwise steer it at the local
    network (router admin, other LAN hosts). Blocks non-HTTP schemes and any
    host that resolves to a non-public address (loopback, RFC1918, link-local,
    CGN, reserved).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"Blocked URL scheme '{parsed.scheme or 'none'}' — only http/https are allowed."
    host = parsed.hostname
    if not host:
        return "Blocked URL: no hostname."
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        return f"Could not resolve host '{host}': {e}"
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            return f"Blocked URL: host '{host}' resolves to non-public address {ip}."
    return None


@log_exceptions()
def read_web_page(url: str) -> dict[str, Any]:
    """
    Fetches and extracts text content from a URL.
    Useful for reading news articles, blog posts, or analysis.
    """
    try:
        # "Stealth" headers to mimic a real Chrome browser visit from Google
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.google.com/",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "cross-site",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache"
        }

        # Use system OpenSSL CA bundle to avoid SSL verification failures.
        # Redirects are followed manually so every hop passes the SSRF guard —
        # a public URL 302-ing to a LAN address is the classic bypass.
        for _ in range(_MAX_REDIRECTS + 1):
            guard_error = _validate_url(url)
            if guard_error:
                return {"url": url, "error": guard_error}
            response = requests.get(url, headers=headers, timeout=10, verify=_CA_BUNDLE, allow_redirects=False)
            if response.is_redirect or response.is_permanent_redirect:
                url = urljoin(url, response.headers.get("Location", ""))
                continue
            break
        else:
            return {"url": url, "error": f"Too many redirects (>{_MAX_REDIRECTS})."}
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove scripts and styles
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Get text with separator to split lines
        text = soup.get_text(separator='\n')

        # Clean clean whitespace
        lines = (line.strip() for line in text.splitlines())

        # Filter out junk lines
        junk_phrases = [
            "Oops, something went wrong",
            "Skip to navigation",
            "Skip to main content",
            "Skip to right column",
            "Simply Wall St",
            "min read"
        ]

        cleaned_lines = []
        for line in lines:
            if not line: continue
            if any(phrase in line for phrase in junk_phrases): continue
            cleaned_lines.append(line)

        chunks = (phrase.strip() for line in cleaned_lines for phrase in line.split("  "))
        clean_text = '\n'.join(chunk for chunk in chunks if chunk)

        # Truncate if too long (max 8000 chars for context window safety)
        if len(clean_text) > 8000:
            clean_text = clean_text[:8000] + "... [Truncated]"

        title = soup.title.string if soup.title else "No Title"

        return {
            "url": url,
            "title": title,
            "content": clean_text,
        }

    except Exception as e:
        return {"url": url, "error": f"Failed to read page: {str(e)}"}

if __name__ == "__main__":
    # Test with the user's link
    url = "https://www.forbes.com/sites/digital-assets/2026/01/29/its-breaking-sudden-us-dollar-crisis-warning-predicted-to-spark-huge-bitcoin-price-boom-to-rival-gold/"
    print(read_web_page(url))
