from datetime import UTC, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from tools.trump_tracker import get_latest_trump_posts

SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:truth="https://truthsocial.com/ns">
    <channel>
        <title>Trump's Truth - Latest Posts</title>
        <item>
            <title><![CDATA[Recent post inside 15 days]]></title>
            <link>https://trumpstruth.org/statuses/38946</link>
            <description><![CDATA[<p>Recent post inside 15 days</p>]]></description>
            <pubDate>Sat, 30 May 2026 22:50:16 +0000</pubDate>
            <truth:originalUrl>https://truthsocial.com/@realDonaldTrump/116665969343279897</truth:originalUrl>
        </item>
        <item>
            <title><![CDATA[Old post outside 15 days]]></title>
            <link>https://trumpstruth.org/statuses/38950</link>
            <description><![CDATA[<p>Old post outside 15 days</p>]]></description>
            <!-- 29 days before May 31, 2026 -->
            <pubDate>Sat, 02 May 2026 02:57:51 +0000</pubDate>
            <truth:originalUrl>https://truthsocial.com/@realDonaldTrump/116666942823439731</truth:originalUrl>
        </item>
    </channel>
</rss>
"""


class MockDatetime:
    @classmethod
    def now(cls, tz=None):
        # Mock now as May 31, 2026
        return datetime(2026, 5, 31, 12, 0, 0, tzinfo=UTC)


@patch("tools.trump_tracker.requests.get")
@patch("tools.trump_tracker.datetime", MockDatetime)
def test_get_latest_trump_posts_success(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = SAMPLE_XML.encode("utf-8")
    mock_get.return_value = mock_response

    # Clear cache for testing
    from tools.cache import _cache
    _cache.clear()

    # Query with 15 days limit
    result = get_latest_trump_posts(days=15, max_posts=30)
    assert "error" not in result
    assert result["source"] == "Truth Social (via trumpstruth.org archive)"

    # Only the first post (May 30) should be included; the second post (May 2) is 29 days old and should be filtered out
    assert len(result["posts"]) == 1

    p1 = result["posts"][0]
    assert p1["text"] == "Recent post inside 15 days"
    assert p1["pub_date"] == "Sat, 30 May 2026 22:50:16 +0000"
    assert p1["url"] == "https://truthsocial.com/@realDonaldTrump/116665969343279897"


@patch("tools.trump_tracker.requests.get")
@patch("tools.trump_tracker.datetime", MockDatetime)
def test_get_latest_trump_posts_large_window(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = SAMPLE_XML.encode("utf-8")
    mock_get.return_value = mock_response

    # Clear cache
    from tools.cache import _cache
    _cache.clear()

    # Query with 40 days limit - should include both posts
    result = get_latest_trump_posts(days=40, max_posts=30)
    assert "error" not in result
    assert len(result["posts"]) == 2


@patch("tools.trump_tracker.requests.get")
@patch("tools.trump_tracker.datetime", MockDatetime)
def test_get_latest_trump_posts_max_posts_limit(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = SAMPLE_XML.encode("utf-8")
    mock_get.return_value = mock_response

    # Clear cache
    from tools.cache import _cache
    _cache.clear()

    # Query with 40 days limit but max_posts=1 - should return only 1 post
    result = get_latest_trump_posts(days=40, max_posts=1)
    assert "error" not in result
    assert len(result["posts"]) == 1


@patch("tools.trump_tracker.requests.get")
@patch("tools.trump_tracker.datetime", MockDatetime)
def test_get_latest_trump_posts_http_error(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_get.return_value = mock_response

    # Clear cache
    from tools.cache import _cache
    _cache.clear()

    result = get_latest_trump_posts(days=15, max_posts=30)
    assert "error" in result
    assert "status code: 500" in result["error"]


@patch("tools.trump_tracker.requests.get")
@patch("tools.trump_tracker.datetime", MockDatetime)
def test_get_latest_trump_posts_invalid_xml(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"invalid xml content"
    mock_get.return_value = mock_response

    # Clear cache
    from tools.cache import _cache
    _cache.clear()

    result = get_latest_trump_posts(days=15, max_posts=30)
    assert "error" in result
    assert "Failed to parse XML feed" in result["error"]
