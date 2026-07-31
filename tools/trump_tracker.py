import html
import re
from datetime import UTC, datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import defusedxml.ElementTree as ET
import requests

from tools.cache import cached
from tools.exception_logger import log_exceptions


@cached(key_func=lambda days=15, max_posts=30: f"trump_yaps:{days}:{max_posts}")
@log_exceptions()
def get_latest_trump_posts(days: int = 15, max_posts: int = 30) -> dict[str, Any]:
    """
    Fetch the latest raw social media posts/statements from Donald Trump's Truth Social account.
    Returns posts within the last `days` window, up to `max_posts` entries.
    """
    url = "https://trumpstruth.org/feed"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return {"error": f"Failed to fetch feed, status code: {resp.status_code}"}

        if not resp.content:
            return {"error": "Received empty response from feed source."}

        # Parse XML
        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError as pe:
            return {"error": f"Failed to parse XML feed: {str(pe)}"}

        items = root.findall(".//item")
        if not items:
            return {
                "source": "Truth Social (via trumpstruth.org archive)",
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "posts": [],
                "note": "No posts found in feed."
            }

        posts = []
        now_utc = datetime.now(UTC)

        for item in items:
            # Stop if we already hit the maximum requested posts
            if len(posts) >= max_posts:
                break

            pub_date_el = item.find("pubDate")
            pub_date = pub_date_el.text if pub_date_el is not None and pub_date_el.text else ""

            # Parse publish date to check timeframe
            if pub_date:
                try:
                    dt = parsedate_to_datetime(pub_date)
                    delta = now_utc - dt
                    if delta.days > days:
                        # Skip this post if it is older than requested days limit
                        continue
                except Exception:
                    # If date parsing fails, we default to including it
                    pass

            title_el = item.find("title")
            desc_el = item.find("description")
            link_el = item.find("link")

            title = title_el.text if title_el is not None and title_el.text else ""
            desc = desc_el.text if desc_el is not None and desc_el.text else ""
            link = link_el.text if link_el is not None and link_el.text else ""

            # Extract clean text from HTML in description
            clean_text = re.sub(r'<[^<]+?>', '', desc).strip()
            clean_text = html.unescape(clean_text)

            if not clean_text and title:
                clean_text = html.unescape(title.strip())

            # Clean up "[No Title]" placeholder
            if clean_text.startswith("[No Title]"):
                clean_text = re.sub(r'^\[No Title\]\s*-\s*Post\s*from\s*[A-Za-z0-9,\s]+', '', clean_text).strip()

            original_url_el = item.find("{https://truthsocial.com/ns}originalUrl")
            original_url = original_url_el.text if original_url_el is not None and original_url_el.text else link

            # If empty text but we have an image/link
            if not clean_text:
                clean_text = f"[Status Update/Media without text. URL: {original_url}]"

            posts.append({
                "text": clean_text,
                "pub_date": pub_date,
                "url": original_url
            })

        return {
            "source": "Truth Social (via trumpstruth.org archive)",
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "posts": posts
        }
    except Exception as e:
        return {"error": f"Failed to retrieve Trump posts: {str(e)}"}


if __name__ == "__main__":
    import pprint
    pprint.pprint(get_latest_trump_posts(days=15, max_posts=3))
