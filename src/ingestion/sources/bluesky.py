"""Bluesky social data source — searches posts via the AT Protocol API.

Uses authenticated access through bsky.social when credentials are provided,
falling back to the public API (which may be blocked from cloud IPs).
"""

import logging
import time
from datetime import datetime, timezone, timedelta

import requests

from src.ingestion.sources.reddit import _keyword_sentiment

logger = logging.getLogger(__name__)

# Separate queries per symbol (OR operator is unreliable on Bluesky search)
SYMBOL_SEARCH_TERMS = {
    "BTC": ["bitcoin", "#btc"],
    "ETH": ["ethereum", "#eth"],
    "SOL": ["solana", "#sol"],
    "AVAX": ["avalanche", "#avax"],
    "LINK": ["chainlink", "#link"],
    "DOGE": ["dogecoin", "#doge"],
    "ADA": ["cardano", "#ada"],
    "DOT": ["polkadot", "#dot"],
}


class BlueskySource:
    """Fetches crypto sentiment from Bluesky post search."""

    PDS_URL = "https://bsky.social"
    PUBLIC_URL = "https://public.api.bsky.app"

    def __init__(self, handle: str = "", app_password: str = ""):
        self.session = requests.Session()
        self._last_request = 0
        self._min_interval = 0.5  # Be polite with rate limits

        self._handle = handle
        self._app_password = app_password
        self._access_jwt = None
        self._refresh_jwt = None

        if handle and app_password:
            self._base_url = self.PDS_URL
            self._authenticate()
        else:
            self._base_url = self.PUBLIC_URL
            logger.info("Bluesky using public API (no credentials)")

    def _authenticate(self):
        """Create a session with Bluesky using handle + app password."""
        resp = requests.post(
            f"{self.PDS_URL}/xrpc/com.atproto.server.createSession",
            json={"identifier": self._handle, "password": self._app_password},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        self._access_jwt = data["accessJwt"]
        self._refresh_jwt = data["refreshJwt"]
        self.session.headers["Authorization"] = f"Bearer {self._access_jwt}"
        logger.info(f"Bluesky authenticated as {data.get('handle', self._handle)}")

    def _refresh_session(self):
        """Refresh an expired access token using the refresh token."""
        resp = requests.post(
            f"{self.PDS_URL}/xrpc/com.atproto.server.refreshSession",
            headers={"Authorization": f"Bearer {self._refresh_jwt}"},
            timeout=15,
        )
        if resp.status_code == 401:
            # Refresh token also expired — full re-auth
            logger.warning("Bluesky refresh token expired, re-authenticating")
            self._authenticate()
            return
        resp.raise_for_status()
        data = resp.json()
        self._access_jwt = data["accessJwt"]
        self._refresh_jwt = data["refreshJwt"]
        self.session.headers["Authorization"] = f"Bearer {self._access_jwt}"
        logger.debug("Bluesky session refreshed")

    def _search_posts(self, query: str, limit: int = 100) -> list[dict]:
        """Search Bluesky posts (sorted by latest, no server-side time filter).

        The `since` parameter exists in the lexicon but is rejected by the
        AppView with a 400. Callers must filter by time client-side.
        """
        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        params = {
            "q": query,
            "limit": min(limit, 100),
            "sort": "latest",
        }

        url = f"{self._base_url}/xrpc/app.bsky.feed.searchPosts"
        resp = self.session.get(url, params=params, timeout=15)
        self._last_request = time.time()

        # Handle auth expiry — refresh and retry once
        if resp.status_code == 401 and self._access_jwt:
            self._refresh_session()
            resp = self.session.get(url, params=params, timeout=15)
            self._last_request = time.time()

        if resp.status_code == 429:
            logger.warning("Bluesky rate limited, backing off 10s")
            time.sleep(10)
        if not resp.ok:
            body = resp.text[:200] if resp.text else ""
            logger.error(f"Bluesky search failed ({resp.status_code}) for q={query!r}: {body}")
        resp.raise_for_status()
        return resp.json().get("posts", [])

    def get_asset_metrics(self, symbol: str, lookback_hours: int = 4) -> dict:
        """Get mention count, sentiment, and engagement for a symbol.

        Returns:
            {
                "mention_count": int,
                "avg_sentiment": float,    # -1 to +1 (VADER compound)
                "total_likes": int,
                "total_reposts": int,
                "total_replies": int,
                "weighted_sentiment": float,  # engagement-weighted sentiment
            }
        """
        search_terms = SYMBOL_SEARCH_TERMS.get(symbol, [symbol.lower()])
        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

        seen_uris = set()
        sentiments = []
        engagement_weights = []
        total_likes = 0
        total_reposts = 0
        total_replies = 0
        api_errors = []

        for term in search_terms:
            try:
                posts = self._search_posts(term, limit=100)
                for post in posts:
                    # Client-side time filter (server doesn't support `since`)
                    indexed_at = post.get("indexedAt", "")
                    if indexed_at:
                        try:
                            post_time = datetime.fromisoformat(indexed_at.replace("Z", "+00:00"))
                            if post_time < cutoff:
                                continue
                        except ValueError:
                            pass
                    uri = post.get("uri", "")
                    if uri in seen_uris:
                        continue
                    seen_uris.add(uri)

                    text = post.get("record", {}).get("text", "")
                    if not text:
                        continue

                    likes = post.get("likeCount", 0)
                    reposts = post.get("repostCount", 0)
                    replies = post.get("replyCount", 0)

                    total_likes += likes
                    total_reposts += reposts
                    total_replies += replies

                    score = _keyword_sentiment(text)
                    sentiments.append(score)
                    # Weight by engagement (min 1 so every post counts)
                    engagement_weights.append(1 + likes + reposts * 2)

            except Exception as e:
                logger.error(f"Bluesky fetch failed for term '{term}': {e}")
                api_errors.append(e)

        # If every search term failed, propagate so health tracker sees the error
        if api_errors and len(api_errors) == len(search_terms):
            raise api_errors[0]

        mention_count = len(seen_uris)
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

        # Engagement-weighted sentiment gives more weight to popular posts
        if sentiments and engagement_weights:
            total_weight = sum(engagement_weights)
            weighted_sentiment = sum(
                s * w for s, w in zip(sentiments, engagement_weights)
            ) / total_weight
        else:
            weighted_sentiment = 0

        return {
            "mention_count": mention_count,
            "avg_sentiment": avg_sentiment,
            "total_likes": total_likes,
            "total_reposts": total_reposts,
            "total_replies": total_replies,
            "weighted_sentiment": weighted_sentiment,
        }
