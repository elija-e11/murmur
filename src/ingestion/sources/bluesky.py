"""Bluesky social data source — searches public posts via the AT Protocol API."""

import logging
import time
from datetime import datetime, timezone, timedelta

import requests

from src.ingestion.sources.reddit import _keyword_sentiment

logger = logging.getLogger(__name__)

# Separate queries per symbol (OR operator is unreliable on Bluesky search)
SYMBOL_SEARCH_TERMS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "SOL": ["solana"],
    "AVAX": ["avalanche"],
    "LINK": ["chainlink"],
    "DOGE": ["dogecoin", "doge"],
    "ADA": ["cardano"],
    "DOT": ["polkadot"],
}


class BlueskySource:
    """Fetches crypto sentiment from Bluesky public post search."""

    BASE_URL = "https://public.api.bsky.app"

    def __init__(self):
        self.session = requests.Session()
        self._last_request = 0
        self._min_interval = 0.5  # Be polite with rate limits

    def _search_posts(self, query: str, limit: int = 100,
                      since: str | None = None) -> list[dict]:
        """Search public Bluesky posts."""
        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        params = {
            "q": query,
            "limit": min(limit, 100),
            "sort": "latest",
            "lang": "en",
        }
        if since:
            params["since"] = since

        url = f"{self.BASE_URL}/xrpc/app.bsky.feed.searchPosts"
        resp = self.session.get(url, params=params, timeout=15)
        self._last_request = time.time()
        if resp.status_code == 429:
            logger.warning("Bluesky rate limited, backing off 10s")
            time.sleep(10)
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
        since = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")

        seen_uris = set()
        sentiments = []
        engagement_weights = []
        total_likes = 0
        total_reposts = 0
        total_replies = 0
        api_errors = []

        for term in search_terms:
            try:
                posts = self._search_posts(term, limit=100, since=since)
                for post in posts:
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
