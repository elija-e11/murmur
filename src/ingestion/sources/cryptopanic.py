"""CryptoPanic news source — aggregated crypto news with community sentiment votes."""

import logging

import requests

logger = logging.getLogger(__name__)

# Map our symbols to CryptoPanic currency codes
SYMBOL_TO_CURRENCY = {
    "BTC": "BTC",
    "ETH": "ETH",
    "SOL": "SOL",
    "AVAX": "AVAX",
    "LINK": "LINK",
    "DOGE": "DOGE",
    "ADA": "ADA",
    "DOT": "DOT",
}


class CryptoPanicSource:
    """Fetches crypto news with sentiment from CryptoPanic API."""

    BASE_URL = "https://cryptopanic.com/api/developer/v2"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()

    def get_news(self, symbol: str | None = None, limit: int = 20) -> list[dict]:
        """Fetch recent news, optionally filtered by symbol.

        Returns list of news items with votes.
        """
        params = {
            "auth_token": self.api_key,
            "kind": "news",
            "public": "true",
        }
        if symbol and symbol in SYMBOL_TO_CURRENCY:
            params["currencies"] = SYMBOL_TO_CURRENCY[symbol]

        try:
            resp = self.session.get(f"{self.BASE_URL}/posts/", params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])[:limit]
            return results
        except Exception as e:
            logger.error(f"CryptoPanic API error: {e}")
            return []

    def get_asset_sentiment(self, symbol: str) -> dict:
        """Compute sentiment metrics from recent news for a symbol.

        Returns:
            {
                "news_count": int,
                "sentiment_score": float,  # -1 to +1
                "bullish_count": int,
                "bearish_count": int,
            }
        """
        news = self.get_news(symbol, limit=30)

        if not news:
            return {
                "news_count": 0,
                "sentiment_score": 0,
                "bullish_count": 0,
                "bearish_count": 0,
            }

        bullish = 0
        bearish = 0

        for item in news:
            votes = item.get("votes", {})
            positive = votes.get("positive", 0)
            negative = votes.get("negative", 0)
            important = votes.get("important", 0)

            # Items with more positive votes are bullish
            if positive > negative:
                bullish += 1
            elif negative > positive:
                bearish += 1

            # "important" votes with positive context = extra bullish weight
            if important > 0 and positive > negative:
                bullish += 1

        total = bullish + bearish
        if total > 0:
            sentiment = (bullish - bearish) / total  # -1 to +1
        else:
            sentiment = 0

        return {
            "news_count": len(news),
            "sentiment_score": sentiment,
            "bullish_count": bullish,
            "bearish_count": bearish,
        }
