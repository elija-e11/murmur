"""CoinGecko community data source — social stats and developer activity."""

import logging
import time

import requests

logger = logging.getLogger(__name__)

# Map our symbols to CoinGecko IDs
SYMBOL_TO_COINGECKO = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "AVAX": "avalanche-2",
    "LINK": "chainlink",
    "DOGE": "dogecoin",
    "ADA": "cardano",
    "DOT": "polkadot",
}


class CoinGeckoSource:
    """Fetches community and market data from CoinGecko free API."""

    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(self):
        self.session = requests.Session()
        self._last_request = 0
        self._min_interval = 1.5  # CoinGecko free tier: ~30 req/min

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        """Rate-limited GET request."""
        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        url = f"{self.BASE_URL}/{endpoint}"
        try:
            resp = self.session.get(url, params=params or {}, timeout=15)
            self._last_request = time.time()
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:
                logger.warning("CoinGecko rate limited, backing off 30s")
                time.sleep(30)
            logger.error(f"CoinGecko API error: {e}")
            return {}
        except Exception as e:
            logger.error(f"CoinGecko request failed: {e}")
            return {}

    def get_coin_data(self, symbol: str) -> dict:
        """Get community stats and market data for a coin.

        Returns:
            {
                "reddit_subscribers": int,
                "reddit_active_48h": int,
                "twitter_followers": int,
                "developer_score": float,
                "community_score": float,
                "market_cap": float,
                "price": float,
                "price_change_24h_pct": float,
                "total_volume": float,
            }
        """
        coin_id = SYMBOL_TO_COINGECKO.get(symbol)
        if not coin_id:
            return {}

        data = self._get(f"coins/{coin_id}", params={
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "true",
            "developer_data": "true",
            "sparkline": "false",
        })

        if not data:
            return {}

        community = data.get("community_data", {})
        market = data.get("market_data", {})

        return {
            "reddit_subscribers": community.get("reddit_subscribers") or 0,
            "reddit_active_48h": community.get("reddit_accounts_active_48h") or 0,
            "twitter_followers": community.get("twitter_followers") or 0,
            "developer_score": data.get("developer_score") or 0,
            "community_score": data.get("community_score") or 0,
            "market_cap": market.get("market_cap", {}).get("usd") or 0,
            "price": market.get("current_price", {}).get("usd") or 0,
            "price_change_24h_pct": market.get("price_change_percentage_24h") or 0,
            "total_volume": market.get("total_volume", {}).get("usd") or 0,
        }

    def get_trending(self) -> list[dict]:
        """Get trending coins on CoinGecko (no API key needed).

        Returns list of {"symbol": str, "name": str, "market_cap_rank": int}.
        """
        data = self._get("search/trending")
        coins = data.get("coins", [])
        return [
            {
                "symbol": c.get("item", {}).get("symbol", "").upper(),
                "name": c.get("item", {}).get("name", ""),
                "market_cap_rank": c.get("item", {}).get("market_cap_rank"),
            }
            for c in coins
        ]
