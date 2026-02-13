"""Social data aggregator — combines Reddit, CryptoPanic, Fear & Greed, and CoinGecko
into unified social metrics matching the format the sentiment analyzer expects."""

import logging
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def product_to_symbol(product_id: str) -> str:
    """Convert Coinbase product ID (e.g., 'BTC-USD') to symbol ('BTC')."""
    return product_id.split("-")[0]


class SocialAggregator:
    """Combines multiple free data sources into unified social metrics.

    Output format per asset (matches the social_data DB schema):
        {
            "asset": str,
            "timestamp": int,
            "galaxy_score": float | None,     # composite score (0-100) computed from sources
            "alt_rank": None,                  # not available without LunarCrush
            "social_volume": float,            # Reddit mentions + news count
            "social_dominance": float,         # asset mentions / total mentions
            "sentiment": float,                # 0-5 scale (matches what sentiment analyzer expects)
            "market_cap": float,
            "price": float,
            "raw_json": str,
        }
    """

    def __init__(self, config: dict):
        secrets = config.get("secrets", {})
        self.sources_available = {}
        self._init_sources(secrets)

    def _init_sources(self, secrets: dict):
        """Initialize available sources. Missing credentials = source skipped."""
        # Reddit
        reddit_id = secrets.get("reddit_client_id", "")
        reddit_secret = secrets.get("reddit_client_secret", "")
        if reddit_id and reddit_secret:
            try:
                from src.ingestion.sources.reddit import RedditSource
                self.sources_available["reddit"] = RedditSource(
                    client_id=reddit_id,
                    client_secret=reddit_secret,
                    user_agent=secrets.get("reddit_user_agent", "murmur-bot/1.0"),
                )
                logger.info("Reddit source initialized")
            except Exception as e:
                logger.warning(f"Reddit source failed to init: {e}")
        else:
            logger.info("Reddit source skipped (no credentials)")

        # CryptoPanic
        cp_key = secrets.get("cryptopanic_api_key", "")
        if cp_key:
            try:
                from src.ingestion.sources.cryptopanic import CryptoPanicSource
                self.sources_available["cryptopanic"] = CryptoPanicSource(api_key=cp_key)
                logger.info("CryptoPanic source initialized")
            except Exception as e:
                logger.warning(f"CryptoPanic source failed to init: {e}")
        else:
            logger.info("CryptoPanic source skipped (no API key)")

        # Fear & Greed — always available (no auth needed)
        try:
            from src.ingestion.sources.fear_greed import FearGreedSource
            self.sources_available["fear_greed"] = FearGreedSource()
            logger.info("Fear & Greed source initialized")
        except Exception as e:
            logger.warning(f"Fear & Greed source failed to init: {e}")

        # CoinGecko — always available (no auth needed)
        try:
            from src.ingestion.sources.coingecko import CoinGeckoSource
            self.sources_available["coingecko"] = CoinGeckoSource()
            logger.info("CoinGecko source initialized")
        except Exception as e:
            logger.warning(f"CoinGecko source failed to init: {e}")

        if not self.sources_available:
            logger.warning("No social data sources available! Add API keys to config/.env")

    def _fetch_reddit(self, symbol: str) -> dict:
        """Fetch Reddit metrics for a symbol."""
        reddit = self.sources_available.get("reddit")
        if not reddit:
            return {}
        try:
            return reddit.get_asset_metrics(symbol)
        except Exception as e:
            logger.error(f"Reddit fetch failed for {symbol}: {e}")
            return {}

    def _fetch_cryptopanic(self, symbol: str) -> dict:
        """Fetch CryptoPanic sentiment for a symbol."""
        cp = self.sources_available.get("cryptopanic")
        if not cp:
            return {}
        try:
            return cp.get_asset_sentiment(symbol)
        except Exception as e:
            logger.error(f"CryptoPanic fetch failed for {symbol}: {e}")
            return {}

    def _fetch_fear_greed(self) -> dict:
        """Fetch current Fear & Greed index."""
        fg = self.sources_available.get("fear_greed")
        if not fg:
            return {}
        try:
            return fg.get_current()
        except Exception as e:
            logger.error(f"Fear & Greed fetch failed: {e}")
            return {}

    def _fetch_coingecko(self, symbol: str) -> dict:
        """Fetch CoinGecko community data for a symbol."""
        cg = self.sources_available.get("coingecko")
        if not cg:
            return {}
        try:
            return cg.get_coin_data(symbol)
        except Exception as e:
            logger.error(f"CoinGecko fetch failed for {symbol}: {e}")
            return {}

    def _compute_composite_score(self, reddit: dict, cryptopanic: dict,
                                  fear_greed: dict, coingecko: dict) -> float | None:
        """Compute a composite score (0-100) from all available sources.

        This replaces LunarCrush's Galaxy Score.
        """
        scores = []
        weights = []

        # Reddit sentiment (-1 to +1) → 0-100
        if reddit.get("mention_count", 0) > 0:
            reddit_score = (reddit["avg_sentiment"] + 1) * 50  # -1..+1 → 0..100
            # Boost if high engagement
            if reddit.get("avg_upvote_ratio", 0.5) > 0.7:
                reddit_score = min(100, reddit_score + 5)
            scores.append(reddit_score)
            weights.append(0.35)

        # CryptoPanic news sentiment (-1 to +1) → 0-100
        if cryptopanic.get("news_count", 0) > 0:
            cp_score = (cryptopanic["sentiment_score"] + 1) * 50
            scores.append(cp_score)
            weights.append(0.25)

        # Fear & Greed (already 0-100)
        if "value" in fear_greed:
            scores.append(fear_greed["value"])
            weights.append(0.25)

        # CoinGecko community score (0-100ish)
        cg_score = coingecko.get("community_score")
        if cg_score and cg_score > 0:
            scores.append(min(100, cg_score))
            weights.append(0.15)

        if not scores:
            return None

        # Weighted average
        total_weight = sum(weights)
        return sum(s * w for s, w in zip(scores, weights)) / total_weight

    def _compute_sentiment(self, reddit: dict, cryptopanic: dict,
                            fear_greed: dict) -> float:
        """Compute sentiment on 0-5 scale (matching what SentimentAnalyzer expects).

        0 = very bearish, 2.5 = neutral, 5 = very bullish.
        """
        scores = []
        weights = []

        # Reddit: -1..+1 → 0..5
        if reddit.get("mention_count", 0) > 0:
            scores.append((reddit["avg_sentiment"] + 1) * 2.5)
            weights.append(0.4)

        # CryptoPanic: -1..+1 → 0..5
        if cryptopanic.get("news_count", 0) > 0:
            scores.append((cryptopanic["sentiment_score"] + 1) * 2.5)
            weights.append(0.35)

        # Fear & Greed: 0..100 → 0..5
        if "value" in fear_greed:
            scores.append(fear_greed["value"] / 20)
            weights.append(0.25)

        if not scores:
            return 2.5  # neutral default

        total_weight = sum(weights)
        return sum(s * w for s, w in zip(scores, weights)) / total_weight

    def _compute_social_volume(self, reddit: dict, cryptopanic: dict) -> float:
        """Compute social volume from mention counts."""
        volume = 0
        volume += reddit.get("mention_count", 0) * 10  # Weight Reddit mentions higher
        volume += reddit.get("total_comments", 0)
        volume += cryptopanic.get("news_count", 0) * 5
        return float(volume)

    def fetch_asset_data(self, symbol: str, fear_greed: dict | None = None) -> dict:
        """Fetch and aggregate all source data for a single asset.

        Args:
            symbol: Coin symbol (e.g., "BTC")
            fear_greed: Pre-fetched Fear & Greed data (shared across assets)

        Returns:
            Record matching the social_data DB schema.
        """
        reddit = self._fetch_reddit(symbol)
        cryptopanic = self._fetch_cryptopanic(symbol)
        fg = fear_greed or self._fetch_fear_greed()
        coingecko = self._fetch_coingecko(symbol)

        composite = self._compute_composite_score(reddit, cryptopanic, fg, coingecko)
        sentiment = self._compute_sentiment(reddit, cryptopanic, fg)
        social_volume = self._compute_social_volume(reddit, cryptopanic)

        raw = {
            "reddit": reddit,
            "cryptopanic": cryptopanic,
            "fear_greed": fg,
            "coingecko": {k: v for k, v in coingecko.items() if k != "raw"},
        }

        return {
            "asset": symbol,
            "timestamp": int(datetime.now(timezone.utc).timestamp()),
            "galaxy_score": composite,
            "alt_rank": None,
            "social_volume": social_volume,
            "social_dominance": None,  # Computed in fetch_watchlist_data across all assets
            "sentiment": sentiment,
            "market_cap": coingecko.get("market_cap"),
            "price": coingecko.get("price"),
            "raw_json": str(raw),
        }

    def fetch_watchlist_data(self, product_ids: list[str]) -> list[dict]:
        """Fetch social data for all watchlist assets.

        Args:
            product_ids: List of Coinbase product IDs (e.g., ["BTC-USD", "ETH-USD"])

        Returns:
            List of social data records ready for database insertion.
        """
        # Fetch Fear & Greed once (it's market-wide, not per-asset)
        fear_greed = self._fetch_fear_greed()

        records = []
        total_social_volume = 0

        for pid in product_ids:
            symbol = product_to_symbol(pid)
            try:
                record = self.fetch_asset_data(symbol, fear_greed=fear_greed)
                total_social_volume += record.get("social_volume", 0)
                records.append(record)
            except Exception as e:
                logger.error(f"Failed to fetch social data for {symbol}: {e}")

        # Compute social dominance (% of total social volume)
        if total_social_volume > 0:
            for r in records:
                r["social_dominance"] = (r.get("social_volume", 0) / total_social_volume) * 100

        source_names = list(self.sources_available.keys())
        logger.info(f"Aggregated social data for {len(records)} assets from {source_names}")
        return records
