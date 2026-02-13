"""Reddit social data source — tracks mention volume, upvotes, and sentiment in crypto subreddits."""

import logging
import time

import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

_vader = SentimentIntensityAnalyzer()

# Add crypto-specific lexicon entries (VADER doesn't know these)
_CRYPTO_LEXICON = {
    "bull": 1.5, "bullish": 2.0, "moon": 2.0, "mooning": 2.5,
    "pump": 1.0, "hodl": 1.0, "accumulate": 1.0, "breakout": 1.5,
    "undervalued": 1.5, "gem": 1.5, "rocket": 2.0, "rally": 1.5,
    "surge": 1.5, "adoption": 1.0,
    "bear": -1.5, "bearish": -2.0, "dump": -1.5, "crash": -2.5,
    "scam": -2.5, "rug": -3.0, "rugpull": -3.0, "rekt": -2.0,
    "plunge": -2.0, "tank": -2.0, "bubble": -1.5, "ponzi": -3.0,
    "fud": -1.0,
}
_vader.lexicon.update(_CRYPTO_LEXICON)

# Subreddits to scan (mix of general + coin-specific)
DEFAULT_SUBREDDITS = [
    "cryptocurrency",
    "CryptoMarkets",
    "bitcoin",
    "ethereum",
    "solana",
    "avalanche",
    "chainlink",
]

# Map coin symbols to search terms
SYMBOL_SEARCH_TERMS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "SOL": ["solana", "sol"],
    "AVAX": ["avalanche", "avax"],
    "LINK": ["chainlink", "link"],
    "DOGE": ["dogecoin", "doge"],
    "ADA": ["cardano", "ada"],
    "DOT": ["polkadot", "dot"],
}


def _keyword_sentiment(text: str) -> float:
    """VADER-based sentiment score with crypto-specific lexicon.

    Returns float from -1 (very bearish) to +1 (very bullish), 0 = neutral.
    """
    return _vader.polarity_scores(text)["compound"]


class RedditSource:
    """Fetches social metrics from Reddit via PRAW."""

    def __init__(self, client_id: str, client_secret: str, user_agent: str = "murmur-bot/1.0"):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )
        self.subreddits = DEFAULT_SUBREDDITS

    def get_asset_metrics(self, symbol: str, lookback_hours: int = 4) -> dict:
        """Get mention count, sentiment, and engagement for a symbol.

        Args:
            symbol: Coin symbol (e.g., "BTC")
            lookback_hours: How far back to search

        Returns:
            {
                "mention_count": int,
                "avg_sentiment": float,   # -1 to +1
                "total_score": int,       # net upvotes across mentions
                "total_comments": int,
                "avg_upvote_ratio": float,
            }
        """
        search_terms = SYMBOL_SEARCH_TERMS.get(symbol, [symbol.lower()])
        query = " OR ".join(search_terms)

        mention_count = 0
        sentiments = []
        total_score = 0
        total_comments = 0
        upvote_ratios = []
        cutoff = time.time() - (lookback_hours * 3600)

        try:
            # Search across crypto subreddits
            subreddit = self.reddit.subreddit("+".join(self.subreddits))
            for submission in subreddit.search(query, sort="new", time_filter="day", limit=100):
                if submission.created_utc < cutoff:
                    continue

                mention_count += 1
                total_score += submission.score
                total_comments += submission.num_comments
                upvote_ratios.append(submission.upvote_ratio)

                # Sentiment from title + selftext
                text = f"{submission.title} {submission.selftext or ''}"
                sentiments.append(_keyword_sentiment(text))

        except Exception as e:
            logger.error(f"Reddit search failed for {symbol}: {e}")

        return {
            "mention_count": mention_count,
            "avg_sentiment": sum(sentiments) / len(sentiments) if sentiments else 0,
            "total_score": total_score,
            "total_comments": total_comments,
            "avg_upvote_ratio": sum(upvote_ratios) / len(upvote_ratios) if upvote_ratios else 0.5,
        }

    def get_trending_mentions(self, lookback_hours: int = 4) -> dict[str, int]:
        """Get mention counts for all tracked symbols across crypto subreddits.

        Returns dict of symbol → mention count.
        """
        counts: dict[str, int] = {}
        cutoff = time.time() - (lookback_hours * 3600)

        try:
            subreddit = self.reddit.subreddit("+".join(self.subreddits))
            for submission in subreddit.hot(limit=200):
                if submission.created_utc < cutoff:
                    continue

                text = f"{submission.title} {submission.selftext or ''}".lower()
                for symbol, terms in SYMBOL_SEARCH_TERMS.items():
                    if any(term in text for term in terms):
                        counts[symbol] = counts.get(symbol, 0) + 1

        except Exception as e:
            logger.error(f"Reddit trending scan failed: {e}")

        return counts
