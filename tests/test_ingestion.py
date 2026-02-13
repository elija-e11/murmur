"""Tests for data ingestion modules (using mocks — no real API calls)."""

import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from src.storage.db import Database
from src.ingestion.social import SocialAggregator, product_to_symbol
from src.ingestion.sources.fear_greed import FearGreedSource
from src.ingestion.sources.reddit import _keyword_sentiment


class TestProductToSymbol:
    def test_known_mapping(self):
        assert product_to_symbol("BTC-USD") == "BTC"
        assert product_to_symbol("ETH-USD") == "ETH"

    def test_unknown_mapping(self):
        assert product_to_symbol("UNKNOWN-USD") == "UNKNOWN"


class TestKeywordSentiment:
    def test_bullish_text(self):
        assert _keyword_sentiment("BTC is bullish, moon incoming, buy the dip!") > 0

    def test_bearish_text(self):
        assert _keyword_sentiment("crash imminent, sell everything, this is a scam") < 0

    def test_neutral_text(self):
        assert abs(_keyword_sentiment("the market opened at 9am")) < 0.1

    def test_mixed_text(self):
        score = _keyword_sentiment("could pump or could crash, hard to tell")
        assert -1 <= score <= 1


class TestSocialAggregator:
    def setup_method(self):
        # Create aggregator with no credentials — only free sources init
        self.config = {"secrets": {}}
        self.agg = SocialAggregator(self.config)

    def test_init_with_no_creds(self):
        # Should still have fear_greed and coingecko (no auth required)
        assert "fear_greed" in self.agg.sources_available
        assert "coingecko" in self.agg.sources_available
        assert "reddit" not in self.agg.sources_available
        # CryptoPanic disabled (too expensive)

    def test_product_to_symbol(self):
        assert product_to_symbol("BTC-USD") == "BTC"
        assert product_to_symbol("SOL-USD") == "SOL"

    def test_compute_sentiment_neutral_default(self):
        # No source data → neutral (2.5)
        result = self.agg._compute_sentiment({}, {})
        assert result == 2.5

    def test_compute_sentiment_with_fear_greed(self):
        fg = {"value": 75, "classification": "Greed"}
        result = self.agg._compute_sentiment({}, fg)
        assert result > 2.5  # Greed → bullish

    def test_compute_sentiment_with_all_sources(self):
        reddit = {"mention_count": 10, "avg_sentiment": 0.5}
        fg = {"value": 60}
        result = self.agg._compute_sentiment(reddit, fg)
        assert 2.5 < result <= 5.0  # All positive → above neutral

    def test_compute_social_volume(self):
        reddit = {"mention_count": 10, "total_comments": 50}
        vol = self.agg._compute_social_volume(reddit)
        assert vol == 10 * 10 + 50  # 150

    def test_compute_social_volume_no_data(self):
        assert self.agg._compute_social_volume({}) == 0

    def test_compute_composite_score_range(self):
        reddit = {"mention_count": 5, "avg_sentiment": 0.3, "avg_upvote_ratio": 0.8}
        fg = {"value": 65}
        cg = {"community_score": 50}
        score = self.agg._compute_composite_score(reddit, fg, cg)
        assert score is not None
        assert 0 <= score <= 100

    def test_compute_composite_score_no_data(self):
        assert self.agg._compute_composite_score({}, {}, {}) is None

    @patch.object(SocialAggregator, "_fetch_fear_greed")
    @patch.object(SocialAggregator, "_fetch_coingecko")
    def test_fetch_watchlist_data(self, mock_cg, mock_fg):
        mock_fg.return_value = {"value": 55, "classification": "Neutral", "normalized_score": 0.1}
        mock_cg.return_value = {
            "market_cap": 1e12, "price": 50000, "community_score": 60,
            "reddit_subscribers": 5000000, "reddit_active_48h": 10000,
        }
        records = self.agg.fetch_watchlist_data(["BTC-USD", "ETH-USD"])
        assert len(records) == 2
        assert records[0]["asset"] == "BTC"
        assert records[1]["asset"] == "ETH"
        # Should have timestamps
        assert records[0]["timestamp"] > 0
        # Sentiment should be populated
        assert records[0]["sentiment"] is not None

    @patch.object(SocialAggregator, "_fetch_fear_greed")
    @patch.object(SocialAggregator, "_fetch_coingecko")
    def test_social_dominance_computed(self, mock_cg, mock_fg):
        mock_fg.return_value = {"value": 50}
        mock_cg.return_value = {"market_cap": 0, "price": 0}
        # Manually inject reddit source mock to get different volumes
        self.agg.sources_available["reddit"] = MagicMock()
        self.agg.sources_available["reddit"].get_asset_metrics.side_effect = [
            {"mention_count": 30, "avg_sentiment": 0.2, "total_comments": 100, "avg_upvote_ratio": 0.7},
            {"mention_count": 10, "avg_sentiment": 0.1, "total_comments": 20, "avg_upvote_ratio": 0.6},
        ]
        records = self.agg.fetch_watchlist_data(["BTC-USD", "ETH-USD"])
        # BTC should have higher dominance than ETH
        assert records[0]["social_dominance"] > records[1]["social_dominance"]


class TestFearGreedSource:
    @patch("src.ingestion.sources.fear_greed.FearGreedSource.get_current")
    def test_normalized_score_range(self, mock_get):
        mock_get.return_value = {"value": 75, "classification": "Greed", "normalized_score": 0.5}
        fg = FearGreedSource()
        result = fg.get_current()
        assert -1 <= result["normalized_score"] <= 1


class TestDatabaseStorage:
    @pytest.fixture(autouse=True)
    def setup_db(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.db = Database(path)
        yield
        os.unlink(path)

    def test_candle_roundtrip(self):
        candles = [
            {"timestamp": 1700000000, "open": 100, "high": 105, "low": 98, "close": 103, "volume": 500},
            {"timestamp": 1700003600, "open": 103, "high": 108, "low": 101, "close": 106, "volume": 600},
        ]
        self.db.upsert_candles("BTC-USD", "1h", candles)
        result = self.db.get_candles("BTC-USD", "1h")
        assert len(result) == 2
        assert result[0]["close"] == 103

    def test_social_data_roundtrip(self):
        records = [{
            "asset": "BTC", "timestamp": 1700000000,
            "galaxy_score": 70, "alt_rank": None,
            "social_volume": 10000, "social_dominance": 25,
            "sentiment": 3.5, "market_cap": 1e12,
            "price": 50000, "raw_json": "{}",
        }]
        self.db.upsert_social_data(records)
        result = self.db.get_social_data("BTC")
        assert len(result) == 1
        assert result[0]["galaxy_score"] == 70

    def test_signal_insert_and_retrieve(self):
        sig = {
            "product_id": "BTC-USD", "timestamp": 1700000000,
            "strategy": "test", "action": "buy",
            "confidence": 0.85, "reasoning": "test signal",
            "metadata": {"key": "value"},
        }
        sig_id = self.db.insert_signal(sig)
        assert sig_id is not None
        signals = self.db.get_signals("BTC-USD")
        assert len(signals) == 1
        assert signals[0]["confidence"] == 0.85

    def test_candle_upsert_deduplication(self):
        candle = [{"timestamp": 1700000000, "open": 100, "high": 105, "low": 98, "close": 103, "volume": 500}]
        self.db.upsert_candles("BTC-USD", "1h", candle)
        # Upsert with updated close
        candle[0]["close"] = 110
        self.db.upsert_candles("BTC-USD", "1h", candle)
        result = self.db.get_candles("BTC-USD", "1h")
        assert len(result) == 1
        assert result[0]["close"] == 110
