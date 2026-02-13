"""Tests for technical and sentiment analysis modules."""

import math
import random
import pytest
from src.analysis.technical import TechnicalAnalyzer
from src.analysis.sentiment import SentimentAnalyzer


def make_candles(n=100, base_price=100.0, trend=0.0):
    """Generate synthetic OHLCV candles."""
    candles = []
    price = base_price
    for i in range(n):
        noise = random.gauss(0, 1)
        price = max(1, price + trend + noise)
        high = price + abs(random.gauss(0, 0.5))
        low = price - abs(random.gauss(0, 0.5))
        volume = random.uniform(100, 1000)
        candles.append({
            "timestamp": 1700000000 + i * 3600,
            "open": price - random.gauss(0, 0.2),
            "high": high,
            "low": low,
            "close": price,
            "volume": volume,
        })
    return candles


DEFAULT_CONFIG = {
    "technical": {
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bb_period": 20,
        "bb_std": 2,
        "ema_periods": [9, 21, 50],
    },
    "sentiment": {
        "zscore_spike_threshold": 2.0,
        "zscore_extreme_threshold": 3.0,
        "rolling_window": 24,
        "momentum_periods": 6,
    },
}


class TestTechnicalAnalyzer:
    def setup_method(self):
        self.analyzer = TechnicalAnalyzer(DEFAULT_CONFIG)

    def test_compute_all_returns_required_keys(self):
        candles = make_candles(100)
        result = self.analyzer.compute_all(candles)
        assert "rsi" in result
        assert "rsi_score" in result
        assert "macd" in result
        assert "macd_score" in result
        assert "bb_upper" in result
        assert "bb_score" in result
        assert "ema_9" in result
        assert "ema_21" in result
        assert "ema_50" in result
        assert "ema_score" in result
        assert "volume_ratio" in result
        assert "composite_score" in result

    def test_rsi_in_range(self):
        candles = make_candles(100)
        result = self.analyzer.compute_all(candles)
        assert 0 <= result["rsi"] <= 100

    def test_scores_bounded(self):
        candles = make_candles(100)
        result = self.analyzer.compute_all(candles)
        for key in ("rsi_score", "macd_score", "bb_score", "ema_score", "composite_score"):
            assert -1 <= result[key] <= 1, f"{key} = {result[key]} out of range"

    def test_insufficient_data(self):
        candles = make_candles(10)  # Not enough for 50-period EMA
        result = self.analyzer.compute_all(candles)
        assert result.get("error") == "insufficient_data"

    def test_uptrend_positive_score(self):
        random.seed(42)
        candles = make_candles(100, trend=0.5)
        result = self.analyzer.compute_all(candles)
        # Strong uptrend should have positive EMA score
        assert result["ema_score"] > 0

    def test_bollinger_bands_ordering(self):
        candles = make_candles(100)
        result = self.analyzer.compute_all(candles)
        assert result["bb_lower"] <= result["bb_middle"] <= result["bb_upper"]


class TestSentimentAnalyzer:
    def setup_method(self):
        self.analyzer = SentimentAnalyzer(DEFAULT_CONFIG)

    def _make_social_records(self, n=30, base_sentiment=3.0, base_volume=1000):
        records = []
        for i in range(n):
            records.append({
                "asset": "BTC",
                "timestamp": 1700000000 + i * 600,
                "galaxy_score": 60 + random.gauss(0, 5),
                "sentiment": base_sentiment + random.gauss(0, 0.3),
                "social_volume": base_volume + random.gauss(0, 100),
                "social_dominance": 30,
            })
        return records

    def test_analyze_returns_required_keys(self):
        records = self._make_social_records()
        result = self.analyzer.analyze(records)
        assert "sentiment_score" in result
        assert "sentiment_momentum" in result
        assert "social_volume_zscore" in result
        assert "social_spike" in result
        assert "crowd_signal" in result

    def test_scores_bounded(self):
        records = self._make_social_records()
        result = self.analyzer.analyze(records)
        assert -1 <= result["sentiment_score"] <= 1
        assert -1 <= result["crowd_signal"] <= 1

    def test_spike_detection(self):
        records = self._make_social_records(30, base_volume=1000)
        # Add a huge spike at the end
        records[-1]["social_volume"] = 5000
        result = self.analyzer.analyze(records)
        assert result["social_spike"] == True

    def test_insufficient_data(self):
        result = self.analyzer.analyze([{"sentiment": 3.0}])
        assert result["crowd_signal"] == 0

    def test_positive_sentiment(self):
        records = self._make_social_records(30, base_sentiment=4.5)
        result = self.analyzer.analyze(records)
        assert result["sentiment_score"] > 0

    def test_negative_sentiment(self):
        records = self._make_social_records(30, base_sentiment=0.5)
        result = self.analyzer.analyze(records)
        assert result["sentiment_score"] < 0
