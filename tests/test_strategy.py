"""Tests for strategy engine and signal generation."""

import pytest
from src.strategy.signals import (
    Action,
    social_momentum_signal,
    divergence_signal,
    hype_filter_signal,
    mean_reversion_signal,
)
from src.strategy.engine import StrategyEngine


DEFAULT_CONFIG = {
    "strategy": {"min_confidence": 0.6},
}


def make_tech(composite=0.3, rsi=50, rsi_score=0, macd_score=0, bb_score=0,
              ema_score=0.2, ema_21=100, volume_ratio=1.0, volume_score=0.5):
    return {
        "composite_score": composite,
        "rsi": rsi, "rsi_score": rsi_score,
        "macd_score": macd_score, "bb_score": bb_score,
        "ema_score": ema_score, "ema_21": ema_21,
        "volume_ratio": volume_ratio, "volume_score": volume_score,
    }


def make_sentiment(crowd=0.3, spike=False, extreme=False, zscore=1.0, momentum=0.1):
    return {
        "crowd_signal": crowd,
        "social_spike": spike,
        "social_extreme": extreme,
        "social_volume_zscore": zscore,
        "sentiment_momentum": momentum,
        "sentiment_score": crowd,
        "galaxy_score": 60,
    }


class TestSocialMomentum:
    def test_strong_buy_signal(self):
        tech = make_tech(ema_score=0.5, volume_ratio=1.5)
        sentiment = make_sentiment(crowd=0.5, spike=True, zscore=2.5)
        sig = social_momentum_signal("BTC-USD", tech, sentiment)
        assert sig.action == Action.BUY
        assert sig.confidence > 0.5

    def test_no_signal_without_spike(self):
        tech = make_tech()
        sentiment = make_sentiment(crowd=0.1, spike=False)
        sig = social_momentum_signal("BTC-USD", tech, sentiment)
        assert sig.action == Action.HOLD

    def test_extreme_spike_reduces_confidence(self):
        tech = make_tech(ema_score=0.5, volume_ratio=1.5)
        sentiment_normal = make_sentiment(crowd=0.5, spike=True, zscore=2.5)
        sentiment_extreme = make_sentiment(crowd=0.5, spike=True, extreme=True, zscore=4.0)

        sig_normal = social_momentum_signal("BTC-USD", tech, sentiment_normal)
        sig_extreme = social_momentum_signal("BTC-USD", tech, sentiment_extreme)
        assert sig_extreme.confidence < sig_normal.confidence


class TestHypeFilter:
    def test_blocks_pump_and_dump(self):
        tech = make_tech(composite=0.1)
        sentiment = make_sentiment(extreme=True, zscore=4.0)
        sig = hype_filter_signal("BTC-USD", tech, sentiment)
        assert sig.action == Action.HOLD
        assert sig.confidence >= 0.7
        assert "pump" in sig.reasoning.lower()

    def test_allows_when_not_extreme(self):
        tech = make_tech()
        sentiment = make_sentiment(extreme=False)
        sig = hype_filter_signal("BTC-USD", tech, sentiment)
        assert sig.confidence == 0


class TestMeanReversion:
    def test_oversold_buy(self):
        tech = make_tech(rsi=25, rsi_score=0.5, bb_score=0.5)
        sentiment = make_sentiment(momentum=0.3)
        sig = mean_reversion_signal("BTC-USD", tech, sentiment)
        assert sig.action == Action.BUY
        assert sig.confidence >= 0.5


class TestStrategyEngine:
    def setup_method(self):
        self.engine = StrategyEngine(DEFAULT_CONFIG)

    def test_hold_when_no_signals(self):
        tech = make_tech()
        sentiment = make_sentiment(crowd=0, spike=False, momentum=0)
        decision = self.engine.evaluate("BTC-USD", tech, sentiment)
        assert decision["action"] == "hold"

    def test_hype_filter_blocks(self):
        tech = make_tech(composite=0.05)
        sentiment = make_sentiment(extreme=True, zscore=5.0)
        decision = self.engine.evaluate("BTC-USD", tech, sentiment)
        assert decision["action"] == "hold"
        assert "hype filter" in decision["reasoning"].lower()

    def test_decision_has_required_keys(self):
        tech = make_tech()
        sentiment = make_sentiment()
        decision = self.engine.evaluate("BTC-USD", tech, sentiment)
        assert "product_id" in decision
        assert "action" in decision
        assert "confidence" in decision
        assert "signals" in decision
        assert "reasoning" in decision
        assert "timestamp" in decision
