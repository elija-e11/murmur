"""Signal generation — translate analysis outputs into trading signals."""

from dataclasses import dataclass, field
from enum import Enum


class Action(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Signal:
    product_id: str
    strategy: str
    action: Action
    confidence: float  # 0 to 1
    reasoning: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "product_id": self.product_id,
            "strategy": self.strategy,
            "action": self.action.value,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "metadata": self.metadata,
        }


def social_momentum_signal(product_id: str, tech: dict, sentiment: dict) -> Signal:
    """Social volume spike + positive sentiment + price above 21 EMA → buy."""
    reasons = []
    confidence = 0

    has_spike = sentiment.get("social_spike", False)
    crowd = sentiment.get("crowd_signal", 0)
    ema_21 = tech.get("ema_21", 0)
    ema_score = tech.get("ema_score", 0)

    if has_spike and crowd > 0.2:
        confidence += 0.4
        reasons.append(f"social spike (z={sentiment.get('social_volume_zscore', 0):.1f})")
    if crowd > 0.3:
        confidence += 0.2
        reasons.append(f"positive crowd signal ({crowd:.2f})")
    if ema_score > 0:
        confidence += 0.2
        reasons.append("price above key EMAs")
    if tech.get("volume_ratio", 1) > 1.2:
        confidence += 0.1
        reasons.append(f"elevated volume ({tech.get('volume_ratio', 1):.1f}x)")

    # Penalize if sentiment is extreme (potential pump)
    if sentiment.get("social_extreme", False):
        confidence *= 0.5
        reasons.append("WARNING: extreme social spike — reduced confidence")

    confidence = min(1.0, confidence)
    action = Action.BUY if confidence >= 0.5 else Action.HOLD

    return Signal(
        product_id=product_id,
        strategy="social_momentum",
        action=action,
        confidence=confidence,
        reasoning="; ".join(reasons) if reasons else "no signals",
        metadata={"crowd_signal": crowd, "ema_score": ema_score},
    )


def divergence_signal(product_id: str, tech: dict, sentiment: dict) -> Signal:
    """Rising sentiment but falling price → flag for review (potential early entry)."""
    reasons = []
    confidence = 0

    sentiment_momentum = sentiment.get("sentiment_momentum", 0)
    ema_score = tech.get("ema_score", 0)
    rsi_score = tech.get("rsi_score", 0)

    # Bullish divergence: sentiment rising but price weak
    if sentiment_momentum > 0.2 and ema_score < 0:
        confidence += 0.3
        reasons.append(f"bullish divergence: sentiment rising ({sentiment_momentum:.2f}) while price weak")

    if rsi_score > 0.2:  # RSI in oversold territory (bullish)
        confidence += 0.2
        reasons.append(f"RSI supports reversal (score={rsi_score:.2f})")

    if sentiment.get("crowd_signal", 0) > 0.2:
        confidence += 0.15
        reasons.append("positive crowd signal building")

    # Bearish divergence: sentiment falling but price strong
    if sentiment_momentum < -0.2 and ema_score > 0.3:
        confidence = 0.3
        reasons = [f"bearish divergence: sentiment falling ({sentiment_momentum:.2f}) while price strong"]
        return Signal(
            product_id=product_id, strategy="divergence_watch",
            action=Action.SELL, confidence=confidence,
            reasoning="; ".join(reasons),
        )

    confidence = min(1.0, confidence)
    action = Action.BUY if confidence >= 0.5 else Action.HOLD

    return Signal(
        product_id=product_id, strategy="divergence_watch",
        action=action, confidence=confidence,
        reasoning="; ".join(reasons) if reasons else "no divergence detected",
    )


def hype_filter_signal(product_id: str, tech: dict, sentiment: dict) -> Signal:
    """Extreme social spike with no technical confirmation → skip (likely pump & dump)."""
    if not sentiment.get("social_extreme", False):
        return Signal(
            product_id=product_id, strategy="hype_filter",
            action=Action.HOLD, confidence=0,
            reasoning="no extreme social activity",
        )

    tech_score = tech.get("composite_score", 0)
    zscore = sentiment.get("social_volume_zscore", 0)

    if tech_score < 0.2:
        return Signal(
            product_id=product_id, strategy="hype_filter",
            action=Action.HOLD, confidence=0.8,
            reasoning=f"HYPE ALERT: extreme social spike (z={zscore:.1f}) with weak technicals ({tech_score:.2f}) — likely pump & dump, avoiding",
        )

    return Signal(
        product_id=product_id, strategy="hype_filter",
        action=Action.HOLD, confidence=0.3,
        reasoning=f"social spike (z={zscore:.1f}) with some technical support ({tech_score:.2f}) — proceed with caution",
    )


def mean_reversion_signal(product_id: str, tech: dict, sentiment: dict) -> Signal:
    """Oversold RSI + recovering sentiment → buy opportunity."""
    reasons = []
    confidence = 0

    rsi = tech.get("rsi", 50)
    rsi_score = tech.get("rsi_score", 0)
    bb_score = tech.get("bb_score", 0)
    sentiment_momentum = sentiment.get("sentiment_momentum", 0)

    if rsi_score > 0.3:  # RSI in oversold territory
        confidence += 0.3
        reasons.append(f"oversold RSI ({rsi:.1f})")

    if bb_score > 0.3:  # Near lower Bollinger Band
        confidence += 0.2
        reasons.append(f"near lower Bollinger Band (score={bb_score:.2f})")

    if sentiment_momentum > 0:
        confidence += 0.2
        reasons.append(f"sentiment recovering ({sentiment_momentum:.2f})")

    if tech.get("volume_ratio", 1) > 1.5:
        confidence += 0.1
        reasons.append("volume surge on dip")

    confidence = min(1.0, confidence)
    action = Action.BUY if confidence >= 0.5 else Action.HOLD

    return Signal(
        product_id=product_id, strategy="mean_reversion",
        action=action, confidence=confidence,
        reasoning="; ".join(reasons) if reasons else "not oversold",
    )
