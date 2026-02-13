"""Strategy engine — combines signals from all strategies into buy/sell/hold decisions."""

import logging
from datetime import datetime, timezone

from src.strategy.signals import (
    Action,
    Signal,
    divergence_signal,
    hype_filter_signal,
    mean_reversion_signal,
    social_momentum_signal,
)

logger = logging.getLogger(__name__)


class StrategyEngine:
    """Runs all strategies and produces a final decision per asset."""

    STRATEGIES = {
        "social_momentum": social_momentum_signal,
        "divergence_watch": divergence_signal,
        "hype_filter": hype_filter_signal,
        "mean_reversion": mean_reversion_signal,
    }

    def __init__(self, config: dict):
        sc = config.get("strategy", {})
        self.min_confidence = sc.get("min_confidence", 0.6)

    def evaluate(self, product_id: str, tech: dict, sentiment: dict) -> dict:
        """Run all strategies and produce a combined decision.

        Returns:
            {
                "product_id": str,
                "action": "buy" | "sell" | "hold",
                "confidence": float,
                "signals": [Signal, ...],
                "reasoning": str,
                "timestamp": int,
            }
        """
        signals: list[Signal] = []
        for name, strategy_fn in self.STRATEGIES.items():
            sig = strategy_fn(product_id, tech, sentiment)
            signals.append(sig)

        # Check hype filter first — if it flags, override to HOLD
        hype_sig = next((s for s in signals if s.strategy == "hype_filter"), None)
        if hype_sig and hype_sig.confidence >= 0.7:
            return {
                "product_id": product_id,
                "action": Action.HOLD.value,
                "confidence": hype_sig.confidence,
                "signals": [s.to_dict() for s in signals],
                "reasoning": f"BLOCKED by hype filter: {hype_sig.reasoning}",
                "timestamp": int(datetime.now(timezone.utc).timestamp()),
            }

        # Aggregate buy/sell signals
        buy_signals = [s for s in signals if s.action == Action.BUY]
        sell_signals = [s for s in signals if s.action == Action.SELL]

        if buy_signals:
            best = max(buy_signals, key=lambda s: s.confidence)
            avg_confidence = sum(s.confidence for s in buy_signals) / len(buy_signals)
            # Boost confidence if multiple strategies agree
            agreement_bonus = min(0.15, (len(buy_signals) - 1) * 0.075)
            final_confidence = min(1.0, avg_confidence + agreement_bonus)

            if final_confidence >= self.min_confidence:
                reasons = [f"{s.strategy}({s.confidence:.2f}): {s.reasoning}" for s in buy_signals]
                return {
                    "product_id": product_id,
                    "action": Action.BUY.value,
                    "confidence": final_confidence,
                    "signals": [s.to_dict() for s in signals],
                    "reasoning": " | ".join(reasons),
                    "timestamp": int(datetime.now(timezone.utc).timestamp()),
                }

        if sell_signals:
            best = max(sell_signals, key=lambda s: s.confidence)
            if best.confidence >= self.min_confidence:
                return {
                    "product_id": product_id,
                    "action": Action.SELL.value,
                    "confidence": best.confidence,
                    "signals": [s.to_dict() for s in signals],
                    "reasoning": f"{best.strategy}: {best.reasoning}",
                    "timestamp": int(datetime.now(timezone.utc).timestamp()),
                }

        # Default: hold — report highest signal confidence so user can see what's building
        all_nonzero = [s for s in signals if s.confidence > 0 and s.strategy != "hype_filter"]
        if all_nonzero:
            best = max(all_nonzero, key=lambda s: s.confidence)
            hold_confidence = best.confidence
            hold_reasoning = f"strongest signal below threshold: {best.strategy}({best.confidence:.2f}) — {best.reasoning}"
        else:
            hold_confidence = 0
            hold_reasoning = "no actionable signals"

        return {
            "product_id": product_id,
            "action": Action.HOLD.value,
            "confidence": hold_confidence,
            "signals": [s.to_dict() for s in signals],
            "reasoning": hold_reasoning,
            "timestamp": int(datetime.now(timezone.utc).timestamp()),
        }
