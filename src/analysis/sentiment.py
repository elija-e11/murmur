"""Sentiment analysis — process social data into actionable signals."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Processes social data into sentiment scores and detects anomalies."""

    def __init__(self, config: dict):
        sc = config.get("sentiment", {})
        self.zscore_spike = sc.get("zscore_spike_threshold", 2.0)
        self.zscore_extreme = sc.get("zscore_extreme_threshold", 3.0)
        self.rolling_window = sc.get("rolling_window", 24)
        self.momentum_periods = sc.get("momentum_periods", 6)

    def analyze(self, social_records: list[dict]) -> dict:
        """Analyze a series of social data records for an asset.

        Args:
            social_records: List of social_data dicts sorted by timestamp ascending.

        Returns:
            {
                "sentiment_score": float,      # -1 to +1 current sentiment level
                "sentiment_momentum": float,   # rate of change in sentiment
                "social_volume_zscore": float,  # Z-score of current social volume
                "social_spike": bool,          # social volume spike detected
                "social_extreme": bool,        # extreme social spike (potential pump)
                "galaxy_score": float | None,  # latest galaxy score
                "crowd_signal": float,         # composite crowd signal -1 to +1
            }
        """
        if len(social_records) < 3:
            return {
                "sentiment_score": 0,
                "sentiment_momentum": 0,
                "social_volume_zscore": 0,
                "social_spike": False,
                "social_extreme": False,
                "galaxy_score": None,
                "crowd_signal": 0,
            }

        result = {}

        # Extract arrays
        sentiments = [r.get("sentiment") or 0 for r in social_records]
        social_volumes = [r.get("social_volume") or 0 for r in social_records]
        galaxy_scores = [r.get("galaxy_score") for r in social_records]

        latest = social_records[-1]

        # --- Sentiment score ---
        # Sentiment is on 0-5 scale; normalize to -1 to +1
        raw_sentiment = latest.get("sentiment") or 0
        result["sentiment_score"] = max(-1, min(1, (raw_sentiment - 2.5) / 2.5))

        # --- Sentiment momentum ---
        if len(sentiments) >= self.momentum_periods:
            recent = sentiments[-self.momentum_periods:]
            older = sentiments[-self.momentum_periods * 2:-self.momentum_periods] if len(sentiments) >= self.momentum_periods * 2 else sentiments[:self.momentum_periods]
            avg_recent = np.mean(recent)
            avg_older = np.mean(older)
            if avg_older > 0:
                result["sentiment_momentum"] = max(-1, min(1, (avg_recent - avg_older) / max(avg_older, 1)))
            else:
                result["sentiment_momentum"] = 0
        else:
            result["sentiment_momentum"] = 0

        # --- Social volume Z-score ---
        window = social_volumes[-self.rolling_window:] if len(social_volumes) >= self.rolling_window else social_volumes
        if len(window) >= 3:
            mean_vol = np.mean(window[:-1])  # exclude current from baseline
            std_vol = np.std(window[:-1])
            current_vol = social_volumes[-1]
            if std_vol > 0:
                zscore = (current_vol - mean_vol) / std_vol
            else:
                zscore = 0
        else:
            zscore = 0

        result["social_volume_zscore"] = zscore
        result["social_spike"] = zscore >= self.zscore_spike
        result["social_extreme"] = zscore >= self.zscore_extreme

        # --- Galaxy Score ---
        valid_gs = [g for g in galaxy_scores if g is not None]
        result["galaxy_score"] = valid_gs[-1] if valid_gs else None

        # --- Composite crowd signal ---
        # Weighted combination of sentiment, momentum, and social interest
        components = []

        # Sentiment direction (40% weight)
        components.append(result["sentiment_score"] * 0.4)

        # Sentiment momentum (30% weight)
        components.append(result["sentiment_momentum"] * 0.3)

        # Social volume interest (20% weight) — spike = positive signal, but extreme = caution
        if result["social_extreme"]:
            vol_signal = 0.3  # Dampen extreme spikes (likely hype)
        elif result["social_spike"]:
            vol_signal = 0.8  # Strong interest
        elif zscore > 0:
            vol_signal = min(1, zscore / self.zscore_spike) * 0.5
        else:
            vol_signal = max(-1, zscore / self.zscore_spike) * 0.3
        components.append(vol_signal * 0.2)

        # Galaxy score (10% weight)
        if result["galaxy_score"] is not None:
            gs_normalized = (result["galaxy_score"] - 50) / 50  # 0-100 → -1 to +1
            components.append(gs_normalized * 0.1)

        result["crowd_signal"] = max(-1, min(1, sum(components)))

        return result
