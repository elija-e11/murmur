"""Technical analysis — compute indicators on OHLCV data and return normalized scores."""

import pandas as pd
import ta


def candles_to_df(candles: list[dict]) -> pd.DataFrame:
    """Convert candle dicts to a pandas DataFrame."""
    df = pd.DataFrame(candles)
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    return df


class TechnicalAnalyzer:
    """Computes technical indicators and returns normalized scores."""

    def __init__(self, config: dict):
        tc = config.get("technical", {})
        self.rsi_period = tc.get("rsi_period", 14)
        self.rsi_overbought = tc.get("rsi_overbought", 70)
        self.rsi_oversold = tc.get("rsi_oversold", 30)
        self.macd_fast = tc.get("macd_fast", 12)
        self.macd_slow = tc.get("macd_slow", 26)
        self.macd_signal = tc.get("macd_signal", 9)
        self.bb_period = tc.get("bb_period", 20)
        self.bb_std = tc.get("bb_std", 2)
        self.ema_periods = tc.get("ema_periods", [9, 21, 50])

    def compute_all(self, candles: list[dict]) -> dict:
        """Compute all indicators and return a summary dict.

        Returns:
            {
                "rsi": float,           # 0-100
                "rsi_score": float,     # -1 to +1 (negative=oversold/bullish, positive=overbought/bearish)
                "macd": float,
                "macd_signal": float,
                "macd_histogram": float,
                "macd_score": float,    # -1 to +1
                "bb_upper": float,
                "bb_middle": float,
                "bb_lower": float,
                "bb_pctb": float,       # %B: 0=at lower band, 1=at upper band
                "bb_score": float,      # -1 to +1
                "ema_{period}": float,  # for each period
                "ema_score": float,     # -1 to +1 based on EMA alignment
                "volume_sma": float,
                "volume_ratio": float,  # current volume / SMA volume
                "volume_score": float,  # 0 to +1
                "composite_score": float,  # weighted average of all scores
            }
        """
        if len(candles) < max(self.macd_slow + self.macd_signal, self.bb_period, max(self.ema_periods)) + 5:
            return {"error": "insufficient_data", "composite_score": 0}

        df = candles_to_df(candles)
        result = {}

        # RSI
        rsi_series = ta.momentum.RSIIndicator(df["close"], window=self.rsi_period).rsi()
        rsi = rsi_series.iloc[-1]
        result["rsi"] = rsi
        # Score: oversold = bullish (+1), overbought = bearish (-1)
        if rsi <= self.rsi_oversold:
            result["rsi_score"] = (self.rsi_oversold - rsi) / self.rsi_oversold  # 0 to +1
        elif rsi >= self.rsi_overbought:
            result["rsi_score"] = -(rsi - self.rsi_overbought) / (100 - self.rsi_overbought)  # -1 to 0
        else:
            # Neutral zone: slight linear interpolation
            result["rsi_score"] = (50 - rsi) / 50 * 0.3

        # MACD
        macd_ind = ta.trend.MACD(
            df["close"], window_slow=self.macd_slow,
            window_fast=self.macd_fast, window_sign=self.macd_signal,
        )
        macd_line = macd_ind.macd().iloc[-1]
        signal_line = macd_ind.macd_signal().iloc[-1]
        histogram = macd_ind.macd_diff().iloc[-1]
        result["macd"] = macd_line
        result["macd_signal"] = signal_line
        result["macd_histogram"] = histogram

        # Normalize MACD score by price to make it comparable across assets
        price = df["close"].iloc[-1]
        macd_pct = histogram / price * 100 if price > 0 else 0
        result["macd_score"] = max(-1, min(1, macd_pct * 10))

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df["close"], window=self.bb_period, window_dev=self.bb_std)
        result["bb_upper"] = bb.bollinger_hband().iloc[-1]
        result["bb_middle"] = bb.bollinger_mavg().iloc[-1]
        result["bb_lower"] = bb.bollinger_lband().iloc[-1]
        bb_width = result["bb_upper"] - result["bb_lower"]
        if bb_width > 0:
            result["bb_pctb"] = (price - result["bb_lower"]) / bb_width
        else:
            result["bb_pctb"] = 0.5
        # Score: near lower band = bullish, near upper = bearish
        result["bb_score"] = max(-1, min(1, (0.5 - result["bb_pctb"]) * 2))

        # EMAs
        ema_values = {}
        for period in self.ema_periods:
            ema = ta.trend.EMAIndicator(df["close"], window=period).ema_indicator().iloc[-1]
            result[f"ema_{period}"] = ema
            ema_values[period] = ema

        # EMA score: price above all EMAs = bullish, below all = bearish
        sorted_periods = sorted(self.ema_periods)
        above_count = sum(1 for p in sorted_periods if price > ema_values[p])
        total = len(sorted_periods)
        # Also check EMA alignment (short > medium > long = bullish)
        aligned_bullish = all(
            ema_values[sorted_periods[i]] > ema_values[sorted_periods[i + 1]]
            for i in range(total - 1)
        )
        aligned_bearish = all(
            ema_values[sorted_periods[i]] < ema_values[sorted_periods[i + 1]]
            for i in range(total - 1)
        )

        ema_score = (above_count / total - 0.5) * 2  # -1 to +1
        if aligned_bullish:
            ema_score = min(1, ema_score + 0.2)
        elif aligned_bearish:
            ema_score = max(-1, ema_score - 0.2)
        result["ema_score"] = ema_score

        # Volume
        vol_sma = df["volume"].rolling(window=20).mean().iloc[-1]
        current_vol = df["volume"].iloc[-1]
        result["volume_sma"] = vol_sma
        result["volume_ratio"] = current_vol / vol_sma if vol_sma > 0 else 1.0
        # Volume score: higher volume = more conviction (capped at 1)
        result["volume_score"] = min(1, max(0, (result["volume_ratio"] - 1) * 0.5 + 0.5))

        # Composite score (simple equal weighting; strategy engine does final weighting)
        scores = [result["rsi_score"], result["macd_score"], result["bb_score"], result["ema_score"]]
        result["composite_score"] = sum(scores) / len(scores)

        return result
