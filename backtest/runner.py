"""Backtesting framework — replay historical data through strategies."""

import logging
import os
import tempfile
from datetime import datetime, timezone

from src.analysis.technical import TechnicalAnalyzer
from src.analysis.sentiment import SentimentAnalyzer
from src.strategy.engine import StrategyEngine
from src.execution.paper import PaperTrader
from src.storage.db import Database
from src.config import get_config

logger = logging.getLogger(__name__)


class BacktestResult:
    """Container for backtest performance metrics."""

    def __init__(self, trades: list[dict], starting_balance: float, ending_balance: float):
        self.trades = trades
        self.starting_balance = starting_balance
        self.ending_balance = ending_balance

    @property
    def total_return(self) -> float:
        return (self.ending_balance - self.starting_balance) / self.starting_balance

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def buy_trades(self) -> int:
        return sum(1 for t in self.trades if t["side"] == "buy")

    @property
    def sell_trades(self) -> int:
        return sum(1 for t in self.trades if t["side"] == "sell")

    @property
    def win_rate(self) -> float:
        """Win rate based on sell trades with positive realized P&L."""
        sells = [t for t in self.trades if t["side"] == "sell"]
        if not sells:
            return 0
        wins = sum(1 for t in sells if t.get("realized_pnl", 0) > 0)
        return wins / len(sells)

    def summary(self) -> str:
        return (
            f"Backtest Results:\n"
            f"  Starting Balance: ${self.starting_balance:,.2f}\n"
            f"  Ending Balance:   ${self.ending_balance:,.2f}\n"
            f"  Total Return:     {self.total_return:+.1%}\n"
            f"  Total Trades:     {self.total_trades} ({self.buy_trades} buys, {self.sell_trades} sells)\n"
            f"  Win Rate:         {self.win_rate:.0%}\n"
        )


class Backtester:
    """Replays stored historical data through the strategy engine."""

    def __init__(self, config: dict | None = None):
        self.config = config or get_config()

    def run(self, product_id: str, source_db_path: str,
            timeframe: str = "1h", start_ts: int | None = None,
            end_ts: int | None = None) -> BacktestResult:
        """Run backtest using stored candles and social data.

        Args:
            product_id: e.g. "BTC-USD"
            source_db_path: Path to database with historical data
            timeframe: Candle timeframe
            start_ts: Start timestamp (unix)
            end_ts: End timestamp (unix)

        Returns:
            BacktestResult with performance metrics
        """
        source_db = Database(source_db_path)

        # Create temp database for backtest execution
        fd, temp_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        bt_db = Database(temp_path)

        tech_analyzer = TechnicalAnalyzer(self.config)
        sentiment_analyzer = SentimentAnalyzer(self.config)
        engine = StrategyEngine(self.config)
        trader = PaperTrader(bt_db, self.config)

        # Load all candles
        candles = source_db.get_candles(product_id, timeframe, limit=10000, since=start_ts)
        if end_ts:
            candles = [c for c in candles if c["timestamp"] <= end_ts]

        asset = product_id.split("-")[0]
        social_records = source_db.get_social_data(asset, limit=10000, since=start_ts)

        logger.info(f"Backtesting {product_id} with {len(candles)} candles")

        # Minimum candles needed before we can analyze
        min_candles = max(
            self.config.get("technical", {}).get("macd_slow", 26) +
            self.config.get("technical", {}).get("macd_signal", 9),
            max(self.config.get("technical", {}).get("ema_periods", [50])),
            self.config.get("technical", {}).get("bb_period", 20),
        ) + 10

        # Simulate forward through time
        for i in range(min_candles, len(candles)):
            window = candles[:i + 1]
            current_price = window[-1]["close"]
            current_ts = window[-1]["timestamp"]

            # Technical analysis on candle window
            tech = tech_analyzer.compute_all(window)
            if tech.get("error"):
                continue

            # Get social records up to this point
            relevant_social = [s for s in social_records if s["timestamp"] <= current_ts]
            sentiment = sentiment_analyzer.analyze(relevant_social[-50:])

            # Strategy evaluation
            decision = engine.evaluate(product_id, tech, sentiment)

            if decision["action"] == "buy":
                trader.execute_buy(product_id, current_price)
            elif decision["action"] == "sell":
                trader.execute_sell(product_id, current_price)

            # Check stop-loss / take-profit
            trader.check_stop_loss_take_profit({product_id: current_price})

        # Close any remaining positions at final price
        final_price = candles[-1]["close"] if candles else 0
        for pos in trader.get_open_positions():
            if pos["asset"] == asset:
                trader.execute_sell(product_id, final_price)

        trades = bt_db.get_trades(execution_mode="paper", limit=10000)
        result = BacktestResult(
            trades=trades,
            starting_balance=self.config.get("execution", {}).get("paper_starting_balance", 10000),
            ending_balance=trader.get_portfolio_value(),
        )

        os.unlink(temp_path)
        return result


def main():
    """CLI entry point for backtesting."""
    import argparse

    parser = argparse.ArgumentParser(description="Murmur Backtester")
    parser.add_argument("--db", required=True, help="Path to database with historical data")
    parser.add_argument("--product", default="BTC-USD", help="Product to backtest")
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    backtester = Backtester()
    result = backtester.run(args.product, args.db, args.timeframe)
    print(result.summary())


if __name__ == "__main__":
    main()
