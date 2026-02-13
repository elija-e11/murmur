"""Tests for paper trading execution and risk management."""

import os
import tempfile
import pytest

from src.storage.db import Database
from src.execution.paper import PaperTrader


DEFAULT_CONFIG = {
    "risk": {
        "max_position_pct": 5.0,
        "stop_loss_pct": 5.0,
        "take_profit_pct": 15.0,
        "max_daily_loss_pct": 3.0,
        "cooldown_minutes": 0,  # Disable for tests
        "max_concurrent_positions": 3,
    },
    "execution": {
        "mode": "paper",
        "paper_starting_balance": 10000.0,
    },
}


@pytest.fixture
def paper_trader():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = Database(path)
    trader = PaperTrader(db, DEFAULT_CONFIG)
    yield trader
    os.unlink(path)


class TestPaperTrader:
    def test_initial_balance(self, paper_trader):
        assert paper_trader.get_balance() == 10000.0

    def test_buy_reduces_cash(self, paper_trader):
        paper_trader.execute_buy("BTC-USD", 50000.0)
        assert paper_trader.get_balance() < 10000.0

    def test_buy_creates_position(self, paper_trader):
        paper_trader.execute_buy("BTC-USD", 50000.0)
        positions = paper_trader.get_open_positions()
        assert len(positions) == 1
        assert positions[0]["asset"] == "BTC"

    def test_sell_returns_cash(self, paper_trader):
        paper_trader.execute_buy("BTC-USD", 50000.0)
        cash_after_buy = paper_trader.get_balance()
        paper_trader.execute_sell("BTC-USD", 55000.0)
        assert paper_trader.get_balance() > cash_after_buy

    def test_sell_closes_position(self, paper_trader):
        paper_trader.execute_buy("BTC-USD", 50000.0)
        paper_trader.execute_sell("BTC-USD", 50000.0)
        positions = paper_trader.get_open_positions()
        assert len(positions) == 0

    def test_position_size_limit(self, paper_trader):
        paper_trader.execute_buy("BTC-USD", 50000.0)
        # Max 5% of 10000 = $500 → ~0.01 BTC
        positions = paper_trader.get_open_positions()
        btc_value = positions[0]["quantity"] * 50000.0
        assert btc_value <= 10000.0 * 0.05 + 1  # Small tolerance

    def test_max_concurrent_positions(self, paper_trader):
        paper_trader.execute_buy("BTC-USD", 50000.0)
        paper_trader.execute_buy("ETH-USD", 3000.0)
        paper_trader.execute_buy("SOL-USD", 100.0)
        # 4th should be blocked
        result = paper_trader.execute_buy("AVAX-USD", 30.0)
        assert result is None

    def test_stop_loss(self, paper_trader):
        paper_trader.execute_buy("BTC-USD", 50000.0)
        # Price drops 6% (below 5% stop-loss)
        sells = paper_trader.check_stop_loss_take_profit({"BTC-USD": 47000.0})
        assert len(sells) == 1

    def test_take_profit(self, paper_trader):
        paper_trader.execute_buy("BTC-USD", 50000.0)
        # Price rises 16% (above 15% take-profit)
        sells = paper_trader.check_stop_loss_take_profit({"BTC-USD": 58000.0})
        assert len(sells) == 1

    def test_no_exit_within_bounds(self, paper_trader):
        paper_trader.execute_buy("BTC-USD", 50000.0)
        # Price within stop-loss and take-profit
        sells = paper_trader.check_stop_loss_take_profit({"BTC-USD": 52000.0})
        assert len(sells) == 0

    def test_portfolio_value(self, paper_trader):
        initial = paper_trader.get_portfolio_value()
        assert initial == 10000.0
        paper_trader.execute_buy("BTC-USD", 50000.0)
        # Update position price
        paper_trader.check_stop_loss_take_profit({"BTC-USD": 50000.0})
        value = paper_trader.get_portfolio_value()
        # Should be approximately the same (minus nothing since same price)
        assert abs(value - 10000.0) < 1.0


class TestRiskManagement:
    def test_cant_sell_without_position(self, paper_trader):
        result = paper_trader.execute_sell("BTC-USD", 50000.0)
        assert result is None

    def test_cant_buy_with_no_cash(self, paper_trader):
        # Drain all cash
        paper_trader.db.upsert_portfolio("USD", 0.0, 1.0, 1.0)
        result = paper_trader.execute_buy("BTC-USD", 50000.0)
        assert result is None
