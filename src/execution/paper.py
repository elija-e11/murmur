"""Paper trading simulator — virtual portfolio tracking with risk management."""

import logging
from datetime import datetime, timezone

from src.storage.db import Database

logger = logging.getLogger(__name__)


class PaperTrader:
    """Simulates trade execution with a virtual portfolio."""

    def __init__(self, db: Database, config: dict):
        self.db = db
        risk = config.get("risk", {})
        exec_cfg = config.get("execution", {})

        self.starting_balance = exec_cfg.get("paper_starting_balance", 10000.0)
        self.max_position_pct = risk.get("max_position_pct", 5.0) / 100
        self.stop_loss_pct = risk.get("stop_loss_pct", 5.0) / 100
        self.take_profit_pct = risk.get("take_profit_pct", 15.0) / 100
        self.max_daily_loss_pct = risk.get("max_daily_loss_pct", 3.0) / 100
        self.cooldown_minutes = risk.get("cooldown_minutes", 60)
        self.max_concurrent = risk.get("max_concurrent_positions", 3)

        self._ensure_cash_position()

    def _ensure_cash_position(self):
        """Initialize USD cash balance if not present."""
        cash = self.db.get_portfolio_asset("USD")
        if cash is None:
            self.db.upsert_portfolio("USD", self.starting_balance, 1.0, 1.0)

    def get_balance(self) -> float:
        """Get current USD cash balance."""
        cash = self.db.get_portfolio_asset("USD")
        return cash["quantity"] if cash else 0

    def get_portfolio_value(self) -> float:
        """Get total portfolio value (cash + positions at current price)."""
        total = self.get_balance()
        for pos in self.db.get_portfolio():
            if pos["asset"] != "USD":
                total += pos["quantity"] * pos["current_price"]
        return total

    def get_open_positions(self) -> list[dict]:
        """Get all non-USD positions."""
        return [p for p in self.db.get_portfolio() if p["asset"] != "USD" and p["quantity"] > 0]

    def check_risk_limits(self, product_id: str, price: float) -> tuple[bool, str]:
        """Check if a trade is allowed by risk management rules.

        Returns:
            (allowed, reason)
        """
        asset = product_id.split("-")[0]

        # Check max concurrent positions
        positions = self.get_open_positions()
        if len(positions) >= self.max_concurrent:
            existing = {p["asset"] for p in positions}
            if asset not in existing:
                return False, f"max concurrent positions ({self.max_concurrent}) reached"

        # Check daily loss limit
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        daily = self.db.get_daily_pnl(limit=1)
        if daily and daily[0]["date"] == today:
            if daily[0]["realized_pnl"] < -(self.get_portfolio_value() * self.max_daily_loss_pct):
                return False, "daily loss limit reached"

        # Check cooldown
        recent = self.db.get_trades(product_id=product_id, execution_mode="paper", limit=1)
        if recent:
            last_time = recent[0]["timestamp"]
            now = int(datetime.now(timezone.utc).timestamp())
            elapsed_min = (now - last_time) / 60
            if elapsed_min < self.cooldown_minutes:
                return False, f"cooldown: {self.cooldown_minutes - elapsed_min:.0f} min remaining"

        return True, "ok"

    def calculate_position_size(self, price: float) -> float:
        """Calculate position size based on max_position_pct of portfolio."""
        portfolio_value = self.get_portfolio_value()
        max_usd = portfolio_value * self.max_position_pct
        available = self.get_balance()
        usd_to_spend = min(max_usd, available)
        return usd_to_spend / price if price > 0 else 0

    def execute_buy(self, product_id: str, price: float, signal_id: int | None = None) -> dict | None:
        """Execute a paper buy order.

        Returns trade record or None if blocked by risk limits.
        """
        allowed, reason = self.check_risk_limits(product_id, price)
        if not allowed:
            logger.warning(f"Buy blocked for {product_id}: {reason}")
            return None

        quantity = self.calculate_position_size(price)
        if quantity <= 0:
            logger.warning(f"Insufficient balance for {product_id}")
            return None

        total = quantity * price
        now = int(datetime.now(timezone.utc).timestamp())
        asset = product_id.split("-")[0]

        trade = {
            "product_id": product_id,
            "side": "buy",
            "order_type": "market",
            "price": price,
            "quantity": quantity,
            "total": total,
            "fee": 0,
            "timestamp": now,
            "signal_id": signal_id,
            "execution_mode": "paper",
            "order_id": f"paper-{now}",
            "status": "filled",
        }
        trade_id = self.db.insert_trade(trade)

        # Update portfolio
        cash = self.get_balance()
        self.db.upsert_portfolio("USD", cash - total, 1.0, 1.0)

        existing = self.db.get_portfolio_asset(asset)
        if existing and existing["quantity"] > 0:
            old_qty = existing["quantity"]
            old_avg = existing["avg_entry_price"]
            new_qty = old_qty + quantity
            new_avg = (old_qty * old_avg + quantity * price) / new_qty
            self.db.upsert_portfolio(asset, new_qty, new_avg, price)
        else:
            self.db.upsert_portfolio(asset, quantity, price, price)

        logger.info(f"PAPER BUY: {quantity:.6f} {asset} @ ${price:.2f} (${total:.2f})")
        return {**trade, "id": trade_id}

    def execute_sell(self, product_id: str, price: float, quantity: float | None = None,
                     signal_id: int | None = None) -> dict | None:
        """Execute a paper sell order. Sells full position if quantity not specified."""
        asset = product_id.split("-")[0]
        position = self.db.get_portfolio_asset(asset)

        if not position or position["quantity"] <= 0:
            logger.warning(f"No position to sell for {asset}")
            return None

        sell_qty = quantity or position["quantity"]
        sell_qty = min(sell_qty, position["quantity"])
        total = sell_qty * price
        now = int(datetime.now(timezone.utc).timestamp())

        # Calculate realized P&L
        entry_price = position["avg_entry_price"]
        realized = (price - entry_price) * sell_qty

        trade = {
            "product_id": product_id,
            "side": "sell",
            "order_type": "market",
            "price": price,
            "quantity": sell_qty,
            "total": total,
            "fee": 0,
            "timestamp": now,
            "signal_id": signal_id,
            "execution_mode": "paper",
            "order_id": f"paper-{now}",
            "status": "filled",
        }
        trade_id = self.db.insert_trade(trade)

        # Update portfolio
        cash = self.get_balance()
        self.db.upsert_portfolio("USD", cash + total, 1.0, 1.0)

        remaining = position["quantity"] - sell_qty
        if remaining > 0:
            self.db.upsert_portfolio(asset, remaining, entry_price, price, realized_pnl=realized)
        else:
            self.db.upsert_portfolio(asset, 0, 0, price, realized_pnl=realized)

        pnl_pct = (realized / (entry_price * sell_qty)) * 100 if entry_price > 0 else 0
        logger.info(
            f"PAPER SELL: {sell_qty:.6f} {asset} @ ${price:.2f} "
            f"(P&L: ${realized:.2f} / {pnl_pct:+.1f}%)"
        )
        return {**trade, "id": trade_id, "realized_pnl": realized}

    def check_stop_loss_take_profit(self, prices: dict[str, float]) -> list[dict]:
        """Check all positions against stop-loss and take-profit levels.

        Args:
            prices: Dict of product_id → current price

        Returns:
            List of executed sell trades.
        """
        sells = []
        for position in self.get_open_positions():
            asset = position["asset"]
            product_id = f"{asset}-USD"
            price = prices.get(product_id)
            if price is None:
                continue

            entry = position["avg_entry_price"]
            if entry <= 0:
                continue

            change_pct = (price - entry) / entry

            if change_pct <= -self.stop_loss_pct:
                logger.warning(f"STOP-LOSS triggered for {asset}: {change_pct*100:+.1f}%")
                result = self.execute_sell(product_id, price)
                if result:
                    sells.append(result)

            elif change_pct >= self.take_profit_pct:
                logger.info(f"TAKE-PROFIT triggered for {asset}: {change_pct*100:+.1f}%")
                result = self.execute_sell(product_id, price)
                if result:
                    sells.append(result)
            else:
                # Update current price in portfolio
                self.db.upsert_portfolio(
                    asset, position["quantity"], entry, price,
                    unrealized_pnl=(price - entry) * position["quantity"],
                )

        return sells
