"""Live trade execution via Coinbase Advanced Trade API with risk management."""

import logging
from datetime import datetime, timezone

from src.ingestion.market import create_coinbase_client
from src.storage.db import Database

logger = logging.getLogger(__name__)


class LiveTrader:
    """Executes real trades on Coinbase with strict risk limits."""

    def __init__(self, db: Database, config: dict, api_key: str = "", api_secret: str = "", key_file: str = ""):
        self.client = create_coinbase_client(api_key, api_secret, key_file)
        if not self.client:
            raise ValueError("Coinbase credentials required for live trading")
        self.db = db

        risk = config.get("risk", {})
        exec_cfg = config.get("execution", {})

        self.max_position_pct = risk.get("max_position_pct", 5.0) / 100
        self.stop_loss_pct = risk.get("stop_loss_pct", 5.0) / 100
        self.take_profit_pct = risk.get("take_profit_pct", 15.0) / 100
        self.max_daily_loss_pct = risk.get("max_daily_loss_pct", 3.0) / 100
        self.cooldown_minutes = risk.get("cooldown_minutes", 60)
        self.max_concurrent = risk.get("max_concurrent_positions", 3)
        self.prefer_limit = exec_cfg.get("prefer_limit_orders", True)
        self.limit_spread_pct = exec_cfg.get("limit_order_spread_pct", 0.1) / 100

    def get_account_balance(self, currency: str = "USD") -> float:
        """Get available balance for a currency."""
        resp = self.client.get_accounts()
        for acct in resp.accounts or []:
            if acct.currency == currency:
                return float((acct.available_balance or {}).get("value", 0))
        return 0

    def get_current_price(self, product_id: str) -> float:
        """Get current market price for a product."""
        resp = self.client.get_product(product_id)
        return float(resp.price or 0)

    def check_risk_limits(self, product_id: str) -> tuple[bool, str]:
        """Check risk management rules before placing a trade."""
        positions = [p for p in self.db.get_portfolio() if p["asset"] != "USD" and p["quantity"] > 0]
        asset = product_id.split("-")[0]

        if len(positions) >= self.max_concurrent:
            existing = {p["asset"] for p in positions}
            if asset not in existing:
                return False, f"max concurrent positions ({self.max_concurrent}) reached"

        # Check cooldown
        recent = self.db.get_trades(product_id=product_id, execution_mode="live", limit=1)
        if recent:
            now = int(datetime.now(timezone.utc).timestamp())
            elapsed_min = (now - recent[0]["timestamp"]) / 60
            if elapsed_min < self.cooldown_minutes:
                return False, f"cooldown: {self.cooldown_minutes - elapsed_min:.0f} min remaining"

        return True, "ok"

    def execute_buy(self, product_id: str, signal_id: int | None = None) -> dict | None:
        """Execute a real buy order on Coinbase."""
        allowed, reason = self.check_risk_limits(product_id)
        if not allowed:
            logger.warning(f"Live buy blocked for {product_id}: {reason}")
            return None

        price = self.get_current_price(product_id)
        if price <= 0:
            logger.error(f"Invalid price for {product_id}")
            return None

        # Calculate position size
        usd_balance = self.get_account_balance("USD")
        # Use a rough portfolio value estimate for position sizing
        max_usd = usd_balance * self.max_position_pct
        spend = min(max_usd, usd_balance * 0.95)  # Keep 5% buffer

        if spend < 1:
            logger.warning(f"Insufficient balance: ${usd_balance:.2f}")
            return None

        now = int(datetime.now(timezone.utc).timestamp())
        client_order_id = f"murmur-{now}"

        try:
            if self.prefer_limit:
                limit_price = str(round(price * (1 - self.limit_spread_pct), 2))
                resp = self.client.limit_order_gtc_buy(
                    client_order_id=client_order_id,
                    product_id=product_id,
                    quote_size=str(round(spend, 2)),
                    limit_price=limit_price,
                )
            else:
                resp = self.client.market_order_buy(
                    client_order_id=client_order_id,
                    product_id=product_id,
                    quote_size=str(round(spend, 2)),
                )

            order_id = getattr(resp, "order_id", client_order_id)
            logger.info(f"LIVE BUY order placed: {product_id} ~${spend:.2f} (order: {order_id})")

            trade = {
                "product_id": product_id,
                "side": "buy",
                "order_type": "limit" if self.prefer_limit else "market",
                "price": price,
                "quantity": spend / price,
                "total": spend,
                "fee": 0,
                "timestamp": now,
                "signal_id": signal_id,
                "execution_mode": "live",
                "order_id": order_id,
                "status": "pending",
            }
            trade_id = self.db.insert_trade(trade)
            return {**trade, "id": trade_id}

        except Exception as e:
            logger.error(f"Failed to place buy order for {product_id}: {e}")
            return None

    def execute_sell(self, product_id: str, quantity: float | None = None,
                     signal_id: int | None = None) -> dict | None:
        """Execute a real sell order on Coinbase."""
        asset = product_id.split("-")[0]
        price = self.get_current_price(product_id)

        if quantity is None:
            quantity = self.get_account_balance(asset)

        if quantity <= 0:
            logger.warning(f"No {asset} to sell")
            return None

        now = int(datetime.now(timezone.utc).timestamp())
        client_order_id = f"murmur-{now}"

        try:
            if self.prefer_limit:
                limit_price = str(round(price * (1 + self.limit_spread_pct), 2))
                resp = self.client.limit_order_gtc_sell(
                    client_order_id=client_order_id,
                    product_id=product_id,
                    base_size=str(quantity),
                    limit_price=limit_price,
                )
            else:
                resp = self.client.market_order_sell(
                    client_order_id=client_order_id,
                    product_id=product_id,
                    base_size=str(quantity),
                )

            order_id = getattr(resp, "order_id", client_order_id)
            total = quantity * price
            logger.info(f"LIVE SELL order placed: {quantity} {asset} ~${total:.2f} (order: {order_id})")

            trade = {
                "product_id": product_id,
                "side": "sell",
                "order_type": "limit" if self.prefer_limit else "market",
                "price": price,
                "quantity": quantity,
                "total": total,
                "fee": 0,
                "timestamp": now,
                "signal_id": signal_id,
                "execution_mode": "live",
                "order_id": order_id,
                "status": "pending",
            }
            trade_id = self.db.insert_trade(trade)
            return {**trade, "id": trade_id}

        except Exception as e:
            logger.error(f"Failed to place sell order for {product_id}: {e}")
            return None
