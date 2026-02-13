"""Market data ingestion — Coinbase Advanced Trade API for OHLCV candles and account info."""

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import websockets
from coinbase.rest import RESTClient

logger = logging.getLogger(__name__)

# Coinbase granularity mapping
TIMEFRAME_MAP = {
    "1m": "ONE_MINUTE",
    "5m": "FIVE_MINUTE",
    "15m": "FIFTEEN_MINUTE",
    "30m": "THIRTY_MINUTE",
    "1h": "ONE_HOUR",
    "2h": "TWO_HOUR",
    "6h": "SIX_HOUR",
    "1d": "ONE_DAY",
}

TIMEFRAME_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "1d": 86400,
}


def create_coinbase_client(api_key: str = "", api_secret: str = "",
                           key_file: str = "") -> RESTClient | None:
    """Create a Coinbase RESTClient using key_file or api_key+api_secret.

    Prefers key_file (CDP JSON with ECDSA P-256 PEM key) when available.
    The key file must have {"name": "...", "privateKey": "-----BEGIN EC PRIVATE KEY-----..."}.
    Generate ECDSA keys at https://portal.cdp.coinbase.com (select ECDSA, not Ed25519).
    Returns None if no credentials are available.
    """
    try:
        if key_file and Path(key_file).exists():
            return RESTClient(key_file=key_file)
        elif api_key and api_secret:
            return RESTClient(api_key=api_key, api_secret=api_secret)
        else:
            logger.warning("No Coinbase credentials configured — market data unavailable")
            return None
    except Exception as e:
        logger.error(f"Failed to create Coinbase client: {e}")
        return None


class MarketDataClient:
    """Fetches OHLCV candles and account info from Coinbase Advanced Trade."""

    def __init__(self, api_key: str = "", api_secret: str = "", key_file: str = ""):
        self.client = create_coinbase_client(api_key, api_secret, key_file)

    def get_accounts(self) -> list[dict]:
        """Fetch all accounts with balances."""
        if not self.client:
            return []
        resp = self.client.get_accounts()
        accounts = []
        for acct in resp.accounts or []:
            bal = float((acct.available_balance or {}).get("value", 0))
            if bal > 0:
                accounts.append({
                    "currency": acct.currency,
                    "available": bal,
                    "hold": float((acct.hold or {}).get("value", 0)),
                })
        return accounts

    def get_product(self, product_id: str) -> dict:
        """Get current product ticker info."""
        if not self.client:
            return {"product_id": product_id, "price": 0, "bid": 0, "volume_24h": 0}
        resp = self.client.get_product(product_id)
        return {
            "product_id": product_id,
            "price": float(resp.price or 0),
            "bid": float(resp.quote_min_size or 0),
            "volume_24h": float(resp.volume_24h or 0),
        }

    def get_candles(self, product_id: str, timeframe: str = "1h",
                    limit: int = 300) -> list[dict]:
        """Fetch OHLCV candles from Coinbase.

        Returns list of dicts with keys: timestamp, open, high, low, close, volume
        sorted ascending by timestamp.
        """
        if not self.client:
            return []
        granularity = TIMEFRAME_MAP.get(timeframe)
        if not granularity:
            # For 4h, fetch 1h candles and aggregate
            if timeframe == "4h":
                return self._aggregate_candles(product_id, "1h", 4, limit)
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        now = int(datetime.now(timezone.utc).timestamp())
        seconds = TIMEFRAME_SECONDS.get(timeframe, 3600)
        start = now - (seconds * limit)

        resp = self.client.get_candles(
            product_id=product_id,
            start=str(start),
            end=str(now),
            granularity=granularity,
        )

        candles = []
        for c in resp.candles or []:
            candles.append({
                "timestamp": int(c.start),
                "open": float(c.open),
                "high": float(c.high),
                "low": float(c.low),
                "close": float(c.close),
                "volume": float(c.volume),
            })

        candles.sort(key=lambda x: x["timestamp"])
        return candles

    def _aggregate_candles(self, product_id: str, base_tf: str,
                           factor: int, limit: int) -> list[dict]:
        """Aggregate smaller candles into larger timeframe."""
        raw = self.get_candles(product_id, base_tf, limit=min(limit * factor, 300))
        aggregated = []
        for i in range(0, len(raw) - factor + 1, factor):
            chunk = raw[i:i + factor]
            aggregated.append({
                "timestamp": chunk[0]["timestamp"],
                "open": chunk[0]["open"],
                "high": max(c["high"] for c in chunk),
                "low": min(c["low"] for c in chunk),
                "close": chunk[-1]["close"],
                "volume": sum(c["volume"] for c in chunk),
            })
        return aggregated


class MarketWebSocket:
    """Real-time price updates via Coinbase WebSocket."""

    WS_URL = "wss://advanced-trade-ws.coinbase.com"

    def __init__(self, product_ids: list[str], on_price_update=None):
        self.product_ids = product_ids
        self.on_price_update = on_price_update
        self._running = False

    async def connect(self):
        """Connect and stream ticker updates."""
        self._running = True
        subscribe_msg = {
            "type": "subscribe",
            "product_ids": self.product_ids,
            "channel": "ticker",
        }

        while self._running:
            try:
                async with websockets.connect(self.WS_URL) as ws:
                    await ws.send(json.dumps(subscribe_msg))
                    logger.info(f"WebSocket connected for {self.product_ids}")

                    async for message in ws:
                        if not self._running:
                            break
                        data = json.loads(message)
                        if data.get("channel") == "ticker":
                            for event in data.get("events", []):
                                for ticker in event.get("tickers", []):
                                    update = {
                                        "product_id": ticker.get("product_id"),
                                        "price": float(ticker.get("price", 0)),
                                        "volume_24h": float(ticker.get("volume_24_h", 0)),
                                        "timestamp": int(time.time()),
                                    }
                                    if self.on_price_update:
                                        self.on_price_update(update)
            except websockets.ConnectionClosed:
                logger.warning("WebSocket disconnected, reconnecting in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"WebSocket error: {e}, reconnecting in 10s...")
                await asyncio.sleep(10)

    def stop(self):
        self._running = False
