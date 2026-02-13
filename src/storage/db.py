"""SQLite storage layer — schema, CRUD operations, and helpers."""

import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy types that json.dumps can't serialize."""
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS candles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_id TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    UNIQUE(product_id, timeframe, timestamp)
                );

                CREATE INDEX IF NOT EXISTS idx_candles_lookup
                    ON candles(product_id, timeframe, timestamp);

                CREATE TABLE IF NOT EXISTS social_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    galaxy_score REAL,
                    alt_rank INTEGER,
                    social_volume REAL,
                    social_dominance REAL,
                    sentiment REAL,
                    market_cap REAL,
                    price REAL,
                    raw_json TEXT,
                    UNIQUE(asset, timestamp)
                );

                CREATE INDEX IF NOT EXISTS idx_social_lookup
                    ON social_data(asset, timestamp);

                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_id TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    strategy TEXT NOT NULL,
                    action TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reasoning TEXT,
                    metadata TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_signals_lookup
                    ON signals(product_id, timestamp);

                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_id TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    total REAL NOT NULL,
                    fee REAL DEFAULT 0,
                    timestamp INTEGER NOT NULL,
                    signal_id INTEGER,
                    execution_mode TEXT NOT NULL DEFAULT 'paper',
                    order_id TEXT,
                    status TEXT NOT NULL DEFAULT 'filled',
                    FOREIGN KEY (signal_id) REFERENCES signals(id)
                );

                CREATE TABLE IF NOT EXISTS portfolio (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset TEXT NOT NULL UNIQUE,
                    quantity REAL NOT NULL DEFAULT 0,
                    avg_entry_price REAL NOT NULL DEFAULT 0,
                    current_price REAL DEFAULT 0,
                    unrealized_pnl REAL DEFAULT 0,
                    realized_pnl REAL DEFAULT 0,
                    updated_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS daily_pnl (
                    date TEXT PRIMARY KEY,
                    starting_balance REAL NOT NULL,
                    ending_balance REAL,
                    realized_pnl REAL DEFAULT 0,
                    trade_count INTEGER DEFAULT 0
                );
            """)

    # --- Candles ---

    def upsert_candles(self, product_id: str, timeframe: str, candles: list[dict]):
        with self._conn() as conn:
            conn.executemany(
                """INSERT INTO candles (product_id, timeframe, timestamp, open, high, low, close, volume)
                   VALUES (:product_id, :timeframe, :timestamp, :open, :high, :low, :close, :volume)
                   ON CONFLICT(product_id, timeframe, timestamp) DO UPDATE SET
                       open=excluded.open, high=excluded.high, low=excluded.low,
                       close=excluded.close, volume=excluded.volume""",
                [{"product_id": product_id, "timeframe": timeframe, **c} for c in candles],
            )

    def get_candles(
        self, product_id: str, timeframe: str, limit: int = 200, since: int | None = None
    ) -> list[dict]:
        with self._conn() as conn:
            if since:
                rows = conn.execute(
                    """SELECT * FROM candles
                       WHERE product_id=? AND timeframe=? AND timestamp>=?
                       ORDER BY timestamp ASC LIMIT ?""",
                    (product_id, timeframe, since, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM candles
                       WHERE product_id=? AND timeframe=?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (product_id, timeframe, limit),
                ).fetchall()
                rows = list(reversed(rows))
            return [dict(r) for r in rows]

    # --- Social Data ---

    def upsert_social_data(self, records: list[dict]):
        with self._conn() as conn:
            conn.executemany(
                """INSERT INTO social_data
                   (asset, timestamp, galaxy_score, alt_rank, social_volume,
                    social_dominance, sentiment, market_cap, price, raw_json)
                   VALUES (:asset, :timestamp, :galaxy_score, :alt_rank, :social_volume,
                           :social_dominance, :sentiment, :market_cap, :price, :raw_json)
                   ON CONFLICT(asset, timestamp) DO UPDATE SET
                       galaxy_score=excluded.galaxy_score, alt_rank=excluded.alt_rank,
                       social_volume=excluded.social_volume, social_dominance=excluded.social_dominance,
                       sentiment=excluded.sentiment, market_cap=excluded.market_cap,
                       price=excluded.price, raw_json=excluded.raw_json""",
                records,
            )

    def get_social_data(self, asset: str, limit: int = 100, since: int | None = None) -> list[dict]:
        with self._conn() as conn:
            if since:
                rows = conn.execute(
                    """SELECT * FROM social_data WHERE asset=? AND timestamp>=?
                       ORDER BY timestamp ASC LIMIT ?""",
                    (asset, since, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM social_data WHERE asset=?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (asset, limit),
                ).fetchall()
                rows = list(reversed(rows))
            return [dict(r) for r in rows]

    # --- Signals ---

    def insert_signal(self, signal: dict) -> int:
        if "metadata" in signal and isinstance(signal["metadata"], dict):
            signal = {**signal, "metadata": json.dumps(signal["metadata"], cls=_NumpyEncoder)}
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT INTO signals (product_id, timestamp, strategy, action, confidence, reasoning, metadata)
                   VALUES (:product_id, :timestamp, :strategy, :action, :confidence, :reasoning, :metadata)""",
                signal,
            )
            return cursor.lastrowid

    def get_signals(self, product_id: str | None = None, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            if product_id:
                rows = conn.execute(
                    "SELECT * FROM signals WHERE product_id=? ORDER BY timestamp DESC LIMIT ?",
                    (product_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?", (limit,)
                ).fetchall()
            return [dict(r) for r in rows]

    # --- Trades ---

    def insert_trade(self, trade: dict) -> int:
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT INTO trades
                   (product_id, side, order_type, price, quantity, total, fee,
                    timestamp, signal_id, execution_mode, order_id, status)
                   VALUES (:product_id, :side, :order_type, :price, :quantity, :total, :fee,
                           :timestamp, :signal_id, :execution_mode, :order_id, :status)""",
                trade,
            )
            return cursor.lastrowid

    def get_trades(
        self, product_id: str | None = None, execution_mode: str | None = None, limit: int = 50
    ) -> list[dict]:
        with self._conn() as conn:
            query = "SELECT * FROM trades WHERE 1=1"
            params: list = []
            if product_id:
                query += " AND product_id=?"
                params.append(product_id)
            if execution_mode:
                query += " AND execution_mode=?"
                params.append(execution_mode)
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            return [dict(r) for r in conn.execute(query, params).fetchall()]

    def get_trades_since(self, since_timestamp: int, execution_mode: str = "paper") -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM trades WHERE timestamp>=? AND execution_mode=?
                   ORDER BY timestamp ASC""",
                (since_timestamp, execution_mode),
            ).fetchall()
            return [dict(r) for r in rows]

    # --- Portfolio ---

    def upsert_portfolio(self, asset: str, quantity: float, avg_entry_price: float,
                         current_price: float = 0, unrealized_pnl: float = 0,
                         realized_pnl: float = 0):
        now = int(datetime.now(timezone.utc).timestamp())
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO portfolio (asset, quantity, avg_entry_price, current_price,
                                          unrealized_pnl, realized_pnl, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(asset) DO UPDATE SET
                       quantity=excluded.quantity, avg_entry_price=excluded.avg_entry_price,
                       current_price=excluded.current_price, unrealized_pnl=excluded.unrealized_pnl,
                       realized_pnl=excluded.realized_pnl, updated_at=excluded.updated_at""",
                (asset, quantity, avg_entry_price, current_price, unrealized_pnl, realized_pnl, now),
            )

    def get_portfolio(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM portfolio WHERE quantity > 0 ORDER BY asset"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_portfolio_asset(self, asset: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM portfolio WHERE asset=?", (asset,)).fetchone()
            return dict(row) if row else None

    # --- Daily P&L ---

    def record_daily_pnl(self, date: str, starting_balance: float, ending_balance: float | None = None,
                         realized_pnl: float = 0, trade_count: int = 0):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO daily_pnl (date, starting_balance, ending_balance, realized_pnl, trade_count)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(date) DO UPDATE SET
                       ending_balance=excluded.ending_balance, realized_pnl=excluded.realized_pnl,
                       trade_count=excluded.trade_count""",
                (date, starting_balance, ending_balance, realized_pnl, trade_count),
            )

    def get_daily_pnl(self, limit: int = 30) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM daily_pnl ORDER BY date DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
