"""Web dashboard — FastAPI app with htmx-powered live updates."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.storage.db import Database
from src.config import get_config

_WEB_DIR = Path(__file__).parent
app = FastAPI(title="Murmur Dashboard")
app.mount("/static", StaticFiles(directory=_WEB_DIR / "static"), name="static")
templates = Jinja2Templates(directory=_WEB_DIR / "templates")


@app.middleware("http")
async def ip_whitelist(request: Request, call_next):
    """Block requests from IPs not in ALLOWED_IPS (comma-separated).

    If ALLOWED_IPS is not set, all IPs are allowed.
    """
    allowed = os.getenv("ALLOWED_IPS", "")
    if allowed:
        allowed_set = {ip.strip() for ip in allowed.split(",")}
        client_ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        if not client_ip:
            client_ip = request.client.host
        if client_ip not in allowed_set:
            return PlainTextResponse("Forbidden", status_code=403)
    return await call_next(request)


def _get_db() -> Database:
    config = get_config()
    return Database(config["database"]["path"])


def _ts_to_str(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%m-%d %H:%M")


# --- Full page ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    db = _get_db()
    config = get_config()
    portfolio = db.get_portfolio()
    cash = 0.0
    positions_value = 0.0
    positions = []

    for p in portfolio:
        if p["asset"] == "USD":
            cash = p["quantity"]
        elif p["quantity"] > 0:
            value = p["quantity"] * p["current_price"]
            entry = p["avg_entry_price"]
            current = p["current_price"]
            pnl = (current - entry) * p["quantity"]
            pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
            positions_value += value
            positions.append({
                **p,
                "value": value,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            })

    total = cash + positions_value
    signals = db.get_signals(limit=20)
    for s in signals:
        s["time_str"] = _ts_to_str(s["timestamp"])

    trades = db.get_trades(limit=20)
    for t in trades:
        t["time_str"] = _ts_to_str(t["timestamp"])

    daily = db.get_daily_pnl(limit=14)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "cash": cash,
        "positions_value": positions_value,
        "total": total,
        "positions": positions,
        "signals": signals,
        "trades": trades,
        "daily_pnl": daily,
        "watchlist": config.get("watchlist", []),
        "mode": config.get("execution", {}).get("mode", "paper"),
    })


# --- htmx partials ---

@app.get("/partials/portfolio", response_class=HTMLResponse)
async def partial_portfolio(request: Request):
    db = _get_db()
    portfolio = db.get_portfolio()
    cash = 0.0
    positions_value = 0.0
    positions = []

    for p in portfolio:
        if p["asset"] == "USD":
            cash = p["quantity"]
        elif p["quantity"] > 0:
            value = p["quantity"] * p["current_price"]
            entry = p["avg_entry_price"]
            current = p["current_price"]
            pnl = (current - entry) * p["quantity"]
            pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
            positions_value += value
            positions.append({
                **p,
                "value": value,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            })

    total = cash + positions_value
    return templates.TemplateResponse("partials/portfolio.html", {
        "request": request,
        "cash": cash,
        "positions_value": positions_value,
        "total": total,
        "positions": positions,
    })


@app.get("/partials/signals", response_class=HTMLResponse)
async def partial_signals(request: Request):
    db = _get_db()
    signals = db.get_signals(limit=20)
    for s in signals:
        s["time_str"] = _ts_to_str(s["timestamp"])
    return templates.TemplateResponse("partials/signals.html", {
        "request": request,
        "signals": signals,
    })


@app.get("/partials/trades", response_class=HTMLResponse)
async def partial_trades(request: Request):
    db = _get_db()
    trades = db.get_trades(limit=20)
    for t in trades:
        t["time_str"] = _ts_to_str(t["timestamp"])
    return templates.TemplateResponse("partials/trades.html", {
        "request": request,
        "trades": trades,
    })


# --- JSON API for charts ---

@app.get("/api/candles/{product_id}")
async def api_candles(product_id: str, timeframe: str = "1h", limit: int = 100):
    db = _get_db()
    candles = db.get_candles(product_id, timeframe, limit=limit)
    return {
        "product_id": product_id,
        "timeframe": timeframe,
        "candles": [
            {
                "time": c["timestamp"],
                "open": c["open"],
                "high": c["high"],
                "low": c["low"],
                "close": c["close"],
                "volume": c["volume"],
            }
            for c in candles
        ],
    }


@app.get("/api/portfolio-history")
async def api_portfolio_history():
    db = _get_db()
    daily = db.get_daily_pnl(limit=30)
    daily.reverse()
    return {
        "dates": [d["date"] for d in daily],
        "balances": [d.get("ending_balance") or d["starting_balance"] for d in daily],
        "pnl": [d.get("realized_pnl", 0) or 0 for d in daily],
    }


@app.get("/api/signals/recent")
async def api_recent_signals(limit: int = 50):
    db = _get_db()
    signals = db.get_signals(limit=limit)
    for s in signals:
        s["time_str"] = _ts_to_str(s["timestamp"])
        # Parse metadata if it's a JSON string
        if isinstance(s.get("metadata"), str):
            try:
                s["metadata"] = json.loads(s["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass
    return {"signals": signals}


def create_app(config_path: str | None = None) -> FastAPI:
    """Factory function for creating the app with specific config."""
    if config_path:
        get_config(config_path)
    return app
