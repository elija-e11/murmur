"""CLI monitoring dashboard — Rich-based terminal display for portfolio status and signals."""

import logging
from datetime import datetime, timezone

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.storage.db import Database

logger = logging.getLogger(__name__)
console = Console()


class Dashboard:
    """Terminal-based monitoring dashboard."""

    def __init__(self, db: Database):
        self.db = db

    def render(self):
        """Render full dashboard to terminal."""
        console.clear()
        console.print(Panel("[bold cyan]MURMUR[/bold cyan] — Crypto Social Trading Bot", style="bold"))
        console.print()

        self._render_portfolio()
        console.print()
        self._render_positions()
        console.print()
        self._render_recent_signals()
        console.print()
        self._render_recent_trades()
        console.print()
        self._render_daily_pnl()

    def _render_portfolio(self):
        """Show portfolio summary."""
        portfolio = self.db.get_portfolio()
        cash = 0
        positions_value = 0

        for p in portfolio:
            if p["asset"] == "USD":
                cash = p["quantity"]
            else:
                positions_value += p["quantity"] * p["current_price"]

        total = cash + positions_value
        table = Table(title="Portfolio Summary", show_header=False, padding=(0, 2))
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("Cash (USD)", f"${cash:,.2f}")
        table.add_row("Positions Value", f"${positions_value:,.2f}")
        table.add_row("Total Value", f"[bold]${total:,.2f}[/bold]")

        console.print(table)

    def _render_positions(self):
        """Show open positions."""
        positions = [p for p in self.db.get_portfolio() if p["asset"] != "USD" and p["quantity"] > 0]

        if not positions:
            console.print("[dim]No open positions[/dim]")
            return

        table = Table(title="Open Positions")
        table.add_column("Asset", style="bold")
        table.add_column("Qty", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("P&L %", justify="right")

        for p in positions:
            entry = p["avg_entry_price"]
            current = p["current_price"]
            pnl = (current - entry) * p["quantity"]
            pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0

            pnl_style = "green" if pnl >= 0 else "red"
            table.add_row(
                p["asset"],
                f"{p['quantity']:.6f}",
                f"${entry:,.2f}",
                f"${current:,.2f}",
                f"[{pnl_style}]${pnl:+,.2f}[/{pnl_style}]",
                f"[{pnl_style}]{pnl_pct:+.1f}%[/{pnl_style}]",
            )

        console.print(table)

    def _render_recent_signals(self, limit: int = 10):
        """Show recent signals."""
        signals = self.db.get_signals(limit=limit)

        if not signals:
            console.print("[dim]No signals yet[/dim]")
            return

        table = Table(title="Recent Signals")
        table.add_column("Time", style="dim")
        table.add_column("Asset", style="bold")
        table.add_column("Strategy")
        table.add_column("Action")
        table.add_column("Confidence", justify="right")
        table.add_column("Reasoning")

        for s in signals:
            ts = datetime.fromtimestamp(s["timestamp"], tz=timezone.utc).strftime("%m-%d %H:%M")
            action = s["action"]
            action_style = {"buy": "green", "sell": "red", "hold": "dim"}.get(action, "white")
            conf = s["confidence"]
            conf_str = f"{conf:.0%}"

            table.add_row(
                ts,
                s["product_id"],
                s["strategy"],
                f"[{action_style}]{action.upper()}[/{action_style}]",
                conf_str,
                s.get("reasoning", "")[:60],
            )

        console.print(table)

    def _render_recent_trades(self, limit: int = 10):
        """Show recent trades."""
        trades = self.db.get_trades(limit=limit)

        if not trades:
            console.print("[dim]No trades yet[/dim]")
            return

        table = Table(title="Recent Trades")
        table.add_column("Time", style="dim")
        table.add_column("Asset", style="bold")
        table.add_column("Side")
        table.add_column("Price", justify="right")
        table.add_column("Qty", justify="right")
        table.add_column("Total", justify="right")
        table.add_column("Mode")

        for t in trades:
            ts = datetime.fromtimestamp(t["timestamp"], tz=timezone.utc).strftime("%m-%d %H:%M")
            side_style = "green" if t["side"] == "buy" else "red"
            table.add_row(
                ts,
                t["product_id"],
                f"[{side_style}]{t['side'].upper()}[/{side_style}]",
                f"${t['price']:,.2f}",
                f"{t['quantity']:.6f}",
                f"${t['total']:,.2f}",
                t["execution_mode"],
            )

        console.print(table)

    def _render_daily_pnl(self, limit: int = 7):
        """Show daily P&L."""
        daily = self.db.get_daily_pnl(limit=limit)

        if not daily:
            return

        table = Table(title="Daily P&L")
        table.add_column("Date")
        table.add_column("P&L", justify="right")
        table.add_column("Trades", justify="right")

        for d in daily:
            pnl = d.get("realized_pnl", 0) or 0
            pnl_style = "green" if pnl >= 0 else "red"
            table.add_row(
                d["date"],
                f"[{pnl_style}]${pnl:+,.2f}[/{pnl_style}]",
                str(d.get("trade_count", 0)),
            )

        console.print(table)

    def print_signal_summary(self, decisions: list[dict]):
        """Print a compact summary of current analysis cycle decisions."""
        console.print(f"\n[bold]Analysis Cycle[/bold] — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

        for d in decisions:
            action = d["action"]
            style = {"buy": "bold green", "sell": "bold red", "hold": "dim"}.get(action, "white")
            conf = d.get("confidence", 0)
            console.print(
                f"  [{style}]{d['product_id']:>10} → {action.upper():4}[/{style}] "
                f"(conf: {conf:.0%}) {d.get('reasoning', '')[:80]}"
            )
