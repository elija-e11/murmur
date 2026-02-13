"""Murmur — Crypto Social Trading Bot entry point."""

import argparse
import asyncio
import logging
import os
import signal
import sys
import threading
from datetime import datetime, timezone

import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler

from src.config import get_config
from src.storage.db import Database
from src.ingestion.market import MarketDataClient
from src.ingestion.social import SocialAggregator
from src.analysis.technical import TechnicalAnalyzer
from src.analysis.sentiment import SentimentAnalyzer
from src.strategy.engine import StrategyEngine
from src.execution.paper import PaperTrader
from src.execution.trader import LiveTrader
from src.monitor.dashboard import Dashboard
from src.web.app import create_app

logger = logging.getLogger("murmur")


class Murmur:
    """Main application orchestrator."""

    def __init__(self, config_path: str | None = None):
        self.config = get_config(config_path)
        self._setup_logging()

        # Initialize components
        self.db = Database(self.config["database"]["path"])
        secrets = self.config["secrets"]

        cb_key_file = secrets.get("coinbase_key_file", "")
        self.market = MarketDataClient(
            api_key=secrets["coinbase_api_key"],
            api_secret=secrets["coinbase_api_secret"],
            key_file=cb_key_file,
        )
        self.social = SocialAggregator(self.config)
        self.tech_analyzer = TechnicalAnalyzer(self.config)
        self.sentiment_analyzer = SentimentAnalyzer(self.config)
        self.engine = StrategyEngine(self.config)
        self.dashboard = Dashboard(self.db)

        # Execution mode
        mode = self.config.get("execution", {}).get("mode", "paper")
        if mode == "live":
            self.trader = LiveTrader(
                self.db, self.config,
                api_key=secrets["coinbase_api_key"],
                api_secret=secrets["coinbase_api_secret"],
                key_file=cb_key_file,
            )
        else:
            self.trader = PaperTrader(self.db, self.config)

        self.watchlist = self.config.get("watchlist", ["BTC-USD", "ETH-USD"])
        self.scheduler = BackgroundScheduler()
        self._running = False

    def _setup_logging(self):
        log_cfg = self.config.get("logging", {})
        level = getattr(logging, log_cfg.get("level", "INFO"))
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

        logging.basicConfig(level=level, format=fmt)

        log_file = log_cfg.get("file")
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(fmt))
            logging.getLogger().addHandler(fh)

    def fetch_market_data(self):
        """Fetch and store candles for all watchlist assets."""
        timeframes = self.config.get("timeframes", ["1h"])
        for product_id in self.watchlist:
            for tf in timeframes:
                try:
                    candles = self.market.get_candles(product_id, tf, limit=200)
                    self.db.upsert_candles(product_id, tf, candles)
                    logger.debug(f"Stored {len(candles)} {tf} candles for {product_id}")
                except Exception as e:
                    logger.error(f"Failed to fetch candles for {product_id}/{tf}: {e}")

    def fetch_social_data(self):
        """Fetch and store social data for all watchlist assets."""
        try:
            records = self.social.fetch_watchlist_data(self.watchlist)
            if records:
                self.db.upsert_social_data(records)
                logger.info(f"Stored social data for {len(records)} assets")
        except Exception as e:
            logger.error(f"Failed to fetch social data: {e}")

    def run_analysis_cycle(self):
        """Run analysis and strategy evaluation for all watchlist assets."""
        decisions = []
        prices = {}

        for product_id in self.watchlist:
            try:
                # Get candles for primary timeframe (1h)
                candles = self.db.get_candles(product_id, "1h", limit=200)
                if not candles:
                    logger.warning(f"No candle data for {product_id}, skipping")
                    continue

                # Technical analysis
                tech = self.tech_analyzer.compute_all(candles)
                if tech.get("error"):
                    logger.warning(f"Insufficient data for {product_id}: {tech['error']}")
                    continue

                prices[product_id] = candles[-1]["close"]

                # Social/sentiment analysis
                asset = product_id.split("-")[0]
                social_records = self.db.get_social_data(asset, limit=50)
                sentiment = self.sentiment_analyzer.analyze(social_records)

                # Strategy evaluation
                decision = self.engine.evaluate(product_id, tech, sentiment)
                decisions.append(decision)

                # Store the signal
                now = int(datetime.now(timezone.utc).timestamp())
                self.db.insert_signal({
                    "product_id": product_id,
                    "timestamp": now,
                    "strategy": "combined",
                    "action": decision["action"],
                    "confidence": decision["confidence"],
                    "reasoning": decision["reasoning"],
                    "metadata": {"tech": tech, "sentiment": sentiment},
                })

                # Execute if actionable
                if decision["action"] == "buy":
                    if isinstance(self.trader, PaperTrader):
                        self.trader.execute_buy(product_id, prices[product_id])
                    else:
                        self.trader.execute_buy(product_id)
                elif decision["action"] == "sell":
                    if isinstance(self.trader, PaperTrader):
                        self.trader.execute_sell(product_id, prices[product_id])
                    else:
                        self.trader.execute_sell(product_id)

            except Exception as e:
                logger.error(f"Analysis failed for {product_id}: {e}", exc_info=True)

        # Check stop-loss / take-profit on all positions
        if isinstance(self.trader, PaperTrader) and prices:
            self.trader.check_stop_loss_take_profit(prices)

        # Display summary
        if decisions:
            self.dashboard.print_signal_summary(decisions)

    def run(self, web: bool = False, web_port: int = 8877):
        """Start the bot with scheduled tasks."""
        logger.info("Starting Murmur...")
        logger.info(f"Mode: {self.config.get('execution', {}).get('mode', 'paper')}")
        logger.info(f"Watchlist: {', '.join(self.watchlist)}")

        self._running = True

        # Initial data fetch
        logger.info("Fetching initial data...")
        self.fetch_market_data()
        self.fetch_social_data()
        self.run_analysis_cycle()

        # Schedule recurring tasks
        intervals = self.config.get("intervals", {})
        self.scheduler.add_job(
            self.fetch_market_data, "interval",
            seconds=intervals.get("candle_fetch", 300),
            id="market_data",
        )
        self.scheduler.add_job(
            self.fetch_social_data, "interval",
            seconds=intervals.get("social_poll", 600),
            id="social_data",
        )
        self.scheduler.add_job(
            self.run_analysis_cycle, "interval",
            seconds=intervals.get("analysis_cycle", 300),
            id="analysis",
        )

        self.scheduler.start()

        # Start web dashboard if requested
        if web:
            self._start_web_server(web_port)

        logger.info("Scheduler started. Press Ctrl+C to stop.")

        # Keep main thread alive
        try:
            while self._running:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def _start_web_server(self, port: int):
        """Launch the web dashboard in a background thread."""
        web_app = create_app()
        config = uvicorn.Config(web_app, host="0.0.0.0", port=port, log_level="warning")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        logger.info(f"Web dashboard running at http://localhost:{port}")

    def stop(self):
        """Gracefully shut down."""
        logger.info("Shutting down Murmur...")
        self._running = False
        self.scheduler.shutdown(wait=False)
        logger.info("Goodbye.")

    def show_dashboard(self):
        """Display the monitoring dashboard."""
        self.dashboard.render()


def main():
    parser = argparse.ArgumentParser(description="Murmur — Crypto Social Trading Bot")
    parser.add_argument("--config", "-c", help="Path to settings.yaml", default=None)
    parser.add_argument("--dashboard", "-d", action="store_true", help="Show CLI dashboard and exit")
    parser.add_argument("--web", "-w", action="store_true", help="Launch web dashboard (standalone or with bot)")
    parser.add_argument("--port", "-p", type=int, default=int(os.environ.get("PORT", 8877)),
                        help="Web dashboard port (default: 8877, or $PORT)")
    parser.add_argument("--fetch", "-f", action="store_true", help="Fetch data once and exit")
    parser.add_argument("--analyze", "-a", action="store_true", help="Run one analysis cycle and exit")
    parser.add_argument("--web-only", action="store_true", help="Run only the web dashboard (no bot)")
    args = parser.parse_args()

    if args.web_only:
        # Standalone web server — just serves the dashboard against existing DB
        get_config(args.config)
        web_app = create_app(args.config)
        logger.info(f"Starting web dashboard at http://localhost:{args.port}")
        uvicorn.run(web_app, host="0.0.0.0", port=args.port, log_level="info")
        return

    app = Murmur(config_path=args.config)

    if args.dashboard:
        app.show_dashboard()
    elif args.fetch:
        app.fetch_market_data()
        app.fetch_social_data()
        logger.info("Data fetch complete.")
    elif args.analyze:
        app.fetch_market_data()
        app.fetch_social_data()
        app.run_analysis_cycle()
        app.show_dashboard()
    else:
        # Handle graceful shutdown
        def handle_signal(signum, frame):
            app.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
        app.run(web=args.web, web_port=args.port)


if __name__ == "__main__":
    main()
