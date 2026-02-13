"""Fear & Greed Index — market-wide crypto sentiment from alternative.me."""

import logging

import requests

logger = logging.getLogger(__name__)


class FearGreedSource:
    """Fetches the Crypto Fear & Greed Index."""

    URL = "https://api.alternative.me/fng/"

    def __init__(self):
        self.session = requests.Session()

    def get_current(self) -> dict:
        """Get current Fear & Greed Index value.

        Returns:
            {
                "value": int,                  # 0 (extreme fear) to 100 (extreme greed)
                "classification": str,         # e.g., "Fear", "Greed", "Neutral"
                "normalized_score": float,     # -1 to +1
            }
        """
        try:
            resp = self.session.get(self.URL, params={"limit": 1}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            entry = data.get("data", [{}])[0]
            value = int(entry.get("value", 50))
            classification = entry.get("value_classification", "Neutral")

            return {
                "value": value,
                "classification": classification,
                "normalized_score": (value - 50) / 50,  # 0-100 → -1 to +1
            }
        except Exception as e:
            logger.error(f"Fear & Greed API error: {e}")
            return {"value": 50, "classification": "Neutral", "normalized_score": 0}

    def get_history(self, days: int = 30) -> list[dict]:
        """Get historical Fear & Greed values.

        Returns list of {"value": int, "timestamp": int} sorted ascending.
        """
        try:
            resp = self.session.get(self.URL, params={"limit": days}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            entries = data.get("data", [])
            result = [
                {
                    "value": int(e.get("value", 50)),
                    "timestamp": int(e.get("timestamp", 0)),
                    "classification": e.get("value_classification", ""),
                }
                for e in entries
            ]
            result.reverse()  # API returns newest first
            return result
        except Exception as e:
            logger.error(f"Fear & Greed history error: {e}")
            return []
