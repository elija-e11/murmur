"""Configuration management — loads settings.yaml + .env variables."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _ROOT / "config"


def load_config(config_path: str | None = None) -> dict:
    """Load and merge YAML config with environment variable overrides."""
    # Load .env file
    env_path = _CONFIG_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # Load YAML settings
    yaml_path = Path(config_path) if config_path else _CONFIG_DIR / "settings.yaml"
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    # Inject secrets from environment
    config["secrets"] = {
        "coinbase_api_key": os.getenv("COINBASE_API_KEY", ""),
        "coinbase_api_secret": os.getenv("COINBASE_API_SECRET", ""),
        "coinbase_key_file": os.getenv("COINBASE_KEY_FILE", ""),
        "reddit_client_id": os.getenv("REDDIT_CLIENT_ID", ""),
        "reddit_client_secret": os.getenv("REDDIT_CLIENT_SECRET", ""),
        "reddit_user_agent": os.getenv("REDDIT_USER_AGENT", "murmur-bot/1.0"),
        "bluesky_handle": os.getenv("BLUESKY_HANDLE", ""),
        "bluesky_app_password": os.getenv("BLUESKY_APP_PASSWORD", ""),
        "cryptopanic_api_key": os.getenv("CRYPTOPANIC_API_KEY", ""),
        "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
    }

    # Database path: env var override (for Railway volume) or config-relative
    db_path = os.getenv("DATABASE_PATH") or config.get("database", {}).get("path", "murmur.db")
    if not os.path.isabs(db_path):
        db_path = str(_ROOT / db_path)
    config.setdefault("database", {})["path"] = db_path

    return config


# Singleton config — loaded once, imported everywhere
_config: dict | None = None


def get_config(config_path: str | None = None) -> dict:
    """Get or initialize the global config."""
    global _config
    if _config is None:
        _config = load_config(config_path)
    return _config
