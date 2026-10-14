import yaml
from pathlib import Path
from functools import lru_cache

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "ui_text.yaml"

@lru_cache(maxsize=1)
def load_all_ui_text():
    """Loads all UI text strings from the external configuration file."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

def get_ui_text(*keys, default=""):
    """
    Retrieves nested UI string from configuration.
    Example: get_ui_text("chat", "hero_title")
    """
    data = load_all_ui_text()
    for k in keys:
        if isinstance(data, dict) and k in data:
            data = data[k]
        else:
            return default
    return data
