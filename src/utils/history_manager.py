import json
import uuid
from datetime import datetime
from pathlib import Path
from src.utils.logger import logger

HISTORY_FILE = Path("data/history.json")

def load_history():
    """Loads all history items from data/history.json."""
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading history file: {e}")
        return []

def save_history(history_items):
    """Saves history items array to data/history.json."""
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history_items, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Error saving history file: {e}")
        return False

def add_history_entry(item_type: str, title: str, details: dict, source: str = "All Documents"):
    """
    Appends a new history entry to history.json.
    item_type: 'quiz' | 'flashcard' | 'summary' | 'chat' | 'material'
    """
    history = load_history()
    now_str = datetime.now().strftime("%b %d, %Y - %I:%M %p")
    entry = {
        "id": str(uuid.uuid4()),
        "type": item_type,
        "title": title,
        "source": source,
        "timestamp": now_str,
        "details": details
    }
    history.insert(0, entry)
    save_history(history)
    return entry

def delete_history_entry(entry_id: str):
    """Deletes a single history entry by ID."""
    history = load_history()
    filtered = [h for h in history if h.get("id") != entry_id]
    save_history(filtered)
    return filtered

def clear_all_history():
    """Clears all history entries."""
    save_history([])
    return []
