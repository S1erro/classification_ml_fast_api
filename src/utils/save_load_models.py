import json
from typing import Any, Optional

import joblib

from ..types.saved_model import SavedModel, SavedModelToHistory

def save_model(model: SavedModel, path: str) -> None:
    joblib.dump(model, path)

def add_to_history(model: SavedModelToHistory, path: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(model.to_dict(), ensure_ascii=False) + "\n")

def load_model(path: str) -> SavedModel:
    model = joblib.load(path)
    return model

def load_from_history(path: str) -> Optional[SavedModelToHistory]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            last_valid_raw: Optional[dict[str, Any]] = None

            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if isinstance(item, dict):
                    last_valid_raw = item

        if last_valid_raw is None:
            return None

        return SavedModelToHistory(**last_valid_raw)
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return None