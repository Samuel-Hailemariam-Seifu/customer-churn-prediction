import json
from pathlib import Path
from typing import Any, Dict

import joblib


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_joblib(obj: Any, path: Path) -> None:
    ensure_parent(path)
    joblib.dump(obj, path)


def load_joblib(path: Path) -> Any:
    return joblib.load(path)


def save_json(payload: Dict[str, Any], path: Path) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
