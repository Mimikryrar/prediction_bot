from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib


DEFAULT_STORE_DIR = Path("models")


def _metadata_path(artifact_path: Path) -> Path:
    return artifact_path.with_suffix(".json")


def save_model(
    model: Any,
    artifact_path: Path,
    *,
    model_version: str,
    symbol_universe: List[str],
    feature_names: List[str],
    train_start: date,
    train_end: date,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path)

    metadata: Dict[str, Any] = {
        "model_version": model_version,
        "symbol_universe": symbol_universe,
        "feature_names": feature_names,
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
    }
    if extra:
        metadata.update(extra)

    meta_path = _metadata_path(artifact_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_model(artifact_path: Path) -> tuple[Any, Dict[str, Any]]:
    artifact_path = Path(artifact_path)
    model = joblib.load(artifact_path)
    meta_path = _metadata_path(artifact_path)
    with open(meta_path) as f:
        metadata = json.load(f)
    return model, metadata
