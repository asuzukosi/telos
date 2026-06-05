"""shared benchmark helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def model_dir_name(model_id: str) -> str:
    return model_id.replace("/", "_")


def sample_entries(
    entries: list[dict[str, Any]],
    num_examples: Optional[int],
    *,
    seed: int = 42,
) -> list[dict[str, Any]]:
    if num_examples is None or num_examples >= len(entries):
        return list(entries)
    rng = random.Random(seed)
    picked = rng.sample(entries, num_examples)
    return sorted(picked, key=lambda e: str(e.get("id", e.get("query_id", ""))))
