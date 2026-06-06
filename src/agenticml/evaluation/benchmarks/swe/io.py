"""swe-bench result rows and preds.json entries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agenticml.evaluation.benchmarks.common import model_dir_name, repo_root
from agenticml.evaluation.benchmarks.swe.loop import SweRunResult


def default_result_dir() -> Path:
    return repo_root() / "results" / "benchmarks" / "swe"


def pred_entry(result: SweRunResult, *, model_id: str) -> dict[str, str]:
    """one instance entry for swebench preds.json / preds.jsonl."""
    return pred_from_row(result.to_row(), model_id=model_id)


def pred_from_row(row: dict[str, Any], *, model_id: str) -> dict[str, str]:
    return {
        "model_name_or_path": model_id,
        "instance_id": str(row["instance_id"]),
        "model_patch": row.get("model_patch") or "",
    }


def write_preds(path: Path, rows: list[dict[str, Any]], *, model_id: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    preds = {str(r["instance_id"]): pred_from_row(r, model_id=model_id) for r in rows}
    path.write_text(json.dumps(preds, indent=2) + "\n")
    return path


def write_results(result_dir: Path, model_id: str, rows: list[dict[str, Any]]) -> None:
    root = result_dir / model_dir_name(model_id)
    root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        iid = str(row["instance_id"])
        (root / f"{iid}.json").write_text(json.dumps(row, indent=2) + "\n")


def load_result_rows(
    result_dir: Path,
    model_id: str,
    *,
    wanted_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    root = result_dir / model_dir_name(model_id)
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        row = json.loads(path.read_text())
        iid = str(row.get("instance_id") or path.stem)
        if wanted_ids is not None and iid not in wanted_ids:
            continue
        out.append(row)
    return out


def result_row(
    result: SweRunResult,
    *,
    model_id: str,
    fmt: str = "agenticml",
) -> dict[str, Any]:
    row = result.to_row()
    row.update(
        {
            "format": fmt,
            "model_id": model_id,
            "success": result.stopped_on == "submitted" and bool(result.model_patch),
        }
    )
    return row
