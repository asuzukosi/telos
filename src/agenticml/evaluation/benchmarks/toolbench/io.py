"""toolbench result rows and persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agenticml.evaluation.benchmarks.common import model_dir_name, repo_root
from agenticml.evaluation.harness.backends.agenticml_backend import BackendRunResult


def default_result_dir() -> Path:
    return repo_root() / "results" / "benchmarks" / "toolbench"


def result_row(
    entry_id: str,
    run: BackendRunResult,
    *,
    success: bool,
    group: str = "G1_instruction",
    query: str = "",
    available_tools: list[dict[str, Any]] | None = None,
    messages: list[dict[str, Any]] | None = None,
    fmt: str = "agenticml",
) -> dict[str, Any]:
    return {
        "id": entry_id,
        "group": group,
        "format": fmt,
        "success": success,
        "query": query,
        "available_tools": available_tools or [],
        "messages": messages or [],
        "final_answer": run.final_answer,
        "steps": run.iterations,
        "stopped_on": run.stopped_on,
        "input_token_count": run.prompt_tokens,
        "output_token_count": run.generated_tokens,
        "latency": run.inference_sec,
        "tool_sec": run.tool_sec,
        "total_sec": run.total_sec,
    }


def write_results(result_dir: Path, model_id: str, rows: list[dict[str, Any]]) -> None:
    root = result_dir / model_dir_name(model_id)
    root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = root / f"{row['id']}.json"
        path.write_text(json.dumps(row, indent=2))


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
        if wanted_ids is not None and str(row.get("id")) not in wanted_ids:
            continue
        out.append(row)
    return out
