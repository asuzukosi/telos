"""tests for benchmark results aggregation."""

from __future__ import annotations

import json
from pathlib import Path

from agenticml.evaluation.benchmarks.aggregate_results import (
    aggregate_results,
    load_result_rows,
    render_markdown,
    row_from_summary,
)
from agenticml.evaluation.benchmarks.format_validity.evaluate import (
    ValidityResult,
    suite_metrics,
    task_result,
)
from agenticml.evaluation.harness.task import TaskResult


def _write_summary(path: Path, *, suite: str, fmt: str, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "meta": {
                    "suite": suite,
                    "model": f"org/{fmt}-model",
                    "format": fmt,
                    "num_run": 2,
                    "sample_seed": 42,
                },
                "metrics": {"n": 2, **metrics},
                "tasks": [],
            }
        )
    )


def test_row_from_summary_primary_fallback(tmp_path: Path):
    path = tmp_path / "bfcl" / "agenticml" / "summary.json"
    _write_summary(
        path,
        suite="bfcl",
        fmt="agenticml",
        metrics={"accuracy": 0.6, "avg_retry_count": 3.5, "avg_total_tokens": 1000.0},
    )
    row = row_from_summary("bfcl", "agenticml", path, json.loads(path.read_text()))
    assert row.primary == 0.6
    assert row.secondary == 3.5
    assert row.avg_total_tokens == 1000.0


def test_aggregate_writes_markdown_and_json(tmp_path: Path):
    root = tmp_path / "benchmarks"
    _write_summary(
        root / "format_validity" / "agenticml" / "summary.json",
        suite="format_validity",
        fmt="agenticml",
        metrics={"valid_rate": 1.0, "parse_rate": 1.0, "avg_wall_sec": 10.0},
    )
    _write_summary(
        root / "swe" / "agenticml" / "summary.json",
        suite="swe",
        fmt="agenticml",
        metrics={"avg_iterations": 3.5, "avg_wall_sec": 200.0},
    )

    md = tmp_path / "benchmark_results.md"
    js = tmp_path / "aggregate.json"
    rows = aggregate_results(root, markdown_path=md, json_path=js)

    assert len(rows) == 2
    assert md.is_file()
    text = md.read_text()
    assert "format_validity" in text
    assert "swe/agenticml" in text or "missing cells" in text
    assert js.is_file()
    assert len(json.loads(js.read_text())["rows"]) == 2


def test_render_markdown_lists_missing_cells(tmp_path: Path):
    rows = load_result_rows(tmp_path)
    md = render_markdown(rows, results_root=tmp_path)
    assert "bfcl/agenticml" in md


def test_format_validity_task_result_and_suite_metrics():
    res = ValidityResult("1", "code", True, True, num_generated_tokens=5)
    tr = task_result(res, prompt=10, infer_sec=0.5)
    assert tr.success is True
    assert tr.tokens.total_tokens == 15

    m = suite_metrics([
        tr,
        TaskResult("2", "code", False, metrics={"parsed_ok": False, "structurally_valid": False}),
    ])
    assert m["parse_rate"] == 0.5
    assert m["valid_rate"] == 0.5
