"""unit tests for swe swebench grader integration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from agenticml.evaluation.benchmarks.swe.io import write_preds
from agenticml.evaluation.benchmarks.swe.score import rows_to_task_results, score


def _row(
    *,
    instance_id: str = "django__django-11099",
    patch: str = "--- a/x.py\n",
    iterations: int = 3,
) -> dict:
    return {
        "instance_id": instance_id,
        "model_patch": patch,
        "stopped_on": "submitted",
        "iterations": iterations,
        "prompt_tokens": 10,
        "generated_tokens": 5,
        "inference_sec": 0.2,
    }


def test_write_preds(tmp_path: Path):
    rows = [_row(), _row(instance_id="sympy__sympy-12454", patch="--- a/y.py\n")]
    path = write_preds(tmp_path / "preds.json", rows, model_id="org/model")
    loaded = json.loads(path.read_text())
    assert loaded["django__django-11099"]["model_name_or_path"] == "org/model"
    assert loaded["django__django-11099"]["model_patch"] == "--- a/x.py\n"


def test_score_writes_summary_without_grader(tmp_path: Path):
    rows = [_row(), _row(instance_id="sympy__sympy-12454", patch="")]
    suite_score = score("org/model", rows, score_dir=tmp_path, run_grader=False)
    assert suite_score.primary is None
    summary = json.loads((tmp_path / "org_model" / "agenticml_subset_summary.json").read_text())
    assert summary["model_id"] == "org/model"
    assert (tmp_path / "org_model" / "preds.json").is_file()


def test_score_with_mock_grader_report(tmp_path: Path):
    rows = [_row(), _row(instance_id="sympy__sympy-12454", patch="--- a/y.py\n")]
    report = {
        "submitted_instances": 2,
        "resolved_instances": 1,
        "resolved_ids": ["django__django-11099"],
    }
    report_path = tmp_path / "org_model.agenticml-swe.json"

    def _fake_grader(*_args, **_kwargs):
        report_path.write_text(json.dumps(report) + "\n")
        return report_path

    with patch("agenticml.evaluation.benchmarks.swe.score.run_swebench_grader", side_effect=_fake_grader):
        suite_score = score("org/model", rows, score_dir=tmp_path)

    assert suite_score.primary == 0.5
    assert suite_score.validity == {
        "django__django-11099": True,
        "sympy__sympy-12454": False,
    }

    tasks = rows_to_task_results(rows, suite_score)
    assert tasks[0].success is True
    assert tasks[1].success is False
