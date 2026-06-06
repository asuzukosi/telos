from __future__ import annotations

from telos.evaluation.benchmarks.format_validity.evaluate import (
    ValidityResult,
    suite_metrics,
    task_result,
)
from telos.evaluation.harness.task import TaskResult


def test_task_result_and_suite_metrics():
    res = ValidityResult("1", "code", True, True, num_generated_tokens=5)
    tr = task_result(res, prompt=10, infer_sec=0.5)
    assert tr.success is True
    assert tr.tokens.total_tokens == 15

    m = suite_metrics([tr, TaskResult("2", "code", False, metrics={"parsed_ok": False, "structurally_valid": False})])
    assert m["parse_rate"] == 0.5
    assert m["valid_rate"] == 0.5
