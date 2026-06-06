from agenticml.evaluation.benchmarks.toolbench.score import rows_to_task_results, score
from tests.evaluation.benchmarks.toolbench.helpers import finish_row


def test_score_structural_pass_rate():
    rows = [
        finish_row(final_answer="a"),
        {**finish_row(final_answer="b"), "id": "2", "success": False, "final_answer": ""},
    ]
    rows[1]["messages"] = rows[1]["messages"][:-2]
    s = score(rows)
    assert s.primary == 0.5
    assert s.extra["structural_pass_rate"] == 0.5
    assert s.extra["cache_pass_rate"] == 0.5
    assert s.extra["avg_steps"] == 2.0
    tasks = rows_to_task_results(rows, s)
    assert len(tasks) == 2
    assert tasks[0].success is True
    assert tasks[0].metrics["cache_pass"] is True
