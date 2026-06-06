from pathlib import Path

from telos.evaluation.benchmarks.suite import RunContext
from telos.evaluation.benchmarks.toolbench.io import load_result_rows, write_results
from telos.evaluation.benchmarks.toolbench.suite import ToolBenchSuite
from tests.test_toolbench_convert import _finish_row


def test_suite_score_and_task_results(tmp_path: Path):
    suite = ToolBenchSuite()
    rows = [
        {
            **_finish_row(final_answer="done"),
            "steps": 3,
            "latency": 1.0,
            "tool_sec": 0.2,
            "total_sec": 1.2,
            "input_token_count": 100,
            "output_token_count": 40,
        },
        {
            **{**_finish_row(final_answer=""), "id": "2", "success": False},
            "steps": 5,
            "latency": 2.0,
            "tool_sec": 0.5,
            "total_sec": 2.5,
            "input_token_count": 200,
            "output_token_count": 80,
        },
    ]
    ctx = RunContext(model_id="org/model", format="telos")
    sc = suite.score(Path("."), ctx, [], rows, score_dir=tmp_path)
    assert sc.primary == 0.5
    assert sc.extra["converted_path"]
    tasks = suite.rows_to_task_results(rows, sc)
    assert len(tasks) == 2
    assert tasks[0].success is True
    assert tasks[0].metrics["steps"] == 3


def test_persist_and_load_result_rows(tmp_path: Path):
    suite = ToolBenchSuite()
    ctx = RunContext(model_id="org/model", format="telos")
    row = {"id": "42", "group": "G1_instruction", "success": True, "steps": 1}
    suite.persist_task_result(tmp_path, ctx, row)
    loaded = suite.load_result_rows(tmp_path, ctx, [{"id": "42"}, {"id": "99"}])
    assert len(loaded) == 1
    assert loaded[0]["id"] == "42"


def test_write_and_load_io_helpers(tmp_path: Path):
    rows = [{"id": "7", "success": False}]
    write_results(tmp_path, "org/model", rows)
    out = load_result_rows(tmp_path, "org/model", wanted_ids={"7"})
    assert out[0]["id"] == "7"
