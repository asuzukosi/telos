from pathlib import Path

from telos.evaluation.benchmarks.suite import RunContext
from telos.evaluation.benchmarks.swe.io import load_result_rows, write_results
from telos.evaluation.benchmarks.swe.suite import SWEBenchLiteSuite


def _row(instance_id: str = "django__django-11099", *, patch: str = "--- a/x.py\n") -> dict:
    return {
        "instance_id": instance_id,
        "model_patch": patch,
        "stopped_on": "submitted",
        "iterations": 2,
        "prompt_tokens": 10,
        "generated_tokens": 5,
        "inference_sec": 0.1,
    }


def test_load_entries_adds_id():
    suite = SWEBenchLiteSuite()
    entries = suite.load_entries(1, seed=42)
    assert len(entries) == 1
    assert entries[0]["id"] == entries[0]["instance_id"]


def test_persist_and_load_result_rows(tmp_path: Path):
    suite = SWEBenchLiteSuite()
    ctx = RunContext(model_id="org/model", format="telos")
    row = _row()
    suite.persist_task_result(tmp_path, ctx, row)
    loaded = suite.load_result_rows(tmp_path, ctx, [{"instance_id": "django__django-11099"}])
    assert len(loaded) == 1
    assert loaded[0]["model_patch"] == "--- a/x.py\n"


def test_score_and_task_results(tmp_path: Path, monkeypatch):
    suite = SWEBenchLiteSuite()
    rows = [_row(), _row(instance_id="sympy__sympy-12454")]
    ctx = RunContext(model_id="org/model", format="telos")

    def _fake_score(model_id, rows, *, score_dir, run_id, **kwargs):
        from telos.evaluation.benchmarks.suite import SuiteScore

        del model_id, score_dir, run_id, kwargs
        return SuiteScore(
            primary=0.5,
            validity={
                "django__django-11099": True,
                "sympy__sympy-12454": False,
            },
        )

    monkeypatch.setattr("telos.evaluation.benchmarks.swe.suite.score", _fake_score)
    sc = suite.score(Path("."), ctx, [], rows)
    assert sc.primary == 0.5
    tasks = suite.rows_to_task_results(rows, sc)
    assert tasks[0].success is True
    assert tasks[1].success is False


def test_write_and_load_io_helpers(tmp_path: Path):
    rows = [_row(instance_id="sympy__sympy-12454")]
    write_results(tmp_path, "org/model", rows)
    out = load_result_rows(tmp_path, "org/model", wanted_ids={"sympy__sympy-12454"})
    assert out[0]["instance_id"] == "sympy__sympy-12454"
