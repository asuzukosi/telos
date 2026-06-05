import json

from telos.evaluation.benchmarks.bfcl.common import (
    count_retry_steps,
    retry_metrics_from_rows,
    write_results,
)
from telos.evaluation.benchmarks.bfcl.score import dedupe_result_rows, load_entry_validity, score


def test_load_entry_validity_treats_missing_ids_as_pass(tmp_path):
    import pytest

    pytest.importorskip("tree_sitter")
    from bfcl_eval.constants.eval_config import VERSION_PREFIX

    score_path = (
        tmp_path
        / "org_model"
        / "non_live"
        / f"{VERSION_PREFIX}_simple_python_score.json"
    )
    score_path.parent.mkdir(parents=True)
    score_path.write_text(
        '{"accuracy": 0.5, "correct_count": 1, "total_count": 2}\n'
        '{"id": "simple_python_1", "valid": false}\n'
    )
    out = load_entry_validity(
        tmp_path,
        "org/model",
        ["simple_python"],
        entry_ids={"simple_python_0", "simple_python_1"},
    )
    assert out == {"simple_python_0": True, "simple_python_1": False}


def test_dedupe_result_rows_uses_numeric_id_order():
    rows = [
        {"id": "simple_python_279", "result": "b"},
        {"id": "simple_python_52", "result": "a"},
    ]
    out = dedupe_result_rows(rows)
    assert [r["id"] for r in out] == ["simple_python_52", "simple_python_279"]


def test_count_retry_steps_multi_turn():
    result = [["step0"], ["a", "b", "c"]]
    assert count_retry_steps(result, "multi_turn_base_0") == 2


def test_count_retry_steps_single_turn():
    assert count_retry_steps('{"name": "fn", "parameters": {}}', "simple_python_1") == 0


def test_retry_metrics_from_rows():
    rows = [
        {"id": "multi_turn_base_0", "result": [["a"], ["b", "c"]]},
        {"id": "simple_python_0", "result": "[]"},
    ]
    m = retry_metrics_from_rows(rows)
    assert m["total_retry_count"] == 1.0
    assert m["avg_retry_count"] == 0.5


def test_score_writes_summary_with_mock_results(tmp_path):
    import pytest

    pytest.importorskip("tree_sitter")
    row = {
        "id": "simple_python_0",
        "result": "[]",
        "input_token_count": 1,
        "output_token_count": 2,
        "latency": 0.1,
    }
    write_results(tmp_path, "org/model", [row])
    suite_score = score(
        "org/model",
        tmp_path,
        [{"id": "simple_python_0"}],
        score_dir=tmp_path / "score",
    )
    assert "simple_python" in suite_score.per_domain
    out = tmp_path / "score" / "org_model" / "telos_subset_summary.json"
    assert out.is_file()
    loaded = json.loads(out.read_text())
    assert loaded["retry"]["total_retry_count"] == 0.0
