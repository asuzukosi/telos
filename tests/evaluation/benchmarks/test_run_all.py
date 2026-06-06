from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agenticml.constants import DEFAULT_AGENTICML_MERGED_MODEL, DEFAULT_CHATML_MERGED_MODEL
from agenticml.evaluation.benchmarks.run_all import matrix_cells, run_matrix
from agenticml.evaluation.harness.task import BenchmarkResult, BenchmarkRunMeta


def test_matrix_cells_default():
    cells = matrix_cells()
    assert len(cells) == 8
    assert cells[0].suite == "bfcl"
    assert cells[0].fmt == "agenticml"
    assert cells[0].model_id == DEFAULT_AGENTICML_MERGED_MODEL
    assert any(c.fmt == "chatml" and c.model_id == DEFAULT_CHATML_MERGED_MODEL for c in cells)


def test_matrix_cells_subset():
    cells = matrix_cells(suites=["toolbench"], formats=["agenticml"])
    assert len(cells) == 1
    assert cells[0].suite == "toolbench"


def test_matrix_cells_unknown_suite():
    with pytest.raises(ValueError, match="unknown suite"):
        matrix_cells(suites=["nope"])


def test_run_all_dry_run(capsys):
    results = run_matrix(suites=["bfcl"], formats=["agenticml"], dry_run=True)
    assert len(results) == 1
    assert results[0][1] is None
    assert "bfcl / agenticml" in capsys.readouterr().out


def test_run_all_calls_run_suite(monkeypatch, tmp_path: Path):
    mock = MagicMock(
        return_value=BenchmarkResult(
            meta=BenchmarkRunMeta(
                suite="bfcl",
                model=DEFAULT_AGENTICML_MERGED_MODEL,
                format="agenticml",
                dataset="bfcl",
                split="subset",
                num_run=1,
            ),
            metrics={"bfcl_primary": 0.5},
        )
    )
    monkeypatch.setattr("agenticml.evaluation.benchmarks.run_all.run_suite", mock)
    results = run_matrix(
        suites=["bfcl"],
        formats=["agenticml"],
        output_root=tmp_path,
    )
    assert len(results) == 1
    mock.assert_called_once()
    assert mock.call_args.args == ("bfcl", DEFAULT_AGENTICML_MERGED_MODEL, "agenticml")
    assert mock.call_args.kwargs["output_dir"] == tmp_path / "bfcl"
    assert mock.call_args.kwargs["run_inference"] is True
    assert mock.call_args.kwargs["run_score"] is True


def test_run_all_continue_on_error(monkeypatch, tmp_path: Path):
    calls: list[str] = []

    def _run(suite, model, fmt, **kwargs):
        del model, fmt, kwargs
        calls.append(suite)
        if suite == "bfcl":
            raise RuntimeError("boom")

    monkeypatch.setattr("agenticml.evaluation.benchmarks.run_all.run_suite", _run)
    results = run_matrix(
        suites=["bfcl", "toolbench"],
        formats=["agenticml"],
        output_root=tmp_path,
        continue_on_error=True,
    )
    assert calls == ["bfcl", "toolbench"]
    assert results[0][1] is None
    assert results[1][1] is None
