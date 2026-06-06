"""run the default benchmark matrix (suites × formats)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from telos.evaluation.benchmarks.common import repo_root
from telos.evaluation.benchmarks.run import SUITES, run_suite
from telos.evaluation.harness.task import BenchmarkResult

DEFAULT_TELLOS_MODEL = "kosiasuzu/telos-llama3.1-8b-lora-merged"
DEFAULT_CHATML_MODEL = "kosiasuzu/chatml-llama3.1-8b-lora-merged"

MATRIX_SUITES = ("bfcl", "toolbench", "swe", "format_validity")
MATRIX_FORMATS = ("telos", "chatml")

DEFAULT_MODELS: dict[str, str] = {
    "telos": DEFAULT_TELLOS_MODEL,
    "chatml": DEFAULT_CHATML_MODEL,
}


@dataclass(frozen=True)
class MatrixCell:
    suite: str
    fmt: str
    model_id: str


def matrix_cells(
    *,
    suites: list[str] | None = None,
    formats: list[str] | None = None,
    models: dict[str, str] | None = None,
) -> list[MatrixCell]:
    resolved_models = dict(DEFAULT_MODELS)
    if models:
        resolved_models.update(models)
    out: list[MatrixCell] = []
    for suite in suites or list(MATRIX_SUITES):
        if suite not in SUITES:
            raise ValueError(f"unknown suite: {suite}")
        for fmt in formats or list(MATRIX_FORMATS):
            if fmt not in resolved_models:
                raise ValueError(f"no model configured for format: {fmt}")
            out.append(MatrixCell(suite=suite, fmt=fmt, model_id=resolved_models[fmt]))
    return out


def run_matrix(
    *,
    suites: list[str] | None = None,
    formats: list[str] | None = None,
    models: dict[str, str] | None = None,
    output_root: Path | None = None,
    num_examples: int | None = None,
    sample_seed: int = 42,
    max_new_tokens: int | None = None,
    run_inference: bool = True,
    run_score: bool = True,
    dry_run: bool = False,
    continue_on_error: bool = False,
) -> list[tuple[MatrixCell, BenchmarkResult | None]]:
    root = output_root or (repo_root() / "results" / "benchmarks")
    cells = matrix_cells(suites=suites, formats=formats, models=models)
    results: list[tuple[MatrixCell, BenchmarkResult | None]] = []

    for i, cell in enumerate(cells, start=1):
        label = f"[{i}/{len(cells)}] {cell.suite} / {cell.fmt} / {cell.model_id}"
        print(label)
        if dry_run:
            results.append((cell, None))
            continue

        run_kw: dict[str, Any] = {
            "output_dir": root / cell.suite,
            "num_examples": num_examples,
            "sample_seed": sample_seed,
            "max_new_tokens": max_new_tokens,
        }
        if cell.suite in ("bfcl", "swe"):
            run_kw["run_inference"] = run_inference
            run_kw["run_score"] = run_score

        try:
            result = run_suite(cell.suite, cell.model_id, cell.fmt, **run_kw)
            results.append((cell, result))
        except Exception as exc:
            print(f"failed: {exc}")
            results.append((cell, None))
            if not continue_on_error:
                raise

    _print_summary(results)
    return results


def _print_summary(results: list[tuple[MatrixCell, BenchmarkResult | None]]) -> None:
    if not results or all(r is None for _, r in results):
        return
    print("matrix summary:")
    for cell, result in results:
        if result is None:
            print(f"  {cell.suite:16} {cell.fmt:6}  skipped")
            continue
        primary = result.metrics.get(f"{cell.suite}_primary")
        if primary is None:
            for key in ("valid_rate", "pass_rate", "resolved_rate", "accuracy"):
                if key in result.metrics:
                    primary = result.metrics[key]
                    break
        metric = f"{primary:.2%}" if isinstance(primary, (int, float)) else "—"
        print(f"  {cell.suite:16} {cell.fmt:6}  {metric}")
