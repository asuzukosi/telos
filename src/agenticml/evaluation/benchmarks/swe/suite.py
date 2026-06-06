"""swe-bench-lite benchmark suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from agenticml.evaluation.benchmarks.common import model_dir_name, sample_entries
from agenticml.evaluation.benchmarks.suite import (
    BenchmarkSuite,
    RunContext,
    SuiteScore,
    create_eval_backend,
    run_format_task,
)
from agenticml.evaluation.benchmarks.swe import agenticml, chatml
from agenticml.evaluation.benchmarks.swe.io import (
    default_result_dir,
    load_result_rows,
    write_results,
)
from agenticml.evaluation.benchmarks.swe.score import rows_to_task_results, score
from agenticml.evaluation.benchmarks.swe.subset import SWEBenchLiteSubset, load_subset
from agenticml.evaluation.harness.backends.chatml_backend import ChatMLBackend
from agenticml.evaluation.harness.backends.agenticml_backend import AgenticMLBackend
from agenticml.evaluation.harness.task import TaskResult


class SWEBenchLiteSuite(BenchmarkSuite):
    name = "swe"

    def __init__(self) -> None:
        self._subset: Optional[SWEBenchLiteSubset] = None

    def default_result_dir(self) -> Path:
        return default_result_dir()

    def load_dataset(self) -> SWEBenchLiteSubset:
        if self._subset is None:
            self._subset = load_subset()
        return self._subset

    def load_entries(
        self,
        num_examples: Optional[int],
        *,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        subset = self.load_dataset()
        picked = sample_entries(subset.entries, num_examples, seed=seed)
        return [{**e, "id": e["instance_id"]} for e in picked]

    def create_backend(self, ctx: RunContext) -> AgenticMLBackend | ChatMLBackend:
        return create_eval_backend(ctx)

    def run_one_task(
        self,
        backend: AgenticMLBackend | ChatMLBackend,
        entry: dict[str, Any],
        ctx: RunContext,
    ) -> dict[str, Any]:
        return run_format_task(
            backend,
            entry,
            ctx,
            agenticml_run=agenticml.run_one_task,
            chatml_run=chatml.run_one_task,
        )

    def persist_task_result(
        self,
        result_dir: Path,
        ctx: RunContext,
        row: dict[str, Any],
    ) -> None:
        write_results(result_dir, ctx.model_id, [row])

    def load_result_rows(
        self,
        result_dir: Path,
        ctx: RunContext,
        entries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        wanted = {str(e["instance_id"]) for e in entries}
        return load_result_rows(result_dir, ctx.model_id, wanted_ids=wanted)

    def score(
        self,
        result_dir: Path,
        ctx: RunContext,
        entries: list[dict[str, Any]],
        rows: list[dict[str, Any]],
        *,
        score_dir: Optional[Path] = None,
    ) -> SuiteScore:
        del result_dir, entries, score_dir
        slug = model_dir_name(ctx.model_id)
        return score(
            ctx.model_id,
            rows,
            score_dir=self.default_result_dir() / "score",
            run_id=f"{slug}-{ctx.format}-swe",
        )

    def rows_to_task_results(
        self,
        rows: list[dict[str, Any]],
        score: SuiteScore,
    ) -> list[TaskResult]:
        return rows_to_task_results(rows, score)
