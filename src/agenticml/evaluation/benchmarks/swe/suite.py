"""swe-bench-lite benchmark suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from telos.evaluation.benchmarks.common import model_dir_name, sample_entries
from telos.evaluation.benchmarks.suite import BenchmarkSuite, RunContext, SuiteScore
from telos.evaluation.benchmarks.swe import chatml, telos
from telos.evaluation.benchmarks.swe.io import (
    default_result_dir,
    load_result_rows,
    write_results,
)
from telos.evaluation.benchmarks.swe.score import rows_to_task_results, score
from telos.evaluation.benchmarks.swe.subset import SWEBenchLiteSubset, load_subset
from telos.evaluation.harness.backends.chatml_backend import ChatMLBackend
from telos.evaluation.harness.backends.telos_backend import TelosBackend
from telos.evaluation.harness.load import AdapterMode
from telos.evaluation.harness.task import TaskResult


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

    def create_backend(self, ctx: RunContext) -> TelosBackend | ChatMLBackend:
        factory = (
            TelosBackend.from_pretrained
            if ctx.format == "telos"
            else ChatMLBackend.from_pretrained
        )
        return factory(
            ctx.model_id,
            adapter_mode=AdapterMode(ctx.adapter_mode),
            adapter_id=ctx.adapter_id,
        )

    def run_one_task(
        self,
        backend: TelosBackend | ChatMLBackend,
        entry: dict[str, Any],
        ctx: RunContext,
    ) -> dict[str, Any]:
        if ctx.format == "telos":
            if not isinstance(backend, TelosBackend):
                raise TypeError(f"expected TelosBackend, got {type(backend).__name__}")
            return telos.run_one_task(backend, entry, ctx)
        if not isinstance(backend, ChatMLBackend):
            raise TypeError(f"expected ChatMLBackend, got {type(backend).__name__}")
        return chatml.run_one_task(backend, entry, ctx)

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
