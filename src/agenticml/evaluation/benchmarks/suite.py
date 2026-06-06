"""benchmark suite interface and generic orchestration."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

from telos.evaluation.benchmarks.common import sample_entries
from telos.evaluation.harness.aggregate import aggregate_efficiency
from telos.evaluation.harness.runner import write_benchmark
from telos.evaluation.harness.task import BenchmarkResult, BenchmarkRunMeta, TaskResult


@dataclass
class RunContext:
    model_id: str
    format: str
    adapter_mode: str = "merged"
    adapter_id: Optional[str] = None
    max_new_tokens: int = 512
    inject_retry_failure: bool = False
    max_iterations: Optional[int] = None


@dataclass
class SuiteScore:
    primary: Optional[float] = None
    per_domain: dict[str, Any] = field(default_factory=dict)
    validity: dict[str, bool] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


class BenchmarkSuite(ABC):
    """public surface: load entries, run one task, score, aggregate."""

    name: str

    @abstractmethod
    def default_result_dir(self) -> Path:
        ...

    @abstractmethod
    def load_dataset(self) -> Any:
        """pinned subset metadata (paths, categories, etc.)."""

    @abstractmethod
    def load_entries(
        self,
        num_examples: Optional[int],
        *,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def create_backend(self, ctx: RunContext) -> Any:
        ...

    @abstractmethod
    def run_one_task(self, backend: Any, entry: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        """return a raw result row (id, tokens, latency, suite-specific payload)."""

    @abstractmethod
    def persist_task_result(
        self,
        result_dir: Path,
        ctx: RunContext,
        row: dict[str, Any],
    ) -> None:
        ...

    @abstractmethod
    def load_result_rows(
        self,
        result_dir: Path,
        ctx: RunContext,
        entries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def score(
        self,
        result_dir: Path,
        ctx: RunContext,
        entries: list[dict[str, Any]],
        rows: list[dict[str, Any]],
        *,
        score_dir: Optional[Path] = None,
    ) -> SuiteScore:
        ...

    @abstractmethod
    def rows_to_task_results(
        self,
        rows: list[dict[str, Any]],
        score: SuiteScore,
    ) -> list[TaskResult]:
        ...

    def dataset_label(self, dataset: Any) -> str:
        if hasattr(dataset, "source"):
            return str(dataset.source)
        return str(dataset)

    def result_split(self) -> str:
        return "subset"

    def run(
        self,
        ctx: RunContext,
        *,
        output_dir: Optional[Path] = None,
        num_examples: Optional[int] = None,
        sample_seed: int = 42,
        run_inference: bool = True,
        run_score: bool = True,
        score_dir: Optional[Path] = None,
    ) -> BenchmarkResult:
        if not run_inference and not run_score:
            raise ValueError("need at least one of inference or scoring")

        dataset = self.load_dataset()
        entries = self.load_entries(num_examples, seed=sample_seed)
        base = output_dir or self.default_result_dir()
        out_score = score_dir or (base.parent / "score")
        bench_out = base / ctx.format

        print(
            f"{self.name} run: {len(entries)} examples"
            + (f", dataset={self.dataset_label(dataset)}" if dataset else "")
        )

        load_sec = 0.0
        rows: list[dict[str, Any]] = []
        suite_score = SuiteScore()

        if run_inference:
            print(f"loading model {ctx.model_id} ({ctx.format}) — then {len(entries)} examples...")
            t0 = time.perf_counter()
            backend = self.create_backend(ctx)
            slug = ctx.model_id.replace("/", "_")
            bar = tqdm(entries, desc=f"{self.name} inference ({slug})", unit="example")
            for entry in bar:
                eid = str(entry.get("id", entry.get("query_id", "")))
                bar.set_postfix_str(eid, refresh=False)
                row = self.run_one_task(backend, entry, ctx)
                rows.append(row)
                self.persist_task_result(base, ctx, row)
            load_sec = time.perf_counter() - t0
            print(f"inference done: {len(rows)} entries -> {base}")
        else:
            rows = self.load_result_rows(base, ctx, entries)

        if run_score:
            print(f"scoring {self.name}...")
            suite_score = self.score(
                base,
                ctx,
                entries,
                rows,
                score_dir=out_score,
            )

        tasks = self.rows_to_task_results(rows, suite_score)
        extra = dict(suite_score.extra)
        if suite_score.primary is not None:
            extra[f"{self.name}_primary"] = suite_score.primary
        if suite_score.per_domain:
            extra[f"{self.name}_per_domain"] = suite_score.per_domain

        meta = BenchmarkRunMeta(
            suite=self.name,
            model=ctx.model_id,
            format=ctx.format,
            adapter_mode=ctx.adapter_mode,
            dataset=self.dataset_label(dataset),
            split=self.result_split(),
            num_run=len(entries),
            sample_seed=sample_seed,
            adapter_id=ctx.adapter_id,
        )
        result = BenchmarkResult(
            meta=meta,
            metrics=aggregate_efficiency(tasks, extra),
            tasks=tasks,
        )
        if run_inference and load_sec and tasks:
            per = load_sec / len(tasks)
            for t in tasks:
                if t.timing.load_sec <= 0:
                    t.timing.load_sec = per

        summary_path, _ = write_benchmark(bench_out, result)
        print(f"benchmark envelope written to {summary_path}")
        if suite_score.primary is not None:
            print(f"{self.name} primary: {suite_score.primary:.2%}")
        if suite_score.validity and tasks:
            passed = sum(1 for t in tasks if t.success)
            print(f"per-example: {passed}/{len(tasks)} passed")
            for t in tasks:
                mark = "pass" if t.success else "fail"
                print(f"  {mark}: {t.task_id}")
        return result
