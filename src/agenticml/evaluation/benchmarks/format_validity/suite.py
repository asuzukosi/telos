"""format validity suite interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, cast

from telos.evaluation.benchmarks.common import repo_root
from telos.evaluation.benchmarks.format_validity.evaluate import (
    FORMAT_SPECS,
    ValidityResult,
    eval_row,
    prepare_eval_dataset,
    suite_metrics,
    task_result,
)
from telos.evaluation.benchmarks.suite import BenchmarkSuite, RunContext, SuiteScore
from telos.evaluation.harness.load import AdapterMode, load_model
from telos.evaluation.harness.task import BenchmarkResult, TaskResult
from datasets import Dataset, load_dataset

DATASET_ID = "kosiasuzu/telos-agent-trajectory-dataset"
SPLIT = "eval"


class FormatValiditySuite(BenchmarkSuite):
    name = "format_validity"

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._stop_ids: list[int] = []
        self._fmt: str = "telos"

    def default_result_dir(self) -> Path:
        return repo_root() / "results" / "benchmarks" / "format_validity"

    def load_dataset(self) -> str:
        return DATASET_ID

    def result_split(self) -> str:
        return SPLIT

    def load_entries(
        self,
        num_examples: Optional[int],
        *,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        ds_full = load_dataset(DATASET_ID, split=SPLIT)
        k = -1 if num_examples is None else num_examples
        ds, _, _, _ = prepare_eval_dataset(ds_full, self._fmt, k, seed)
        dataset = cast(Dataset, ds)
        return [dict(dataset[i]) for i in range(len(dataset))]

    def run(self, ctx: RunContext, **kwargs: Any) -> BenchmarkResult:
        self._fmt = ctx.format
        return super().run(ctx, **kwargs)

    def create_backend(self, ctx: RunContext) -> Any:
        spec = FORMAT_SPECS[ctx.format]
        mode = AdapterMode(ctx.adapter_mode)
        self._model = load_model(ctx.model_id, mode, ctx.adapter_id)
        self._tokenizer = spec.load_tokenizer(ctx.model_id)
        self._model.eval()
        self._stop_ids = spec.stop_token_ids(self._tokenizer)
        return self._model

    def run_one_task(
        self,
        backend: Any,
        entry: dict[str, Any],
        ctx: RunContext,
    ) -> dict[str, Any]:
        spec = FORMAT_SPECS[ctx.format]
        out = eval_row(
            entry,
            backend,
            self._tokenizer,
            spec,
            self._stop_ids,
            ctx.max_new_tokens,
        )
        if out is None:
            return {
                "id": entry["id"],
                "domain": entry.get("domain", "unknown"),
                "skipped": True,
            }
        res, prompt, infer = out
        return {
            "id": res.id,
            "domain": res.domain,
            "validity": res,
            "prompt_tokens": prompt,
            "inference_sec": infer,
        }

    def persist_task_result(
        self,
        result_dir: Path,
        ctx: RunContext,
        row: dict[str, Any],
    ) -> None:
        del result_dir, ctx, row

    def load_result_rows(
        self,
        result_dir: Path,
        ctx: RunContext,
        entries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        del result_dir, ctx
        return [{"id": e["id"], "domain": e.get("domain", "unknown")} for e in entries]

    def score(
        self,
        result_dir: Path,
        ctx: RunContext,
        entries: list[dict[str, Any]],
        rows: list[dict[str, Any]],
        *,
        score_dir: Optional[Path] = None,
    ) -> SuiteScore:
        del result_dir, ctx, entries, score_dir
        tasks = self.rows_to_task_results(rows, SuiteScore())
        metrics = suite_metrics(tasks)
        return SuiteScore(
            primary=metrics.get("valid_rate"),
            per_domain=metrics.get("by_domain", {}),
            extra=metrics,
        )

    def rows_to_task_results(
        self,
        rows: list[dict[str, Any]],
        score: SuiteScore,
    ) -> list[TaskResult]:
        del score
        out: list[TaskResult] = []
        for row in rows:
            if row.get("skipped"):
                out.append(
                    TaskResult(
                        str(row["id"]),
                        str(row.get("domain") or "unknown"),
                        success=False,
                        metrics={"skipped": True},
                    )
                )
                continue
            res: ValidityResult = row["validity"]
            out.append(
                task_result(
                    res,
                    prompt=int(row.get("prompt_tokens") or 0),
                    infer_sec=float(row.get("inference_sec") or 0.0),
                )
            )
        return out
