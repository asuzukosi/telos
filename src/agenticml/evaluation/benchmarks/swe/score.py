"""score swe-bench predictions via upstream swebench run_evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from telos.evaluation.benchmarks.common import model_dir_name, repo_root
from telos.evaluation.benchmarks.swe.common import ensure_swebench_importable
from telos.evaluation.benchmarks.swe.io import write_preds
from telos.evaluation.benchmarks.swe.subset import SWE_LITE_DATASET, SWE_LITE_SPLIT
from telos.evaluation.benchmarks.suite import SuiteScore
from telos.evaluation.harness.task import TaskResult, TaskTiming, TaskTokens


def _parse_report(
    report: dict[str, Any],
    instance_ids: list[str],
) -> tuple[Optional[float], dict[str, bool]]:
    submitted = int(report.get("submitted_instances") or 0)
    resolved_ids = set(report.get("resolved_ids") or [])
    primary = (int(report.get("resolved_instances") or 0) / submitted) if submitted else None
    return primary, {iid: iid in resolved_ids for iid in instance_ids}


def run_swebench_grader(
    preds_path: Path,
    instance_ids: list[str],
    *,
    run_id: str,
    work_dir: Path,
    max_workers: int = 4,
    timeout: int = 1_800,
) -> Optional[Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    prev = Path.cwd()
    os.chdir(work_dir)
    try:
        ensure_swebench_importable()
        from swebench.harness.run_evaluation import main as run_evaluation

        return run_evaluation(
            dataset_name=SWE_LITE_DATASET,
            split=SWE_LITE_SPLIT,
            instance_ids=instance_ids,
            predictions_path=str(preds_path.resolve()),
            max_workers=max_workers,
            force_rebuild=False,
            cache_level="env",
            clean=False,
            open_file_limit=4096,
            run_id=run_id,
            timeout=timeout,
            namespace="swebench",
            rewrite_reports=False,
            modal=False,
            report_dir=str(work_dir),
        )
    finally:
        os.chdir(prev)


def score(
    model_id: str,
    rows: list[dict[str, Any]],
    *,
    score_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
    max_workers: int = 4,
    timeout: int = 1_800,
    run_grader: bool = True,
) -> SuiteScore:
    slug = model_dir_name(model_id)
    model_dir = (score_dir or repo_root() / "results" / "benchmarks" / "swe" / "score") / slug
    model_dir.mkdir(parents=True, exist_ok=True)

    preds_path = write_preds(model_dir / "preds.json", rows, model_id=model_id)
    instance_ids = [str(r["instance_id"]) for r in rows]
    rid = run_id or f"{slug}-telos-swe"

    report_path: Optional[Path] = None
    if run_grader:
        report_path = run_swebench_grader(
            preds_path,
            instance_ids,
            run_id=rid,
            work_dir=model_dir,
            max_workers=max_workers,
            timeout=timeout,
        )

    primary: Optional[float] = None
    validity: dict[str, bool] = {}
    if report_path and report_path.is_file():
        primary, validity = _parse_report(json.loads(report_path.read_text()), instance_ids)

    summary = {
        "model_id": model_id,
        "run_id": rid,
        "resolved_rate": primary,
        "preds_path": str(preds_path),
        "report_path": str(report_path) if report_path else None,
    }
    (model_dir / "telos_subset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    n = len(rows)
    return SuiteScore(
        primary=primary,
        validity=validity,
        extra={
            "run_id": rid,
            "preds_path": str(preds_path),
            "report_path": str(report_path) if report_path else None,
            "avg_iterations": sum(int(r.get("iterations") or 0) for r in rows) / n if n else 0.0,
        },
    )


def rows_to_task_results(
    rows: list[dict[str, Any]],
    score: SuiteScore,
) -> list[TaskResult]:
    return [
        TaskResult(
            task_id=str(row["instance_id"]),
            domain="swe_bench_lite",
            success=score.validity.get(str(row["instance_id"])),
            metrics={
                "stopped_on": row.get("stopped_on"),
                "iterations": row.get("iterations"),
                "submitted": bool(row.get("model_patch")),
            },
            timing=TaskTiming(
                inference_sec=float(row.get("inference_sec") or 0.0),
                total_sec=float(row.get("inference_sec") or 0.0),
            ),
            tokens=TaskTokens(
                prompt_tokens=int(row.get("prompt_tokens") or 0),
                generated_tokens=int(row.get("generated_tokens") or 0),
            ),
        )
        for row in rows
    ]
