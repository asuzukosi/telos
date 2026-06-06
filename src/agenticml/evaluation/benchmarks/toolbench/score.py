"""toolbench scoring via upstream tooleval helpers."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, cast

from agenticml.evaluation.benchmarks.common import model_dir_name
from agenticml.evaluation.benchmarks.suite import SuiteScore
from agenticml.evaluation.benchmarks.toolbench.convert import row_to_converted, rows_to_converted_map
from agenticml.evaluation.harness.task import TaskResult, TaskTiming, TaskTokens


def _tooleval_imports():
    from agenticml.evaluation.benchmarks.toolbench.common import ensure_toolbench_on_path

    root = ensure_toolbench_on_path()
    tooleval_dir = root / "toolbench" / "tooleval"
    path = str(tooleval_dir)
    if path not in sys.path:
        sys.path.insert(0, path)
    from toolbench.tooleval.evaluators.registered_cls.rtl import ReinforceToolLearningEvaluator
    from toolbench.tooleval.utils import get_steps

    return get_steps, ReinforceToolLearningEvaluator


def structural_pass(example: dict[str, Any]) -> tuple[bool, str]:
    get_steps, rtl_cls = _tooleval_imports()
    evaluator = object.__new__(rtl_cls)
    try:
        not_hallucinate = evaluator.check_has_hallucination(
            example["available_tools"], example["answer"]
        )
    except Exception:
        not_hallucinate = True

    _steps, final_step = get_steps(example)
    if "'name': 'Finish'" not in final_step:
        return False, "no finish"

    final_answer = example["answer"].get("final_answer") or ""
    if not final_answer or "give_up_and_restart" in final_answer:
        return False, "empty final answer"

    if not not_hallucinate:
        return False, "hallucination"

    return True, "structural pass"


def _gpt_pass(example: dict[str, Any]) -> tuple[Optional[bool], str]:
    if not os.environ.get("OPENAI_API_KEY"):
        return None, "gpt skipped (no OPENAI_API_KEY)"

    try:
        _get_steps, rtl_cls = _tooleval_imports()
        from evaluators import load_registered_automatic_evaluator
        from evaluators.registered_cls.rtl import AnswerPass, AnswerStatus, TaskStatus

        from agenticml.evaluation.benchmarks.toolbench.common import ensure_toolbench_on_path

        tooleval_dir = ensure_toolbench_on_path() / "toolbench" / "tooleval"
        evaluator = cast(
            rtl_cls,
            load_registered_automatic_evaluator(
                evaluator_name="tooleval_gpt-3.5-turbo_default",
                evaluators_cfg_path=str(tooleval_dir / "evaluators"),
            ),
        )
        task = {
            "query": example["query"],
            "available_tools": example["available_tools"],
        }
        is_solved, _ = cast(
            tuple[AnswerStatus, str],
            evaluator.check_is_solved(task, example["answer"], return_reason=False),
        )
        task_solvable, _ = cast(
            tuple[TaskStatus, str],
            evaluator.check_task_solvable(
                task,
                has_been_solved=is_solved == AnswerStatus.Solved,
                return_reason=False,
            ),
        )
        passed = evaluator.is_passed(
            task,
            example["answer"],
            answer_status=is_solved,
            task_status=task_solvable,
        )
        return passed == AnswerPass.Passed, "gpt judge"
    except Exception as exc:
        return None, f"gpt skipped ({exc})"


def score_row(row: dict[str, Any], *, use_gpt: bool = False) -> dict[str, Any]:
    converted = row_to_converted(row)
    cache_pass = bool(row.get("success"))
    out: dict[str, Any] = {
        "cache_pass": cache_pass,
        "structural_pass": False,
        "gpt_pass": None,
        "reason": "not converted",
    }
    if converted is None:
        return out

    ok, reason = structural_pass(converted)
    out["structural_pass"] = ok
    out["reason"] = reason
    if use_gpt:
        gpt_ok, gpt_reason = _gpt_pass(converted)
        out["gpt_pass"] = gpt_ok
        if gpt_ok is not None:
            out["reason"] = gpt_reason
    return out


def write_converted(
    score_dir: Path,
    model_id: str,
    group: str,
    converted: dict[str, dict[str, Any]],
) -> Path:
    root = score_dir / model_dir_name(model_id)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{group}.json"
    path.write_text(json.dumps(converted, indent=2))
    return path


def rows_to_task_results(
    rows: list[dict[str, Any]],
    score: SuiteScore,
) -> list[TaskResult]:
    per_row = score.extra.get("per_row") or {}
    out: list[TaskResult] = []
    for row in rows:
        rid = str(row["id"])
        row_score = per_row.get(rid, {})
        ok = bool(row_score.get("structural_pass"))
        out.append(
            TaskResult(
                task_id=rid,
                domain=str(row.get("group") or "G1_instruction"),
                success=ok,
                metrics={
                    "structural_pass": ok,
                    "cache_pass": bool(row_score.get("cache_pass")),
                    "gpt_pass": row_score.get("gpt_pass"),
                    "steps": int(row.get("steps") or 0),
                    "stopped_on": row.get("stopped_on"),
                },
                timing=TaskTiming(
                    inference_sec=float(row.get("latency") or 0.0),
                    tool_sec=float(row.get("tool_sec") or 0.0),
                    total_sec=float(row.get("total_sec") or row.get("latency") or 0.0),
                ),
                tokens=TaskTokens(
                    prompt_tokens=int(row.get("input_token_count") or 0),
                    generated_tokens=int(row.get("output_token_count") or 0),
                ),
                detail={
                    "final_answer": row.get("final_answer"),
                    "score_reason": row_score.get("reason"),
                },
            )
        )
    return out


def score(
    rows: list[dict[str, Any]],
    *,
    score_dir: Optional[Path] = None,
    model_id: Optional[str] = None,
    group: str = "G1_instruction",
    use_gpt: bool = False,
) -> SuiteScore:
    n = len(rows)
    if n == 0:
        return SuiteScore(primary=0.0, extra={"n": 0, "pass_rate": 0.0})

    use_gpt = use_gpt or os.environ.get("TOOLEVAL_GPT", "").lower() in ("1", "true", "yes")
    per_row: dict[str, dict[str, Any]] = {}
    structural_passed = 0
    cache_passed = 0
    gpt_passed = 0
    gpt_scored = 0

    for row in rows:
        row_score = score_row(row, use_gpt=use_gpt)
        per_row[str(row["id"])] = row_score
        if row_score["structural_pass"]:
            structural_passed += 1
        if row_score["cache_pass"]:
            cache_passed += 1
        if row_score["gpt_pass"] is True:
            gpt_passed += 1
        if row_score["gpt_pass"] is not None:
            gpt_scored += 1

    converted_path: Optional[str] = None
    if score_dir is not None and model_id is not None:
        converted = rows_to_converted_map(rows)
        if converted:
            path = write_converted(score_dir, model_id, group, converted)
            converted_path = str(path.resolve())

    structural_rate = structural_passed / n
    cache_rate = cache_passed / n
    extra: dict[str, Any] = {
        "n": n,
        "pass_rate": structural_rate,
        "structural_pass_rate": structural_rate,
        "cache_pass_rate": cache_rate,
        "avg_steps": sum(int(r.get("steps") or 0) for r in rows) / n,
        "per_row": per_row,
    }
    if converted_path:
        extra["converted_path"] = converted_path
    if gpt_scored:
        extra["gpt_pass_rate"] = gpt_passed / gpt_scored

    return SuiteScore(primary=structural_rate, extra=extra)
