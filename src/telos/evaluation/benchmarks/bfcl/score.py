"""score bfcl result files via upstream bfcl evaluate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from telos.evaluation.benchmarks.bfcl.common import ResultHandler, model_dir_name, retry_metrics_from_rows
from telos.evaluation.benchmarks.bfcl.subset import load_subset, load_subset_id_map
from telos.evaluation.benchmarks.common import repo_root
from telos.evaluation.benchmarks.suite import SuiteScore


def _skip_category(test_category: str) -> bool:
    u = _ensure_utils()
    return (
        u.is_chatable(test_category)
        or u.is_sql(test_category)
        or u.is_executable(test_category)
        or u.is_memory_prereq(test_category)
    )


def _ensure_utils():
    from telos.evaluation.benchmarks.bfcl.subset import ensure_bfcl_on_path

    ensure_bfcl_on_path()
    from bfcl_eval import utils as u

    return u


def _ensure_checker_model_config(model_id: str) -> None:
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING

    key = model_id.replace("_", "/")
    if key in MODEL_CONFIG_MAPPING:
        return
    ref = "meta-llama/Llama-3.1-8B-Instruct"
    if ref not in MODEL_CONFIG_MAPPING:
        ref = next(iter(MODEL_CONFIG_MAPPING))
    MODEL_CONFIG_MAPPING[key] = MODEL_CONFIG_MAPPING[ref]


def result_path_for_category(
    result_dir: Path,
    model_id: str,
    category: str,
    *,
    sample_id: Optional[str] = None,
) -> Path:
    u = _ensure_utils()
    from bfcl_eval.constants.eval_config import VERSION_PREFIX

    sid = sample_id
    if sid is None:
        id_map = load_subset_id_map()
        ids = id_map.get(category)
        if not ids:
            raise KeyError(f"unknown bfcl category: {category}")
        sid = ids[0]
    group = u.get_directory_structure_by_id(sid)
    return (
        result_dir
        / model_dir_name(model_id)
        / group
        / f"{VERSION_PREFIX}_{category}_result.json"
    )


def dedupe_result_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    u = _ensure_utils()
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_id[str(row["id"])] = row
    return sorted(by_id.values(), key=u.sort_key)


def load_result_rows(
    result_dir: Path,
    model_id: str,
    categories: list[str],
) -> list[dict[str, Any]]:
    u = _ensure_utils()
    id_map = load_subset_id_map()
    rows: list[dict[str, Any]] = []
    for cat in categories:
        ids = id_map.get(cat, [])
        if not ids:
            continue
        path = result_path_for_category(
            result_dir, model_id, cat, sample_id=ids[0]
        )
        if not path.is_file():
            continue
        rows.extend(u.load_file(path, sort_by_id=True))
    return rows


def load_entry_validity(
    score_dir: Path,
    model_id: str,
    categories: list[str],
    *,
    entry_ids: Optional[set[str]] = None,
) -> dict[str, bool]:
    u = _ensure_utils()
    from bfcl_eval.constants.eval_config import VERSION_PREFIX

    slug = model_dir_name(model_id)
    failures: set[str] = set()
    for cat in categories:
        if _skip_category(cat):
            continue
        path = (
            score_dir
            / slug
            / u.get_directory_structure_by_category(cat)
            / f"{VERSION_PREFIX}_{cat}_score.json"
        )
        if not path.is_file():
            continue
        for row in u.load_file(path):
            eid = row.get("id")
            if eid is None:
                continue
            if row.get("valid") is False:
                failures.add(str(eid))
    if entry_ids is None:
        return {eid: eid not in failures for eid in failures}
    return {eid: eid not in failures for eid in entry_ids}


def default_result_dir() -> Path:
    return repo_root() / "results" / "benchmarks" / "bfcl"


def entry_categories(entries: list[dict[str, Any]]) -> list[str]:
    return sorted({str(e["id"]).rsplit("_", 1)[0] for e in entries})


def score(
    model_id: str,
    result_dir: Path,
    entries: list[dict[str, Any]],
    *,
    score_dir: Optional[Path] = None,
    partial: bool = True,
) -> SuiteScore:
    """run gorilla evaluate_task per category; return unified suite score."""
    from telos.evaluation.benchmarks.bfcl.subset import ensure_bfcl_scoring

    ensure_bfcl_scoring()
    from bfcl_eval.eval_checker.eval_runner import evaluate_task

    u = _ensure_utils()
    subset = load_subset()
    cats = entry_categories(entries)
    slug = model_dir_name(model_id)
    model_root = result_dir / slug
    if not model_root.is_dir():
        raise FileNotFoundError(f"no bfcl results for {model_id} under {result_dir.resolve()}")

    out_score = score_dir or (result_dir.parent / "score")
    out_score.mkdir(parents=True, exist_ok=True)
    handler = ResultHandler.from_model_id(model_id)
    _ensure_checker_model_config(model_id)
    leaderboard: dict[str, Any] = {}
    per_category: dict[str, dict[str, Any]] = {}

    from tqdm import tqdm

    score_cats = [c for c in cats if not _skip_category(c)]
    for cat in tqdm(score_cats, desc=f"bfcl scoring ({slug})", unit="category"):
        path = result_path_for_category(
            result_dir, model_id, cat, sample_id=subset.id_map[cat][0]
        )
        if not path.is_file():
            tqdm.write(f"skip {cat}: no result file at {path}")
            continue
        loaded = u.load_file(path, sort_by_id=True)
        tqdm.write(f"scoring {cat} ({len(loaded)} results)...")
        model_result = dedupe_result_rows(loaded)
        leaderboard = evaluate_task(
            cat,
            result_dir,
            out_score,
            model_result,
            slug,
            handler,
            leaderboard,
            allow_missing=partial,
        )
        if slug in leaderboard and cat in leaderboard[slug]:
            per_category[cat] = dict(leaderboard[slug][cat])

    rows = load_result_rows(result_dir, model_id, cats)
    entry_ids = {str(r["id"]) for r in rows}
    validity = load_entry_validity(out_score, model_id, cats, entry_ids=entry_ids)

    primary: Optional[float] = None
    if per_category:
        accs = [v["accuracy"] for v in per_category.values() if "accuracy" in v]
        counts = [v.get("total_count", 0) for v in per_category.values()]
        if accs and sum(counts) > 0:
            primary = sum(a * c for a, c in zip(accs, counts)) / sum(counts)
        elif accs:
            primary = sum(accs) / len(accs)

    summary_path = out_score / slug / "telos_subset_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "model_id": model_id,
                "registry_slug": slug,
                "result_dir": str(result_dir.resolve()),
                "score_dir": str(out_score.resolve()),
                "categories_scored": sorted(per_category.keys()),
                "per_category": per_category,
                "accuracy": primary,
                "retry": retry_metrics_from_rows(rows),
            },
            indent=2,
        )
        + "\n"
    )

    return SuiteScore(
        primary=primary,
        per_domain=per_category,
        validity=validity,
        extra={
            "accuracy": primary,
            "per_category": per_category,
            **retry_metrics_from_rows(rows),
        },
    )


def rows_to_task_results(
    rows: list[dict[str, Any]],
    score: SuiteScore,
) -> list:
    from telos.evaluation.benchmarks.bfcl.common import count_retry_steps
    from telos.evaluation.harness.task import TaskResult, TaskTiming, TaskTokens

    out: list[TaskResult] = []
    for row in rows:
        eid = str(row["id"])
        cat = eid.rsplit("_", 1)[0]
        passed = score.validity.get(eid)
        out.append(
            TaskResult(
                task_id=eid,
                domain=cat,
                success=passed,
                metrics={
                    "retry_steps": count_retry_steps(row.get("result"), eid),
                    "result": row.get("result"),
                },
                timing=TaskTiming(
                    inference_sec=float(row.get("latency") or 0.0),
                    total_sec=float(row.get("latency") or 0.0),
                ),
                tokens=TaskTokens(
                    prompt_tokens=int(row.get("input_token_count") or 0),
                    generated_tokens=int(row.get("output_token_count") or 0),
                ),
            )
        )
    return out
