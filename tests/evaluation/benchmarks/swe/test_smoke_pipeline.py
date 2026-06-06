"""smoke: 2 real swe-bench-lite entries through persist → preds → score (no docker grader)."""

from __future__ import annotations
import json
from pathlib import Path
from agenticml.evaluation.benchmarks.suite import RunContext
from agenticml.evaluation.benchmarks.swe.io import write_preds
from agenticml.evaluation.benchmarks.swe.prelude import instance_to_prelude
from agenticml.evaluation.benchmarks.swe.score import score
from agenticml.evaluation.benchmarks.swe.subset import load_entries
from agenticml.evaluation.benchmarks.swe.suite import SWEBenchLiteSuite
from agenticml.evaluation.benchmarks.swe.agenticml import run_one_task
from agenticml.evaluation.harness.backends.agenticml_backend import AgenticMLBackend
from agenticml.evaluation.harness.runner import write_benchmark
from agenticml.evaluation.harness.aggregate import aggregate_efficiency
from agenticml.evaluation.harness.task import BenchmarkResult, BenchmarkRunMeta
from tests.fake_tokenizer import FakeTokenizer
from tests.fixtures.generators import HfScriptedGenerator
from tests.fixtures.swe import SweFakeEnv


def _backend() -> AgenticMLBackend:
    from tests.wire_fixtures import W_ACTION

    action = f'{W_ACTION}{{"tool":"bash","command":"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"}}'
    return AgenticMLBackend(
        tokenizer=FakeTokenizer(),
        generator=HfScriptedGenerator([action, action]),
    )


def test_swe_smoke_two_real_entries(tmp_path: Path):
    entries = load_entries(2, seed=42)
    assert len(entries) == 2
    for entry in entries:
        assert entry.get("problem_statement")
        prelude = instance_to_prelude(entry)
        assert prelude[1]["type"] == "mission"
        assert entry["problem_statement"][:40] in prelude[1]["content"]

    suite = SWEBenchLiteSuite()
    ctx = RunContext(model_id="org/swe-smoke-model", format="agenticml")
    backend = _backend()
    rows = []
    for entry in entries:
        patch = f"--- a/{entry['instance_id']}.py\n+++ b/{entry['instance_id']}.py\n"
        row = run_one_task(
            backend,
            entry,
            ctx,
            env=SweFakeEnv(submit_patch=patch),
            max_iterations=3,
        )
        rows.append(row)
        suite.persist_task_result(tmp_path, ctx, row)

    loaded = suite.load_result_rows(tmp_path, ctx, entries)
    assert len(loaded) == 2
    assert {r["instance_id"] for r in loaded} == {e["instance_id"] for e in entries}

    preds_path = write_preds(tmp_path / "preds.json", rows, model_id=ctx.model_id)
    preds = json.loads(preds_path.read_text())
    for entry in entries:
        iid = entry["instance_id"]
        assert iid in preds
        assert preds[iid]["model_name_or_path"] == ctx.model_id
        assert preds[iid]["instance_id"] == iid
        assert preds[iid]["model_patch"].startswith("---")

    suite_score = score(ctx.model_id, rows, score_dir=tmp_path / "score", run_grader=False)
    tasks = suite.rows_to_task_results(rows, suite_score)
    assert len(tasks) == 2
    assert all(t.metrics["submitted"] for t in tasks)

    meta = BenchmarkRunMeta(
        suite="swe",
        model=ctx.model_id,
        format="agenticml",
        dataset=suite.dataset_label(suite.load_dataset()),
        split="subset",
        num_run=2,
    )
    result = BenchmarkResult(
        meta=meta,
        metrics=aggregate_efficiency(tasks, dict(suite_score.extra)),
        tasks=tasks,
    )
    summary_path, _ = write_benchmark(tmp_path / "agenticml", result)
    envelope = json.loads(summary_path.read_text())
    assert envelope["meta"]["suite"] == "swe"
    assert len(envelope["tasks"]) == 2
    assert envelope["tasks"][0]["task_id"] in {e["instance_id"] for e in entries}

    print("swe smoke ok:", [e["instance_id"] for e in entries])
