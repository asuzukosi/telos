"""smoke: 2 real swe-bench-lite entries through persist → preds → score (no docker grader)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from telos.evaluation.benchmarks.suite import RunContext
from telos.evaluation.benchmarks.swe.io import load_result_rows, write_preds
from telos.evaluation.benchmarks.swe.prelude import instance_to_prelude
from telos.evaluation.benchmarks.swe.score import score
from telos.evaluation.benchmarks.swe.subset import load_entries
from telos.evaluation.benchmarks.swe.suite import SWEBenchLiteSuite
from telos.evaluation.benchmarks.swe.telos import run_one_task
from telos.evaluation.harness.backends.telos_backend import TelosBackend
from telos.evaluation.harness.runner import write_benchmark
from telos.evaluation.harness.aggregate import aggregate_efficiency
from telos.evaluation.harness.task import BenchmarkResult, BenchmarkRunMeta
from telos.runtime.hf_generator import HfGenerator
from telos.tokenizer import TelosTokenizer


class _FakeTokenizer:
    end_id = 999_999

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join("<|end|>" if i == self.end_id else chr(i) for i in ids)

    @property
    def hf(self):
        return self


class _ScriptedGenerator:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)

    def __call__(self, input_ids, eos_token_id, max_new_tokens, *, pad_token_id=None, **_):
        del input_ids, pad_token_id
        text = self._responses.pop(0)
        ids = [ord(c) for c in text]
        stop = eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id
        ids.append(stop)
        return ids[:max_new_tokens]


class _FakeEnv:
    def __init__(self, *, submit_patch: str):
        self.submit_patch = submit_patch
        self.commands: list[str] = []

    def execute(self, action: dict, cwd: str = "") -> dict:
        del cwd
        cmd = action.get("command", "")
        self.commands.append(cmd)
        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in cmd:
            from minisweagent.exceptions import Submitted

            raise Submitted(
                {
                    "role": "exit",
                    "content": self.submit_patch,
                    "extra": {"exit_status": "Submitted", "submission": self.submit_patch},
                }
            )
        return {"output": "ok\n", "returncode": 0, "exception_info": ""}


def _backend() -> TelosBackend:
    action = '<|action|>{"tool":"bash","command":"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"}'
    return TelosBackend(
        tokenizer=cast(TelosTokenizer, _FakeTokenizer()),
        generator=cast(HfGenerator, _ScriptedGenerator([action, action])),
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
    ctx = RunContext(model_id="org/swe-smoke-model", format="telos")
    backend = _backend()
    rows = []
    for entry in entries:
        patch = f"--- a/{entry['instance_id']}.py\n+++ b/{entry['instance_id']}.py\n"
        row = run_one_task(
            backend,
            entry,
            ctx,
            env=_FakeEnv(submit_patch=patch),
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
        format="telos",
        adapter_mode="merged",
        dataset=suite.dataset_label(suite.load_dataset()),
        split="subset",
        num_run=2,
    )
    result = BenchmarkResult(
        meta=meta,
        metrics=aggregate_efficiency(tasks, dict(suite_score.extra)),
        tasks=tasks,
    )
    summary_path, _ = write_benchmark(tmp_path / "telos", result)
    envelope = json.loads(summary_path.read_text())
    assert envelope["meta"]["suite"] == "swe"
    assert len(envelope["tasks"]) == 2
    assert envelope["tasks"][0]["task_id"] in {e["instance_id"] for e in entries}

    print("swe smoke ok:", [e["instance_id"] for e in entries])
