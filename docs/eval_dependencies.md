# eval dependencies

Benchmark suites: **BFCL**, **ToolBench upstream** (cached tool results), **SWE-bench-Lite** (`swebench` grader), plus optional **format validity** on our models.

## python

```bash
pip install -e ".[eval]"
git submodule update --init --recursive
pip install -e ".[eval-benchmarks]"
```

| extra | packages |
|-------|----------|
| `eval` | torch, datasets, peft, tqdm |
| `eval-benchmarks` | `telos[eval]`, [swebench](https://github.com/princeton-nlp/SWE-bench), editable [`bfcl_eval`](../third_party/gorilla/berkeley-function-call-leaderboard) (gorilla submodule) |

## third_party (git submodules)

Benchmark upstream code lives under `third_party/`. After clone:

```bash
git submodule update --init --recursive
```

| path | upstream |
|------|----------|
| [`third_party/gorilla`](../third_party/gorilla) | https://github.com/ShishirPatil/gorilla — BFCL scoring and ChatML eval path |
| [`third_party/SWE-bench`](../third_party/SWE-bench) | https://github.com/princeton-nlp/SWE-bench — datasets and `run_evaluation` grader |
| [`third_party/mini-swe-agent`](../third_party/mini-swe-agent) | https://github.com/SWE-agent/mini-swe-agent — agent loop for SWE-bench runs |
| [`third_party/ToolBench`](../third_party/ToolBench) | https://github.com/OpenBMB/ToolBench — upstream tool env + inference (`toolbench.inference.server`) |

Benchmark layout: per-suite folders under `src/telos/evaluation/benchmarks/` with a shared `BenchmarkSuite` interface (`suite.py`: `run_one_task`, `score`, `aggregate` via harness envelope).

BFCL subset IDs: `SUBSET_IDS` in [`bfcl/subset.py`](../src/telos/evaluation/benchmarks/bfcl/subset.py) (45 cases, seed 42; excludes irrelevance — misaligned with tool-first Telos). Package: `telos.evaluation.benchmarks.bfcl` (`subset`, `common`, `telos`, `chatml`, `score`, `suite`). Orchestrator: `BFCLSuite` / `run_bfcl_suite` or CLI:

```bash
telos eval-benchmarks --suite bfcl --format telos --model <hf_id> --num-examples 5
telos eval-benchmarks --suite bfcl --format chatml --model <hf_id> --num-examples 5
telos eval-benchmarks --suite format_validity --format telos --model <hf_id> --num-examples 100
```

Format validity always uses `kosiasuzu/telos-agent-trajectory-dataset` / `eval` (hardcoded in `format_validity/suite.py`).

Writes gorilla result files under `results/benchmarks/bfcl/`, scores via upstream `evaluate_task`, and a harness envelope at `results/benchmarks/bfcl/<format>/summary.json`. Re-score without inference: `--score-only`. Optional: `--inject-retry-failure`, `--no-score`.

Scoring imports `bfcl_eval.eval_checker.eval_runner`, which loads gorilla’s full dependency set (`overrides`, `tree_sitter`, etc.). `.[eval-benchmarks]` installs the submodule package; `.[eval]` alone is not enough for BFCL.

ToolBench subset: `SUBSET_IDS` in [`toolbench/subset.py`](../src/telos/evaluation/benchmarks/toolbench/subset.py) pins 10 `G1_instruction` query IDs. Loader: `telos.evaluation.benchmarks.toolbench.subset`. Data root: `TOOLBENCH_DATA` env or `third_party/ToolBench` (`data/test_instruction/G1_instruction.json`, `data/test_query_ids/G1_instruction.json`).

ToolBench cache: [`toolbench/cache.py`](../src/telos/evaluation/benchmarks/toolbench/cache.py) exposes `CachedToolEnv` and `execute_tool_call`. Tool calls run via upstream `get_rapidapi_response(..., api_customization=True)` against local `data/toolenv/tools/*/api.py`; responses are persisted under `data/tool_response_cache/` for replay. No live RapidAPI proxy.

ToolBench driver: `ToolBenchSuite` in [`toolbench/suite.py`](../src/telos/evaluation/benchmarks/toolbench/suite.py) runs telos/chatml backends against `CachedToolEnv` via `registry_from_env`. CLI:

```bash
telos eval-benchmarks --suite toolbench --format telos --model <hf_id> --num-examples 3
```

ToolBench scoring: [`toolbench/convert.py`](../src/telos/evaluation/benchmarks/toolbench/convert.py) converts result traces to upstream tooleval `answer_details` format. [`toolbench/score.py`](../src/telos/evaluation/benchmarks/toolbench/score.py) applies structural pass-rate checks aligned with `eval_pass_rate.py` (Finish in final step, no tool hallucination, non-empty final answer). Converted answers are written under `results/benchmarks/score/<model>/G1_instruction.json`. Optional full GPT judge: set `OPENAI_API_KEY` and `TOOLEVAL_GPT=1`.

SWE-bench-Lite subset: `SUBSET_IDS` in [`swe/subset.py`](../src/telos/evaluation/benchmarks/swe/subset.py) (30 instances, seed 42 from `SWE-Bench_Lite` / `test`). Drivers: [`swe/telos.py`](../src/telos/evaluation/benchmarks/swe/telos.py), [`swe/chatml.py`](../src/telos/evaluation/benchmarks/swe/chatml.py); wire layer: [`swe/prelude.py`](../src/telos/evaluation/benchmarks/swe/prelude.py), [`swe/registry.py`](../src/telos/evaluation/benchmarks/swe/registry.py), [`swe/loop.py`](../src/telos/evaluation/benchmarks/swe/loop.py). Ops: [`eval_swe_bench.md`](eval_swe_bench.md).

Related (reference only):

- SWE-agent (reference): https://github.com/SWE-agent/SWE-agent

If you already use `.[eval-benchmarks]`, you do not need a separate `PYTHONPATH` hack for gorilla. ToolBench imports resolve via `third_party/ToolBench` on `sys.path` (`ensure_toolbench_on_path`) and pyright/pytest `extraPaths`.

## ops

- SWE-bench grading needs Docker and a predictions directory before calling `swebench.harness.run_evaluation`.
- ToolBench eval in this repo uses pinned cache artifacts only (no live RapidAPI in the default setup).
