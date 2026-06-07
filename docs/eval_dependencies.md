# eval dependencies

Benchmark suites: **BFCL**, **ToolBench upstream** (cached tool results), **SWE-bench-Lite** (`swebench` grader), plus optional **format validity** on our models.

Full pipeline from init through eval (with per-step smoke tests): [`recipe.md`](../recipe.md).

## python

```bash
pip install --upgrade pip
pip install -e ".[eval]"
git submodule update --init --recursive

# staged bfcl install (recommended). `pip install -e ".[eval-benchmarks]"` alone often
# fails: pyproject pins bfcl_eval with a relative file:// URL that pip rejects.
pip install -e third_party/gorilla/berkeley-function-call-leaderboard
pip install soundfile   # bfcl scoring: qwen-agent imports this at load time
pip install swebench   # optional; SWE grading only
```

| extra | packages |
|-------|----------|
| `eval` | torch, datasets, tqdm, termcolor (toolbench upstream) |
| `eval-benchmarks` (metadata) | `agenticml[eval]`, [swebench](https://github.com/princeton-nlp/SWE-bench), editable [`bfcl_eval`](../third_party/gorilla/berkeley-function-call-leaderboard) — install bfcl via the staged command above |

Verify:

```bash
python -c "import bfcl_eval; print('bfcl ok')"
```

## toolbench data (~2 GB)

Eval needs the OpenBMB **on-disk tree** under `data/` (`test_instruction`, `test_query_ids`, `toolenv`, `tool_response_cache`). Not a generic HF parquet dataset.

```bash
cd third_party/ToolBench
hf download nullwwg/toolbench-data data.zip --repo-type dataset --local-dir .
python -c "import zipfile; zipfile.ZipFile('data.zip').extractall('.')"
cd ../..
```

**Important:** pass `--repo-type dataset`. Without it, `hf download nullwwg/toolbench-data ...` searches for a *model* repo and fails with `Repository not found`.

| source | status |
|--------|--------|
| [`nullwwg/toolbench-data`](https://huggingface.co/datasets/nullwwg/toolbench-data) | community mirror of OpenBMB `data.zip` (preferred) |
| OpenBMB [Google Drive](https://github.com/OpenBMB/ToolBench) / Tsinghua Cloud | often dead (404); do not rely on upstream wget links |
| `Maurus/ToolBench` on HF | wrong format (flat table; no `toolenv` / cache tree) |
| your own Hub dataset | upload local `data.zip`, then `hf download <you>/toolbench-data data.zip --repo-type dataset` |

Set `TOOLBENCH_DATA` to the ToolBench root if `data/` is not under `third_party/ToolBench`. On minimal hosts without `unzip`, use the `python -c "import zipfile; ..."` one-liner above.

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

Benchmark layout: per-suite folders under `src/agenticml/evaluation/benchmarks/` with a shared `BenchmarkSuite` interface (`suite.py`: `run_one_task`, `score`, `aggregate` via harness envelope). Tests mirror this layout under `tests/evaluation/benchmarks/`.

Matrix runner and published table:

```bash
agenticml eval-run-all --dry-run          # preview suites × formats
agenticml eval-run-all --num-examples 5   # run default merged models
agenticml eval-aggregate-results          # write docs/benchmark_results.md
```

Default models: `kosiasuzu/agenticml-llama3.1-8b-lora-merged` (agenticml format), `kosiasuzu/chatml-llama3.1-8b-lora-merged` (chatml). Override with `--agenticml-model` / `--chatml-model`.

BFCL subset IDs: `SUBSET_IDS` in [`bfcl/subset.py`](../src/agenticml/evaluation/benchmarks/bfcl/subset.py) (45 cases, seed 42; excludes irrelevance — misaligned with tool-first AgenticML). Long multi-turn vehicle/travel cases (e.g. `multi_turn_base_67`) are swapped for shorter 2–3 turn examples in the pinned subset. Package: `agenticml.evaluation.benchmarks.bfcl` (`subset`, `common`, `agenticml`, `chatml`, `score`, `suite`). Orchestrator: `BFCLSuite` / `run_suite("bfcl", ...)` or CLI:

```bash
agenticml eval-benchmarks --suite bfcl --format agenticml --model <hf_id> --num-examples 5
agenticml eval-benchmarks --suite bfcl --format chatml --model <hf_id> --num-examples 5
agenticml eval-benchmarks --suite format_validity --format agenticml --model <hf_id> --num-examples 100
```

Format validity always uses `kosiasuzu/agenticml-agent-trajectory-dataset` / `eval` (hardcoded in `format_validity/suite.py`).

Writes gorilla result files under `results/benchmarks/bfcl/`, scores via upstream `evaluate_task`, and a harness envelope at `results/benchmarks/bfcl/<format>/summary.json`. Re-score without inference: `--score-only`. Optional: `--inject-retry-failure`, `--no-score`.

Scoring imports `bfcl_eval.eval_checker.eval_runner`, which loads gorilla’s full dependency set (`overrides`, `tree_sitter`, etc.). Install the gorilla submodule editable (see staged install above); `.[eval]` alone is not enough for BFCL.

ToolBench subset: `SUBSET_IDS` in [`toolbench/subset.py`](../src/agenticml/evaluation/benchmarks/toolbench/subset.py) pins 10 `G1_instruction` query IDs. Loader: `agenticml.evaluation.benchmarks.toolbench.subset`. Data root: `TOOLBENCH_DATA` env or `third_party/ToolBench` (`data/test_instruction/G1_instruction.json`, `data/test_query_ids/G1_instruction.json`).

ToolBench cache: [`toolbench/cache.py`](../src/agenticml/evaluation/benchmarks/toolbench/cache.py) exposes `CachedToolEnv` and `execute_tool_call`. Tool calls run via upstream `get_rapidapi_response(..., api_customization=True)` against local `data/toolenv/tools/*/api.py`; responses are persisted under `data/tool_response_cache/` for replay. No live RapidAPI proxy.

ToolBench driver: `ToolBenchSuite` in [`toolbench/suite.py`](../src/agenticml/evaluation/benchmarks/toolbench/suite.py) runs agenticml/chatml backends against `CachedToolEnv` via `registry_from_env`. CLI:

```bash
agenticml eval-benchmarks --suite toolbench --format agenticml --model <hf_id> --num-examples 3
```

ToolBench scoring: [`toolbench/convert.py`](../src/agenticml/evaluation/benchmarks/toolbench/convert.py) converts result traces to upstream tooleval `answer_details` format. [`toolbench/score.py`](../src/agenticml/evaluation/benchmarks/toolbench/score.py) applies structural pass-rate checks aligned with `eval_pass_rate.py` (Finish in final step, no tool hallucination, non-empty final answer). Converted answers are written under `results/benchmarks/score/<model>/G1_instruction.json`. Optional full GPT judge: set `OPENAI_API_KEY` and `TOOLEVAL_GPT=1`.

SWE-bench-Lite subset: `SUBSET_IDS` in [`swe/subset.py`](../src/agenticml/evaluation/benchmarks/swe/subset.py) (30 instances, seed 42 from `SWE-Bench_Lite` / `test`). Drivers: [`swe/agenticml.py`](../src/agenticml/evaluation/benchmarks/swe/agenticml.py), [`swe/chatml.py`](../src/agenticml/evaluation/benchmarks/swe/chatml.py); wire layer: [`swe/prelude.py`](../src/agenticml/evaluation/benchmarks/swe/prelude.py), [`swe/registry.py`](../src/agenticml/evaluation/benchmarks/swe/registry.py), [`swe/loop.py`](../src/agenticml/evaluation/benchmarks/swe/loop.py). Ops: [`eval_swe_bench.md`](eval_swe_bench.md).

Related (reference only):

- SWE-agent (reference): https://github.com/SWE-agent/SWE-agent

After the editable `bfcl_eval` install, you do not need a separate `PYTHONPATH` hack for gorilla. ToolBench imports resolve via `third_party/ToolBench` on `sys.path` (`ensure_toolbench_on_path`) and pyright/pytest `extraPaths`.

## ops

- SWE-bench grading needs Docker and a predictions directory before calling `swebench.harness.run_evaluation`.
- ToolBench eval in this repo uses pinned cache artifacts only (no live RapidAPI in the default setup).
