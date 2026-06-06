# end-to-end recipe: base llama → merged models → evaluation

This walkthrough rebuilds the AgenticML study from scratch: two **merged** checkpoints (AgenticML frames vs ChatML messages) trained on the **same** trajectory dataset, then scored on the same benchmark matrix.

All inference and eval commands load **merged** Hugging Face checkpoints only (`--model kosiasuzu/...-lora-merged`). There is no PEFT / adapter loading path in the CLI.

Copy-paste commands also live in [`command.txt`](command.txt).

**Convention:** run the paired `pytest` line(s) in each phase **before** the `agenticml` / `torchrun` commands below. Post-step checks (`verify-embeddings`, hub loads) run after the command.

---

## prerequisites

| Requirement | Notes |
|-------------|--------|
| Python ≥3.10 | |
| NVIDIA GPU + driver | Training and eval are impractical on CPU |
| Hugging Face access | `hf auth login` |
| Meta Llama 3.1 license | `meta-llama/Llama-3.1-8B` and `meta-llama/Llama-3.1-8B-Instruct` |
| OpenRouter (optional) | Synthetic data generation only (`OPENROUTER_API_KEY` in `.env`) |
| Docker (SWE only) | For `--suite swe` grading |

**Install workspace:**

```bash
pip install -e ".[dev,train,eval,data,data-gen]"
# install a cuda-matched torch wheel from https://pytorch.org/ if needed
wandb login   # optional; training logs to wandb when configured
```

**Smoke — environment OK:**

```bash
pytest tests/test_frames.py tests/test_bridge.py tests/test_sdk.py -q
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

---

## phase 1: init checkpoints (from meta llama base)

Two init bases share `meta-llama/Llama-3.1-8B` weights but differ in tokenizer / embedding rows.

### 1a. agenticml init

Maps AgenticML frame markers (`<|goal|>` … `<|reward|>`) onto Llama reserved slots via mean-pooled seed embeddings.

```bash
pytest tests/test_agentic_template.py -q

agenticml init-embeddings --format agenticml \
  --base-model meta-llama/Llama-3.1-8B \
  --repo-id kosiasuzu/agenticml-agent-llama-3.1-8b-init
```

**Check after init:**

```bash
agenticml verify-embeddings --format agenticml \
  --model kosiasuzu/agenticml-agent-llama-3.1-8b-init
```

### 1b. chatml init

Same base weights; instruct tokenizer vocab; ChatML special-token rows initialized.

```bash
pytest tests/evaluation/harness/backends/test_chatml_backend.py tests/test_tokenizer_helpers.py -q

agenticml init-embeddings --format chatml \
  --base-model meta-llama/Llama-3.1-8B \
  --instruct-tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --repo-id kosiasuzu/chatml-agent-llama-3.1-8b-init
```

**Check after init:**

```bash
agenticml verify-embeddings --format chatml \
  --model kosiasuzu/chatml-agent-llama-3.1-8b-init
```

---

## phase 2: trajectory dataset

### option A — use published dataset (fastest)

```bash
# rows include `frames` (agenticml) and `messages` (chatml) after clean-and-push
# kosiasuzu/agenticml-agent-trajectory-dataset
```

### option B — generate + publish locally

The clean step validates frames, fills missing ChatML `messages` via [`bridge.frames_to_messages`](src/agenticml/bridge.py), dedupes by mission, stratified train/eval split, and pushes to the Hub.

```bash
pytest tests/test_bridge.py tests/evaluation/benchmarks/test_aggregate_results.py -q

# requires OPENROUTER_API_KEY
agenticml data-synthetic-gen \
  --target 500 \
  --workers 4 \
  --out data/generated_smoke.jsonl

agenticml data-clean-push \
  --input data/generated_smoke.jsonl \
  --repo-id kosiasuzu/agenticml-agent-trajectory-dataset
```

**Check after push:**

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('kosiasuzu/agenticml-agent-trajectory-dataset', split='train')
row = ds[0]
assert 'frames' in row and 'messages' in row
print('ok', row['id'], row['domain'])
"
```

---

## phase 3: fine-tune both formats (merged hub push only)

Both runs use the **same dataset**; AgenticML trains on `frames`, ChatML on `messages`. LoRA hub push: adapters to `<hub-repo-id>-adapter`, merged weights to `--hub-repo-id`.

### 3a. agenticml lora → merged

```bash
pytest tests/training/ -q

# single gpu smoke
agenticml train-on-format --format agenticml \
  --model-id kosiasuzu/agenticml-agent-llama-3.1-8b-init \
  --dataset kosiasuzu/agenticml-agent-trajectory-dataset \
  --output-dir outputs/agenticml-lora-smoke \
  --run-name agenticml-lora-smoke \
  --limit-train 32 \
  --limit-eval 8

# full run (multi-gpu)
torchrun --standalone --nproc_per_node=2 -m agenticml.cli.commands.train_on_format --format agenticml \
  --model-id kosiasuzu/agenticml-agent-llama-3.1-8b-init \
  --dataset kosiasuzu/agenticml-agent-trajectory-dataset \
  --output-dir outputs/agenticml-lora-full \
  --run-name agenticml-lora-full \
  --hub-repo-id kosiasuzu/agenticml-llama3.1-8b-lora-merged
```

### 3b. chatml lora → merged

ChatML trains on the dataset `messages` column via `tokenizer.apply_chat_template`.
Use the ChatML init checkpoint for `--model-id` (weights + instruct tokenizer from
`init-embeddings --format chatml`). Hub push publishes adapters to `<hub-repo-id>-adapter`
and merged weights + tokenizer to `--hub-repo-id`.

```bash
agenticml train-on-format --format chatml \
  --model-id kosiasuzu/chatml-agent-llama-3.1-8b-init \
  --dataset kosiasuzu/agenticml-agent-trajectory-dataset \
  --output-dir outputs/chatml-lora-smoke \
  --run-name chatml-lora-smoke \
  --limit-train 32 \
  --limit-eval 8

torchrun --standalone --nproc_per_node=2 -m agenticml.cli.commands.train_on_format --format chatml \
  --model-id kosiasuzu/chatml-agent-llama-3.1-8b-init \
  --dataset kosiasuzu/agenticml-agent-trajectory-dataset \
  --output-dir outputs/chatml-lora-full \
  --run-name chatml-lora-full \
  --hub-repo-id kosiasuzu/chatml-llama3.1-8b-lora-merged
```

**Smoke after training:** confirm `outputs/*/config.json` exists and optional hub repo loads:

```bash
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('kosiasuzu/agenticml-llama3.1-8b-lora-merged')
print('agenticml merged ok', tok.convert_tokens_to_ids('<|reserved_special_token_7|>'))
"
```

---

## phase 4: eval dependencies

See [`docs/eval_dependencies.md`](docs/eval_dependencies.md) for BFCL / ToolBench / SWE layout.

```bash
pytest tests/evaluation/benchmarks/ -q --ignore=tests/evaluation/benchmarks/swe/test_smoke_pipeline.py

pip install -e ".[eval-benchmarks]"
git submodule update --init --recursive
```

ToolBench cached data (one-time):

```bash
cd third_party/ToolBench
huggingface-cli download nullwwg/toolbench-data data.zip --local-dir .
unzip -o data.zip
cd ../..
```

**Check after setup:**

```bash
agenticml eval-run-all --dry-run   # lists 8 matrix cells, no gpu
```

---

## phase 5: per-suite smoke (catch errors before long runs)

Use **merged** model ids. Run these in order; each step should finish in minutes (except first SWE docker pull).

### format validity (parse + structure)

```bash
agenticml eval-benchmarks --suite format_validity --format agenticml \
  --model kosiasuzu/agenticml-llama3.1-8b-lora-merged \
  --num-examples 5 --output-dir results/benchmarks/format_validity

agenticml eval-benchmarks --suite format_validity --format chatml \
  --model kosiasuzu/chatml-llama3.1-8b-lora-merged \
  --num-examples 5 --output-dir results/benchmarks/format_validity
```

### toolbench (cached tools, no live api)

```bash
agenticml eval-benchmarks --suite toolbench --format agenticml \
  --model kosiasuzu/agenticml-llama3.1-8b-lora-merged \
  --num-examples 1 --output-dir results/benchmarks/toolbench
```

### bfcl (subset)

```bash
agenticml eval-benchmarks --suite bfcl --format agenticml \
  --model kosiasuzu/agenticml-llama3.1-8b-lora-merged \
  --num-examples 3 --no-score --output-dir results/benchmarks/bfcl
```

### swe (docker; inference only first)

```bash
# pre-pull images — can take 10–30+ min first time; see docs/eval_swe_bench.md
python -c "
from agenticml.evaluation.benchmarks.swe.env import pull_instance_image
from agenticml.evaluation.benchmarks.swe.subset import load_entries
for e in load_entries(1, seed=42):
    pull_instance_image(e)
"

SWE_VERBOSE=1 agenticml eval-benchmarks --suite swe --format agenticml \
  --model kosiasuzu/agenticml-llama3.1-8b-lora-merged \
  --num-examples 1 --max-iterations 20 --no-score \
  --output-dir results/benchmarks/swe
```

---

## phase 6: full benchmark matrix + publish

```bash
# inference-only smoke across suites (no swe docker grade)
agenticml eval-run-all --num-examples 3 --no-score \
  --suites bfcl toolbench format_validity swe

# full matrix (long; use --continue-on-error)
agenticml eval-run-all --continue-on-error

# re-grade bfcl/swe from saved rows
agenticml eval-run-all --score-only --suites bfcl swe

# publish table
agenticml eval-aggregate-results
```

Outputs: `results/benchmarks/<suite>/<format>/summary.json`. Aggregated markdown: [`docs/benchmark_results.md`](docs/benchmark_results.md).

### publish hub model cards

Edit [`model_cards/`](model_cards/) locally, then push each file as the repo `README.md`:

```bash
hf upload kosiasuzu/agenticml-llama3.1-8b-lora-merged \
  model_cards/agenticml-llama3.1-8b-lora-merged.md README.md \
  --commit-message "update model card"

hf upload kosiasuzu/chatml-llama3.1-8b-lora-merged \
  model_cards/chatml-llama3.1-8b-lora-merged.md README.md \
  --commit-message "update model card"
```

---

## quick reference: default merged models

| Format | Merged checkpoint |
|--------|-------------------|
| agenticml | `kosiasuzu/agenticml-llama3.1-8b-lora-merged` |
| chatml | `kosiasuzu/chatml-llama3.1-8b-lora-merged` |

Init bases (training only): `kosiasuzu/agenticml-agent-llama-3.1-8b-init`, `kosiasuzu/chatml-agent-llama-3.1-8b-init`.

ChatML inference: `AutoTokenizer.from_pretrained(kosiasuzu/chatml-llama3.1-8b-lora-merged)` (instruct chat template pushed with merged weights).

---

## format A/B: same task, two serializations

See [README — same task, two traces](README.md#same-task-two-traces-agenticml-vs-chatml) and model cards for side-by-side examples. Conversion logic: [`src/agenticml/bridge.py`](src/agenticml/bridge.py).
