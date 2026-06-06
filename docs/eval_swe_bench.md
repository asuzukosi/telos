# swe-bench-lite eval (ops)

**SWE-bench-Lite** runs through [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) and grades with upstream [swebench](https://github.com/princeton-nlp/SWE-bench) `run_evaluation`. Telos and ChatML each get a model adapter; the agent loop and grader stay upstream.

## subset

Pinned instance IDs live in `SUBSET_IDS` in [`swe/subset.py`](../src/telos/evaluation/benchmarks/swe/subset.py): **30** tasks sampled with seed 42 from `princeton-nlp/SWE-Bench_Lite` / `test` (300 total).

Loader: `telos.evaluation.benchmarks.swe.subset` (`load_subset_ids`, `load_subset`, `load_entries`).

```python
from telos.evaluation.benchmarks.swe.subset import load_subset

subset = load_subset()
print(subset.instance_ids[0], subset.entries[0]["repo"])
```

## live progress

Per-step action logs (bash command + truncated output):

```bash
SWE_VERBOSE=1 telos eval-benchmarks --suite swe --format telos --model <model> --num-examples 1 --no-score
```

Other signals while a run is in flight:

- **tqdm** postfix shows the current `instance_id`
- **mini-swe** docker logs: `DEBUG:minisweagent.environment` when containers start
- **gpu**: `watch -n1 nvidia-smi` — generation should show util spikes
- **container shell**: `docker exec -it <minisweagent-name> bash` (repo at `/testbed`)

Full trajectory lands in per-instance JSON under `results/benchmarks/swe/<model_slug>/` after each task completes.

## gpu

SWE inference runs the **8B model in a multi-turn loop** (up to 250 steps). It effectively requires a working GPU.

If you see `CUDA initialization: The NVIDIA driver on your system is too old`, PyTorch falls back to CPU and a single step can take hours. Verify before eval:

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'torch cuda:', torch.version.cuda)"
nvidia-smi
```

Fix by aligning **driver** and **PyTorch CUDA build** (reinstall torch for your driver, or update the NVIDIA driver). A 4060 Ti with ~16GB is enough; CPU-only smoke is not practical.

## dependencies

```bash
pip install -e ".[eval-benchmarks]"
git submodule update --init --recursive
```

| component | path / package |
|-----------|----------------|
| SWE-bench grader | `swebench` (pyproject extra) + `third_party/SWE-bench` submodule |
| Agent loop | `third_party/mini-swe-agent` submodule |
| Dataset | Hugging Face `princeton-nlp/SWE-Bench_Lite` (cached on first load) |

## docker

- Eval **requires Docker** (or Singularity on HPC — see mini-swe-agent docs).
- The user running `telos` must access Docker **without sudo** (`docker ps` works in the same shell).
- `docker run` **exit 126** — permission denied on `/var/run/docker.sock` (`sudo usermod -aG docker $USER`, re-login).
- `docker run` **exit 125** + `containerd.sock: connection refused` — **containerd is down**. Fix before eval:

```bash
sudo journalctl -u containerd -n 30 --no-pager
sudo systemctl stop docker containerd
sudo rm -f /run/containerd/containerd.sock
sudo containerd config default | sudo tee /etc/containerd/config.toml
sudo systemctl start containerd && sudo systemctl start docker
sudo systemctl status containerd   # must be active (running)
```

- **Pre-pull instance images** before `telos eval-benchmarks --suite swe`. First pull can take 10–30+ minutes per image; `docker run` during eval only waits ~600s.

```bash
python -c "
from telos.evaluation.benchmarks.swe.env import pull_instance_image
from telos.evaluation.benchmarks.swe.subset import load_entries
for e in load_entries(2, seed=42):
    print('pulling', e['instance_id'])
    pull_instance_image(e)
"
```

- Disk: SWE eval images are large (~tens of GB across many instances).

## mini-swe-agent (reference)

Upstream batch runner:

```bash
cd third_party/mini-swe-agent
pip install -e .
mini-extra swebench \
  --subset lite \
  --split test \
  --model <provider/model> \
  --filter 'django__django-11099|sympy__sympy-12454' \
  -w 1 \
  -o /tmp/swe-out
```

Single-instance debug:

```bash
mini-extra swebench-single --subset lite --split test -i sympy__sympy-12454 --model <model>
```

Docs: `third_party/mini-swe-agent/docs/usage/swebench.md`

## grading (reference)

After predictions exist as `preds.json` / `preds.jsonl`:

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-Bench_Lite \
  --split test \
  --predictions_path /path/to/preds.jsonl \
  --max_workers 4 \
  --run_id telos-swe-smoke
```

Logs: `logs/run_evaluation/` under the working directory. Primary metric: **resolved rate** (instance-level pass).

## telos harness

- Subset: `telos.evaluation.benchmarks.swe.subset`
- Prelude: `instance_to_prelude` builds goal/mission from `problem_statement` + SWE instructions
- Tools: `registry_from_env` maps telos `bash` actions to mini-swe `env.execute` (captures `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` as `model_patch`)
- Loop: `run_telos_swe(backend, bridge, instance)` drives `TelosBackend.step` until submit or limit
- Task entry: `telos.run_one_task` / `chatml.run_one_task` wire docker env + loop; `io.pred_entry` for `preds.json`
- Grading: `telos.evaluation.benchmarks.swe.score` writes `preds.json`, calls `swebench.harness.run_evaluation`, and reads resolved rate from the upstream report

CLI:

```bash
# inference only (no docker grader) — smoke with 3 instances
telos eval-benchmarks --suite swe --format telos --model <model> --num-examples 3 --no-score

# full run: agent loop + swebench grader
telos eval-benchmarks --suite swe --format telos --model <model> --num-examples 3

# grade existing result rows
telos eval-benchmarks --suite swe --format telos --model <model> --score-only
```

Output: `results/benchmarks/swe/<format>/summary.json` (envelope), per-instance rows under `results/benchmarks/swe/<model_slug>/`, grader artifacts under `results/benchmarks/swe/score/<model_slug>/`.

## smoke checklist

1. `docker info` succeeds.
2. `python -c "from telos.evaluation.benchmarks.swe.subset import load_subset; print(len(load_subset().entries))"` → `30`.
3. `mini-extra swebench-single --subset lite --split test -i astropy__astropy-14995 ...` completes one trajectory (optional).
4. Grade a tiny preds file with `run_evaluation` (optional).
