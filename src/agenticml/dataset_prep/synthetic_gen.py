"""parallel synthetic agenticml trajectory jsonl generation via openrouter."""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from agenticml.prompts.synthetic_data_generation import SYNTHETIC_DATA_GENERATION_PROMPT

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_OUT_PATH = _REPO_ROOT / "data" / "generated.jsonl"
_DEFAULT_MODEL = "qwen/qwen3.5-plus-20260420"


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def parse_trajectory_lines(text: str) -> list[str]:
    out: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError:
            continue
        out.append(line)
    return out


def fetch_batch(
    client,
    *,
    model: str,
    prompt: str,
    batch: int,
    max_tokens: int,
    max_retries: int,
    base_backoff_s: float,
) -> list[str]:
    instruction = f"\n\nFor this run, generate exactly {batch} trajectories."
    user_content = prompt + instruction
    tid = threading.current_thread().name

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": user_content}],
            )
            choice = response.choices[0]
            msg = choice.message
            content = getattr(msg, "content", None) if msg is not None else None
            if content is None:
                print(
                    f"log: [{tid}] content is none, retry "
                    f"{attempt + 1}/{max_retries}"
                )
                time.sleep(base_backoff_s * (2**attempt))
                continue
            text = content.strip()
            if not text:
                print(
                    f"log: [{tid}] content empty after strip, retry "
                    f"{attempt + 1}/{max_retries}"
                )
                time.sleep(base_backoff_s * (2**attempt))
                continue
            lines = parse_trajectory_lines(text)
            if not lines:
                print(
                    f"log: [{tid}] no valid jsonl lines, retry "
                    f"{attempt + 1}/{max_retries}"
                )
                time.sleep(base_backoff_s * (2**attempt))
                continue
            print(
                f"log: [{tid}] batch ok, parsed {len(lines)} lines "
                f"(attempt {attempt + 1})"
            )
            return lines
        except Exception as e:
            print(
                f"log: [{tid}] api error, retry {attempt + 1}/{max_retries}: {e}"
            )
            time.sleep(base_backoff_s * (2**attempt))
    print(f"log: [{tid}] batch gave up after {max_retries} retries")
    return []


def run_synthetic_gen(
    *,
    target: int = 20_000,
    batch: int = 20,
    workers: int = 16,
    max_tokens: int = 16_000,
    max_retries: int = 10,
    backoff_s: float = 1.5,
    out_path: str | Path | None = None,
    model: str = _DEFAULT_MODEL,
    prompt: str | None = None,
) -> None:
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()

    out = Path(out_path) if out_path is not None else _DEFAULT_OUT_PATH
    prompt_text = prompt if prompt is not None else SYNTHETIC_DATA_GENERATION_PROMPT

    print(
        f"log: start target={target} batch={batch} workers={workers} "
        f"max_tokens={max_tokens} model={model} out={out}"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    written = count_jsonl_lines(out)
    print(f"log: resume existing_lines={written} need={target - written}")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    def work_unit() -> list[str]:
        return fetch_batch(
            client,
            model=model,
            prompt=prompt_text,
            batch=batch,
            max_tokens=max_tokens,
            max_retries=max_retries,
            base_backoff_s=backoff_s,
        )

    with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(
        max_workers=max(1, workers),
        thread_name_prefix="synth",
    ) as executor:
        futures: set = set()
        while written < target or futures:
            while written < target and len(futures) < max(1, workers):
                futures.add(executor.submit(work_unit))
            if not futures:
                break
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                try:
                    lines = fut.result()
                except Exception as e:
                    print(f"log: worker future failed: {e}")
                    lines = []
                before = written
                for line in lines:
                    if written >= target:
                        break
                    f.write(line + "\n")
                    written += 1
                f.flush()
                added = written - before
                print(
                    f"log: flush ok, +{added} lines (batch had {len(lines)}), "
                    f"total={written}/{target}"
                )
    print(f"log: done total_lines={written} path={out}")
