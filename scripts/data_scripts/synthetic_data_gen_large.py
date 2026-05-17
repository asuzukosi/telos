# parallel jsonl generation via openrouter (openai-compatible client)
import json
import os
import time
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PROMPT_PATH = _REPO_ROOT / "prompts" / "synthetic_data_generation.txt"

TARGET = 20000
BATCH = 20
WORKERS = 16
MAX_TOKENS = 16000
MAX_RETRIES = 10
BACKOFF_S = 1.5
OUT_PATH = _REPO_ROOT / "data" / "generated.jsonl"
MODEL = "qwen/qwen3.5-plus-20260420"


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
    client: OpenAI,
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


def main() -> None:
    print(
        f"log: start target={TARGET} batch={BATCH} workers={WORKERS} "
        f"max_tokens={MAX_TOKENS} model={MODEL} out={OUT_PATH}"
    )
    prompt = _PROMPT_PATH.read_text(encoding="utf-8")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    written = count_jsonl_lines(OUT_PATH)
    print(f"log: resume existing_lines={written} need={TARGET - written}")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    def work_unit() -> list[str]:
        return fetch_batch(
            client,
            model=MODEL,
            prompt=prompt,
            batch=BATCH,
            max_tokens=MAX_TOKENS,
            max_retries=MAX_RETRIES,
            base_backoff_s=BACKOFF_S,
        )

    with OUT_PATH.open("a", encoding="utf-8") as f, ThreadPoolExecutor(
        max_workers=max(1, WORKERS),
        thread_name_prefix="synth",
    ) as executor:
        futures = set()
        while written < TARGET or futures:
            while written < TARGET and len(futures) < max(1, WORKERS):
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
                    if written >= TARGET:
                        break
                    f.write(line + "\n")
                    written += 1
                f.flush()
                added = written - before
                print(
                    f"log: flush ok, +{added} lines (batch had {len(lines)}), "
                    f"total={written}/{TARGET}"
                )
    print(f"log: done total_lines={written} path={OUT_PATH}")


if __name__ == "__main__":
    main()
