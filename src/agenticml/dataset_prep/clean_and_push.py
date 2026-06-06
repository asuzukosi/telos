"""validate, deduplicate, split, and push agenticml trajectory jsonl to huggingface."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict

from agenticml.bridge import bridge
from agenticml.trajectory import Trajectory
from agenticml.validators import validate


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    for i, line in enumerate(path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"  skip line {i}: invalid json ({e.msg})")
            continue
        if "frames" not in obj or "id" not in obj or "domain" not in obj:
            print(f"skipping line: {line} (bad format)")
            continue
        domain = obj["domain"] or "unknown"
        messages = obj.get("messages")
        if not messages:
            messages = bridge.frames_to_messages(obj["frames"])
        records.append({
            "id": obj["id"],
            "frames": obj["frames"],
            "domain": domain,
            "messages": messages,
        })
    return records


def _is_valid(frames: list[dict]) -> tuple[bool, str]:
    try:
        traj = Trajectory(frames)
    except Exception as e:
        return False, f"construction error: {e}"
    violations = validate(traj.to_frames())
    if violations:
        return False, f"validation: {violations[0]}"
    return True, ""


def validate_all(records: list[dict]) -> tuple[list[dict], dict[str, int]]:
    kept: list[dict] = []
    reasons: dict[str, int] = defaultdict(int)
    for r in records:
        ok, reason = _is_valid(r["frames"])
        if ok:
            kept.append(r)
        else:
            reasons[reason[:60]] += 1
    return kept, dict(reasons)


def _normalize_mission(text: str) -> str:
    return " ".join(text.lower().split())


def _mission_text(frames: list[dict]) -> str | None:
    for f in frames:
        if f.get("type") == "mission":
            content = f.get("content")
            return content if isinstance(content, str) else None
    return None


def deduplicate(records: list[dict]) -> tuple[list[dict], int]:
    seen: set[str] = set()
    kept: list[dict] = []
    dropped = 0
    for r in records:
        mission = _mission_text(r["frames"])
        if mission is None:
            kept.append(r)
            continue
        key = hashlib.sha1(_normalize_mission(mission).encode()).hexdigest()
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(r)
    return kept, dropped


def stratified_split(
    records: list[dict],
    eval_frac: float,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_domain[r["domain"]].append(r)

    train: list[dict] = []
    eval_set: list[dict] = []
    for _domain, items in by_domain.items():
        rng.shuffle(items)
        n_eval = max(1, int(round(len(items) * eval_frac))) if len(items) >= 10 else 0
        eval_set.extend(items[:n_eval])
        train.extend(items[n_eval:])
    rng.shuffle(train)
    rng.shuffle(eval_set)
    return train, eval_set


def _to_hf_record(r: dict) -> dict:
    return {
        "id": r["id"],
        "domain": r["domain"],
        "frames": json.dumps(r["frames"], ensure_ascii=False),
        "messages": json.dumps(r["messages"], ensure_ascii=False),
        "num_frames": len(r["frames"]),
    }


def push_to_hub(
    train: list[dict],
    eval_set: list[dict],
    repo_id: str,
) -> None:
    ds = DatasetDict({
        "train": Dataset.from_list([_to_hf_record(r) for r in train]),
        "eval": Dataset.from_list([_to_hf_record(r) for r in eval_set]),
    })
    print(f"\nuploading to {repo_id}...")
    ds.push_to_hub(repo_id)
    print("done.")


def run_clean_and_push(
    input_path: str | Path,
    repo_id: str,
    *,
    eval_frac: float = 0.05,
    split_seed: int = 42,
) -> None:
    path = Path(input_path)
    print(f"loading {path}...")
    generated = _load_jsonl(path)
    print(f"  {len(generated)} records loaded")

    print("validating...")
    valid, reasons = validate_all(generated)
    pct = 100 * len(valid) / max(1, len(generated))
    print(f"  {len(valid)} / {len(generated)} valid ({pct:.1f}%)")
    if reasons:
        print("  rejection reasons:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1])[:10]:
            print(f"    {count:6d}  {reason}")

    print("deduplicating by mission...")
    deduped, dropped = deduplicate(valid)
    print(f"  dropped {dropped} duplicate(s); {len(deduped)} remain")

    domain_counts: dict[str, int] = defaultdict(int)
    for r in deduped:
        domain_counts[r["domain"]] += 1
    print("domain distribution:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"  {count:6d}  {domain}")

    print(f"splitting (eval_frac={eval_frac})...")
    train, eval_set = stratified_split(deduped, eval_frac, seed=split_seed)
    print(f"  train: {len(train)}, eval: {len(eval_set)}")

    print("\npushing to huggingface...")
    push_to_hub(train, eval_set, repo_id)
