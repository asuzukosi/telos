"""
prepare and push the synthetic trajectory corpus to huggingface.
pipeline:
  1. load raw jsonl of generated trajectories.
  2. validate each against the telos parser, renderer, and validator.
  3. deduplicate on mission text (exact match after normalization).
  4. mix in the 50 hand-authored seed trajectories (training side only).
  5. stratify-split into train (95%) and eval (5%) by domain.
  6. convert to datasets.Dataset and push to huggingface.
usage:
    python -m scripts.data_scripts.clean_and_push_telos_data \
        --input data/generated.jsonl \
        --seeds data/seeds.jsonl \
        --repo-id your-username/telos-trajectories-v1 \
"""

from __future__ import annotations
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from datasets import Dataset, DatasetDict
from telos.frames import parse, render
from telos.trajectory import Trajectory
from telos.validators import validate

def _load_jsonl(path: Path) -> list[dict]:
    """load a jsonl file as a list of records.
    each line must be an object with 'id', 'frames', 'domain' (optional 'messages').
    falsy domain is stored as 'unknown' for stratification.
    """
    records: list[dict] = []
    for i, line in enumerate(path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"  skip line {i}: invalid JSON ({e.msg})")
            continue
        if "frames" not in obj or "id" not in obj or "domain" not in obj:
            print(f"skipping line: {line} (bad format)")
            continue
        domain = obj["domain"]
        if not domain:
            domain = "unknown"
        records.append({
            "id": obj["id"],
            "frames": obj["frames"],
            "domain": domain,
            "messages": obj.get("messages"),
        })
    return records


def _is_valid(frames: list[dict]) -> tuple[bool, str]:
    """run a record through parser, renderer, and validator.
    returns (ok, reason). reason is empty on success, descriptive on failure.
    """
    try:
        traj = Trajectory(frames)
    except Exception as e:
        return False, f"construction error: {e}"
    rendered = render(traj.to_frames())
    try:
        reparsed = parse(rendered)
    except Exception as e:
        return False, f"re-parse error: {e}"
    if len(reparsed) != len(traj):
        return False, f"frame count changed on round-trip ({len(traj)} -> {len(reparsed)})"
    violations = validate(traj.to_frames())
    if violations:
        return False, f"validation: {violations[0]}"
    return True, ""


def validate_all(records: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """filter records to those that pass validation. return (kept, reasons_counter)."""
    kept: list[dict] = []
    reasons: dict[str, int] = defaultdict(int)
    for r in records:
        ok, reason = _is_valid(r["frames"])
        if ok:
            kept.append(r)
        else:
            # bucket reasons by the first 60 chars for a clean histogram
            reasons[reason[:60]] += 1
    return kept, dict(reasons)


def _normalize_mission(text: str) -> str:
    """lowercase, collapse whitespace, strip punctuation for dedup keying."""
    return " ".join(text.lower().split())


def _mission_text(frames: list[dict]) -> str | None:
    """pull the first mission frame's content. None if absent."""
    for f in frames:
        if f.get("type") == "mission":
            content = f.get("content")
            return content if isinstance(content, str) else None
    return None


def deduplicate(records: list[dict]) -> tuple[list[dict], int]:
    """drop records with duplicate normalized missions. keep first occurrence."""
    seen: set[str] = set()
    kept: list[dict] = []
    dropped = 0
    for r in records:
        mission = _mission_text(r["frames"])
        if mission is None:
            # records without a mission cannot be deduped this way; keep them
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
    """split records into train/eval, stratified by domain."""
    rng = random.Random(seed)
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_domain[r["domain"]].append(r)

    train: list[dict] = []
    eval_set: list[dict] = []
    for domain, items in by_domain.items():
        rng.shuffle(items)
        n_eval = max(1, int(round(len(items) * eval_frac))) if len(items) >= 10 else 0
        eval_set.extend(items[:n_eval])
        train.extend(items[n_eval:])
    rng.shuffle(train)
    rng.shuffle(eval_set)
    return train, eval_set


def _to_hf_record(r: dict) -> dict:
    """shape a record for the HF dataset. frames are stored as JSON strings
    because their structure is heterogeneous (content can be str, dict,
    null, number) which the arrow type system handles awkwardly."""
    return {
        "id": r["id"],
        "domain": r["domain"],
        "frames": json.dumps(r["frames"], ensure_ascii=False),
        "messages": json.dumps(r["messages"], ensure_ascii=False),
        "num_frames": len(r["frames"]),
    }


def push(train: list[dict], eval_set: list[dict], repo_id: str, private: bool) -> None:
    ds = DatasetDict({
        "train": Dataset.from_list([_to_hf_record(r) for r in train]),
        "eval": Dataset.from_list([_to_hf_record(r) for r in eval_set]),
    })
    print(f"\nuploading to {repo_id} (private={private})...")
    ds.push_to_hub(repo_id, private=False)
    print("done.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="JSONL of generated trajectories")
    p.add_argument("--repo-id", required=True,
                   help="HF dataset repo id, e.g. your-username/telos-trajectories-v1")
    args = p.parse_args()

    print(f"loading {args.input}...")
    generated = _load_jsonl(Path(args.input))
    print(f"  {len(generated)} records loaded")

    print("validating...")
    valid, reasons = validate_all(generated)
    print(f"  {len(valid)} / {len(generated)} valid "
          f"({100 * len(valid) / max(1, len(generated)):.1f}%)")
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

    print(f"splitting (eval_frac=0.05)...")
    train, eval_set = stratified_split(deduped, 0.05    )
    print(f"  train: {len(train)}, eval: {len(eval_set)}")

    print("\npushing to huggingface...")
    push(train, eval_set, args.repo_id, True)
    return


if __name__ == "__main__":
    main()