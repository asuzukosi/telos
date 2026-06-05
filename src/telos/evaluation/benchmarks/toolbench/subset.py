"""load pinned toolbench G1_instruction subset (3.1)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from telos.evaluation.benchmarks.common import repo_root

TOOLBENCH_ROOT_REL = Path("third_party/ToolBench")
SUBSET_SOURCE = "telos.evaluation.benchmarks.toolbench.subset:SUBSET_IDS"
INSTRUCTION_FILE = Path("data/test_instruction/G1_instruction.json")
QUERY_IDS_FILE = Path("data/test_query_ids/G1_instruction.json")

# pinned G1_instruction smoke subset (10 instruction query_id values)
SUBSET_IDS: dict[str, list[str]] = {
    "G1_instruction": [
        "577",
        "588",
        "608",
        "1073",
        "1572",
        "1856",
        "2121",
        "2144",
        "2213",
        "2354",
    ],
}


def default_data_root() -> Path:
    env = os.environ.get("TOOLBENCH_DATA")
    if env:
        return Path(env)
    return repo_root() / TOOLBENCH_ROOT_REL


def instruction_path(data_root: Path | None = None) -> Path:
    return (data_root or default_data_root()) / INSTRUCTION_FILE


def query_ids_path(data_root: Path | None = None) -> Path:
    return (data_root or default_data_root()) / QUERY_IDS_FILE


def ensure_toolbench_data(data_root: Path | None = None) -> Path:
    root = data_root or default_data_root()
    inst = instruction_path(root)
    qids = query_ids_path(root)
    if not inst.is_file():
        raise FileNotFoundError(
            f"toolbench instruction file not found at {inst}; "
            "set TOOLBENCH_DATA or clone OpenBMB/ToolBench under third_party/ToolBench"
        )
    if not qids.is_file():
        raise FileNotFoundError(
            f"toolbench query id file not found at {qids}; "
            "set TOOLBENCH_DATA or clone OpenBMB/ToolBench under third_party/ToolBench"
        )
    return root


def load_subset_id_map(
    id_map: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    raw = id_map if id_map is not None else SUBSET_IDS
    out: dict[str, list[str]] = {}
    for group, ids in raw.items():
        if not isinstance(group, str) or not isinstance(ids, list) or not ids:
            continue
        if not all(isinstance(i, str) for i in ids):
            raise ValueError(f"{SUBSET_SOURCE}: {group!r} ids must be strings")
        out[group] = ids
    if not out:
        raise ValueError(f"{SUBSET_SOURCE}: no groups with ids")
    return out


def _instruction_index_by_query_id(instructions: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, entry in enumerate(instructions):
        qid = entry.get("query_id")
        if qid is not None:
            out[str(qid)] = i
    return out


def _resolve_instruction_index(
    qid: str,
    instructions: list[dict[str, Any]],
    *,
    file_index: dict[str, int],
    by_query_id: dict[str, int],
) -> int:
    if qid in by_query_id:
        return by_query_id[qid]
    if qid in file_index:
        return file_index[qid]
    if qid.isdigit():
        idx = int(qid)
        if 0 <= idx < len(instructions):
            return idx
    raise KeyError(
        f"query id {qid!r} not found in instruction query_id fields or {QUERY_IDS_FILE}"
    )


def _load_query_id_index(path: Path) -> dict[str, int]:
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        return {str(qid): i for i, qid in enumerate(raw)}
    if isinstance(raw, dict):
        out: dict[str, int] = {}
        for k, v in raw.items():
            if isinstance(v, int):
                out[str(k)] = v
            else:
                out[str(k)] = int(v)
        return out
    raise ValueError(f"{path}: expected list or dict of query ids")


@dataclass(frozen=True)
class ToolBenchSubset:
    group: str
    entries: list[dict[str, Any]]
    id_map: dict[str, list[str]]
    data_root: Path
    source: str = SUBSET_SOURCE

    @property
    def categories(self) -> list[str]:
        return list(self.id_map.keys())


def load_subset(
    *,
    id_map: dict[str, list[str]] | None = None,
    data_root: Path | None = None,
    group: str = "G1_instruction",
) -> ToolBenchSubset:
    """load upstream G1_instruction tasks for pinned query ids."""
    root = ensure_toolbench_data(data_root)
    resolved = load_subset_id_map(id_map)
    if group not in resolved:
        raise KeyError(f"{SUBSET_SOURCE}: missing group {group!r}")

    instructions = json.loads(instruction_path(root).read_text())
    if not isinstance(instructions, list):
        raise ValueError(f"{instruction_path(root)}: expected list of tasks")

    file_index = _load_query_id_index(query_ids_path(root))
    by_query_id = _instruction_index_by_query_id(instructions)
    entries: list[dict[str, Any]] = []
    for qid in resolved[group]:
        idx = _resolve_instruction_index(
            qid,
            instructions,
            file_index=file_index,
            by_query_id=by_query_id,
        )
        entry = dict(instructions[idx])
        entry["query_id"] = qid
        entry["id"] = qid
        entry["group"] = group
        entries.append(entry)

    return ToolBenchSubset(
        group=group,
        entries=entries,
        id_map=resolved,
        data_root=root,
    )
