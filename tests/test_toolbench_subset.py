import json
from pathlib import Path

import pytest

from telos.evaluation.benchmarks.toolbench.subset import (
    SUBSET_IDS,
    SUBSET_SOURCE,
    load_subset,
    load_subset_id_map,
)


def test_subset_ids_has_ten_entries():
    id_map = load_subset_id_map()
    assert sum(len(v) for v in id_map.values()) == 10
    assert id_map == SUBSET_IDS
    assert "G1_instruction" in id_map


def test_subset_ids_live_in_module():
    assert SUBSET_SOURCE.endswith(":SUBSET_IDS")


def test_load_subset_from_fixture(tmp_path: Path):
    data_root = tmp_path / "toolbench"
    (data_root / "data/test_instruction").mkdir(parents=True)
    (data_root / "data/test_query_ids").mkdir(parents=True)

    instructions = [
        {"query": "task a", "api_list": []},
        {"query": "task b", "api_list": []},
        {"query": "task c", "api_list": []},
    ]
    (data_root / "data/test_instruction/G1_instruction.json").write_text(
        json.dumps(instructions)
    )
    (data_root / "data/test_query_ids/G1_instruction.json").write_text(
        json.dumps(["101", "102", "103"])
    )

    sub = load_subset(
        id_map={"G1_instruction": ["101", "103"]},
        data_root=data_root,
    )
    assert len(sub.entries) == 2
    assert sub.entries[0]["query_id"] == "101"
    assert sub.entries[0]["query"] == "task a"
    assert sub.entries[1]["query_id"] == "103"
    assert sub.entries[1]["query"] == "task c"


def test_load_subset_missing_data_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_subset(
            id_map={"G1_instruction": ["1"]},
            data_root=tmp_path / "missing",
        )
