import pytest

from agenticml.evaluation.benchmarks.bfcl.subset import (
    SUBSET_IDS,
    SUBSET_SOURCE,
    ensure_bfcl_on_path,
    load_subset,
    load_subset_id_map,
)
from agenticml.evaluation.benchmarks.common import repo_root


@pytest.fixture(scope="module")
def bfcl_available():
    root = repo_root() / "third_party/gorilla/berkeley-function-call-leaderboard"
    if not root.is_dir():
        pytest.skip("gorilla submodule not initialized")
    ensure_bfcl_on_path()
    pytest.importorskip("bfcl_eval")


def test_subset_ids_has_forty_five_entries():
    id_map = load_subset_id_map()
    assert sum(len(v) for v in id_map.values()) == 45
    assert id_map == {k: v for k, v in SUBSET_IDS.items()}
    assert "irrelevance" not in id_map
    assert "memory_kv" not in id_map
    assert "web_search_base" not in id_map


def test_load_bfcl_subset_entries(bfcl_available):
    sub = load_subset()
    assert len(sub.entries) == 45
    assert len(sub.categories) == len(sub.id_map)
    ids = {e["id"] for e in sub.entries}
    expected = {i for ids_per_cat in sub.id_map.values() for i in ids_per_cat}
    assert ids == expected
    assert sub.entries[0].get("question") is not None
    assert sub.source == SUBSET_SOURCE


def test_subset_ids_live_in_module():
    assert "simple_python" in SUBSET_IDS
    assert SUBSET_SOURCE.endswith(":SUBSET_IDS")
