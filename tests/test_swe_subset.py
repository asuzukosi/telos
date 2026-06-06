import pytest

from telos.evaluation.benchmarks.swe.subset import (
    SUBSET_IDS,
    SUBSET_SOURCE,
    load_subset_ids,
)


def test_subset_ids_has_thirty_entries():
    ids = load_subset_ids()
    assert len(ids) == 30
    assert ids == SUBSET_IDS
    assert ids[0] == "astropy__astropy-14995"
    assert all("__" in iid for iid in ids)


def test_subset_ids_live_in_module():
    assert SUBSET_SOURCE.endswith(":SUBSET_IDS")


def test_load_subset_ids_override():
    assert load_subset_ids(["foo__bar-1", "baz__qux-2"]) == ["foo__bar-1", "baz__qux-2"]


def test_load_subset_ids_empty_override_raises():
    with pytest.raises(ValueError):
        load_subset_ids([])
