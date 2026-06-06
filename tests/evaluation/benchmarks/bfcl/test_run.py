import random
from pathlib import Path

from agenticml.evaluation.benchmarks.bfcl.subset import BFCLSubset
from agenticml.evaluation.benchmarks.common import sample_entries


def _fake_subset(n: int) -> BFCLSubset:
    entries = [{"id": f"simple_python_{i}"} for i in range(n)]
    return BFCLSubset(
        categories=["simple_python"],
        entries=entries,
        id_map={"simple_python": [e["id"] for e in entries]},
        source="test.fake_subset",
    )


def test_sample_entries_all():
    sub = _fake_subset(5)
    assert len(sample_entries(sub.entries, None)) == 5


def test_sample_entries_fixed_seed():
    sub = _fake_subset(20)
    a = sample_entries(sub.entries, 5, seed=99)
    b = sample_entries(sub.entries, 5, seed=99)
    assert [e["id"] for e in a] == [e["id"] for e in b]
    assert len(a) == 5
    rng = random.Random(99)
    expected = sorted(rng.sample(sub.entries, 5), key=lambda e: e["id"])
    assert a == expected
