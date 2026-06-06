"""load pinned swe-bench-lite subset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from agenticml.evaluation.benchmarks.common import sample_entries

SWE_LITE_DATASET = "princeton-nlp/SWE-Bench_Lite"
SWE_LITE_SPLIT = "test"
SUBSET_SOURCE = "agenticml.evaluation.benchmarks.swe.subset:SUBSET_IDS"

# pinned test instances (30 ids, seed 42 sample from swe-bench_lite test split)
SUBSET_IDS: list[str] = [
    "astropy__astropy-14995",
    "django__django-11099",
    "django__django-11133",
    "django__django-11283",
    "django__django-11422",
    "django__django-12915",
    "django__django-13033",
    "django__django-13315",
    "django__django-13551",
    "django__django-14382",
    "django__django-14752",
    "django__django-14855",
    "django__django-15851",
    "django__django-16400",
    "django__django-16408",
    "django__django-16527",
    "django__django-16816",
    "django__django-17087",
    "matplotlib__matplotlib-23476",
    "matplotlib__matplotlib-25498",
    "matplotlib__matplotlib-26020",
    "pytest-dev__pytest-5413",
    "pytest-dev__pytest-5692",
    "scikit-learn__scikit-learn-13497",
    "sphinx-doc__sphinx-8282",
    "sphinx-doc__sphinx-8474",
    "sympy__sympy-12454",
    "sympy__sympy-16792",
    "sympy__sympy-20442",
    "sympy__sympy-21627",
]


def load_subset_ids(ids: list[str] | None = None) -> list[str]:
    resolved = list(ids if ids is not None else SUBSET_IDS)
    if not resolved:
        raise ValueError(f"{SUBSET_SOURCE}: no instance ids")
    if not all(isinstance(i, str) and i for i in resolved):
        raise ValueError(f"{SUBSET_SOURCE}: ids must be non-empty strings")
    return resolved


@dataclass(frozen=True)
class SWEBenchLiteSubset:
    instance_ids: list[str]
    entries: list[dict[str, Any]]
    dataset: str
    split: str
    source: str = SUBSET_SOURCE

    @property
    def categories(self) -> list[str]:
        return ["swe_bench_lite"]


def _load_dataset_entries(
    instance_ids: list[str],
    *,
    dataset: str = SWE_LITE_DATASET,
    split: str = SWE_LITE_SPLIT,
) -> list[dict[str, Any]]:
    from datasets import Dataset, load_dataset

    ds = cast(Dataset, load_dataset(dataset, split=split))
    by_id: dict[str, dict[str, Any]] = {}
    for i in range(len(ds)):
        row = dict(ds[i])
        by_id[str(row["instance_id"])] = row
    missing = [iid for iid in instance_ids if iid not in by_id]
    if missing:
        raise KeyError(f"instance ids not in {dataset} split={split}: {missing[:5]}")
    return [by_id[iid] for iid in instance_ids]


def load_subset(
    *,
    instance_ids: list[str] | None = None,
    dataset: str = SWE_LITE_DATASET,
    split: str = SWE_LITE_SPLIT,
) -> SWEBenchLiteSubset:
    resolved = load_subset_ids(instance_ids)
    entries = _load_dataset_entries(resolved, dataset=dataset, split=split)
    return SWEBenchLiteSubset(
        instance_ids=resolved,
        entries=entries,
        dataset=dataset,
        split=split,
    )


def load_entries(
    num_examples: int | None,
    *,
    instance_ids: list[str] | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    subset = load_subset(instance_ids=instance_ids)
    return sample_entries(subset.entries, num_examples, seed=seed)
