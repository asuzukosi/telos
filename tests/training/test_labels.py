"""tests for training.labels."""

from __future__ import annotations

from agenticml.training.labels import mask_agenticml_runtime_labels


def test_mask_agenticml_runtime_labels():
    agent_ids = {10, 20}
    runtime_ids = {10}
    ids = [10, 1, 2, 20, 3]
    labels = mask_agenticml_runtime_labels(ids, agent_ids, runtime_ids)
    assert labels == [-100, -100, -100, 20, -100]
