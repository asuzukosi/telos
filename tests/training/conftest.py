"""shared fixtures for training tests."""

from __future__ import annotations

from agenticml.training.types import RunConfig, TrainingMode


def run_config(
    *,
    mode: TrainingMode = TrainingMode.LORA,
    learning_rate: float | None = None,
) -> RunConfig:
    return RunConfig(
        model_id="test/model",
        dataset_id="test/dataset",
        train_split="train",
        eval_split="eval",
        output_dir="outputs/test",
        project="test",
        run_name="test-run",
        mode=mode,
        learning_rate=learning_rate,
    )
