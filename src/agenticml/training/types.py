from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TrainingPromptField(str, Enum):
    """dataset column used as apply_chat_template input."""
    FRAMES = "frames"
    MESSAGES = "messages"


class TrainingFormat(str, Enum):
    """serialization format for supervised training."""
    AGENTICML = "agenticml"
    CHATML = "chatml"

    @property
    def prompt_field(self) -> TrainingPromptField:
        if self is TrainingFormat.AGENTICML:
            return TrainingPromptField.FRAMES
        return TrainingPromptField.MESSAGES


class TrainingMode(str, Enum):
    """weight-update strategy for supervised training."""
    LORA = "lora"
    FULL = "full"


LORA_LEARNING_RATE = 2e-4
FULL_LEARNING_RATE = 2e-5


@dataclass
class RunConfig:
    model_id: str
    dataset_id: str
    train_split: str
    eval_split: str
    output_dir: str
    project: str
    run_name: str
    mode: TrainingMode = TrainingMode.LORA

    max_length: int = 2048
    per_device_batch_size: int = 1
    grad_accum_steps: int = 32
    learning_rate: float | None = None
    max_grad_norm: float = 1.0
    num_epochs: float = 2.0
    warmup_ratio: float = 0.03
    seed: int = 42
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
