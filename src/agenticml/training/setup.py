from __future__ import annotations

import os
import random

import numpy as np
import torch
from transformers import TrainingArguments

from agenticml.training.types import (
    FULL_LEARNING_RATE,
    LORA_LEARNING_RATE,
    RunConfig,
    TrainingMode,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_learning_rate(cfg: RunConfig) -> float:
    if cfg.learning_rate is not None:
        return cfg.learning_rate
    if cfg.mode is TrainingMode.FULL:
        return FULL_LEARNING_RATE
    return LORA_LEARNING_RATE


def make_training_args(cfg: RunConfig) -> TrainingArguments:
    return TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=resolve_learning_rate(cfg),
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=cfg.max_grad_norm,
        weight_decay=0.1,
        bf16=True,
        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        report_to=["wandb"],
        run_name=cfg.run_name,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )


def maybe_init_wandb(cfg: RunConfig) -> None:
    os.environ.setdefault("WANDB_PROJECT", cfg.project)
    os.environ.setdefault("WANDB_NAME", cfg.run_name)
