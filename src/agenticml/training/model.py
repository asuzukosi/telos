from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from agenticml.training.types import TrainingMode


def build_lora_model(model_id: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
    )
    return get_peft_model(model, lora_cfg)


def build_full_model(model_id: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable()
    return model


def build_model(model_id: str, mode: TrainingMode):
    if mode is TrainingMode.LORA:
        return build_lora_model(model_id)
    return build_full_model(model_id)


def print_trainable(model) -> None:
    trainable = 0
    total = 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    pct = 100 * trainable / total
    print(f"trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")
