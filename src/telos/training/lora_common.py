from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model

from telos.tokenizer import TelosTokenizer
from telos.trajectory import Trajectory


TELOS_MODEL_MARKERS = ["<|belief|>", "<|plan|>", "<|think|>", "<|action|>", "<|end|>"]


@dataclass
class RunConfig:
    model_id: str
    dataset_id: str
    train_split: str
    eval_split: str
    output_dir: str
    project: str
    run_name: str

    max_length: int = 1536
    per_device_batch_size: int = 1
    grad_accum_steps: int = 32
    learning_rate: float = 2e-4
    max_grad_norm: float = 1.0
    num_epochs: float = 2.0
    warmup_ratio: float = 0.03
    seed: int = 42
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
            "lm_head",       # spec: include lm_head
        ],
    )
    model = get_peft_model(model, lora_cfg)
    return model


def make_training_args(cfg: RunConfig) -> TrainingArguments:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    kwargs = dict(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
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

    if world_size > 1:
        kwargs["fsdp"] = "full_shard auto_wrap"
        kwargs["fsdp_config"] = {
            "use_orig_params": True,
            "activation_checkpointing": True,
        }

    return TrainingArguments(**kwargs)


def maybe_init_wandb(cfg: RunConfig) -> None:
    os.environ.setdefault("WANDB_PROJECT", cfg.project)
    os.environ.setdefault("WANDB_NAME", cfg.run_name)


def print_trainable(model) -> None:
    trainable = 0
    total = 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    pct = 100 * trainable / total
    print(f"trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")


def mask_telos_runtime_labels(
    input_ids: list[int],
    agent_marker_ids: set[int],
    runtime_marker_ids: set[int],
) -> list[int]:
    labels = list(input_ids)
    in_model_block = False
    for i, tok in enumerate(input_ids):
        in_model_block = (tok in agent_marker_ids) and not (tok in runtime_marker_ids)
        if not in_model_block:
            labels[i] = -100
    return labels


def _find_subseq(haystack: list[int], needle: list[int], start: int = 0) -> int:
    if not needle:
        return -1
    n = len(needle)
    for i in range(start, len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return -1


def mask_assistant_only(input_ids: list[int], tokenizer) -> list[int]:
    """build labels so loss applies only on assistant turns (ChatML fallback).

    the model still sees the full conversation in input_ids. labels use -100 on
    positions we do not train on (system, user, tool, etc.). Assistant message
    content and its closing <|eot_id|> copy input_ids so the model learns to
    predict those tokens.
    """
    # start with every position ignored by the loss
    labels = [-100] * len(input_ids)

    start_header = tokenizer.encode("<|start_header_id|>", add_special_tokens=False)
    end_header = tokenizer.encode("<|end_header_id|>", add_special_tokens=False)
    end_of_turn = tokenizer.encode("<|eot_id|>", add_special_tokens=False)
    assistant_role = tokenizer.encode("assistant", add_special_tokens=False)

    pos = 0
    while pos < len(input_ids):
        # find the next <|start_header_id|> ... role ... <|end_header_id|> block
        header_start = _find_subseq(input_ids, start_header, pos)
        if header_start == -1:
            break

        role_start = header_start + len(start_header)
        header_end = _find_subseq(input_ids, end_header, role_start)
        if header_end == -1:
            break

        role_token_ids = input_ids[role_start:header_end]
        content_start = header_end + len(end_header)

        turn_end = _find_subseq(input_ids, end_of_turn, content_start)
        if turn_end == -1:
            turn_end = len(input_ids)

        if role_token_ids == assistant_role:
            # unmask assistant content plus <|eot_id|> (model must learn to end the turn)
            supervise_until = min(turn_end + len(end_of_turn), len(input_ids))
            for j in range(content_start, supervise_until):
                labels[j] = input_ids[j]

        # continue scanning after this message's <|eot_id|>
        pos = turn_end + len(end_of_turn)

    return labels


def tokenize_telos_data_for_training(
    ex: dict,
    *,
    tt: TelosTokenizer,
    max_length: int,
) -> dict:
    frames = json.loads(ex["frames"])
    trajectory = Trajectory(frames)
    ids = tt.apply_trajectory_template(trajectory, tokenize=True, return_tensors="pt")
    if isinstance(ids, torch.Tensor):
        if ids.ndim == 2 and ids.shape[0] == 1:
            ids = ids[0]
        ids = ids.tolist()
    if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
        ids = ids[0]
    if not isinstance(ids, list) or len(ids) == 0 or not isinstance(ids[0], int):
        return {"input_ids": [], "labels": [], "attention_mask": []}

    agent_marker_ids = {tt.belief_id, tt.plan_id, tt.think_id, tt.action_id, tt.end_id}
    runtime_marker_ids = {
        tt.goal_id,
        tt.mission_id,
        tt.obs_id,
        tt.result_id,
        tt.feedback_id,
        tt.reward_id,
    }
    labels = mask_telos_runtime_labels(ids, agent_marker_ids, runtime_marker_ids)
    ids, labels = truncate(ids, labels, max_length)
    attn = [1] * len(ids)
    return {"input_ids": ids, "labels": labels, "attention_mask": attn}


def tokenize_chatml_data_for_training(ex: dict, *, tokenizer, max_length: int) -> dict:
    try:
        messages = json.loads(ex["messages"])
    except Exception:
        return {"input_ids": [], "labels": [], "attention_mask": []}

    if not isinstance(messages, list) or len(messages) == 0:
        return {"input_ids": [], "labels": [], "attention_mask": []}

    ids = None
    assistant_mask = None
    try:
        out = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        ids = out["input_ids"]
        assistant_mask = out.get("assistant_tokens_mask", None)
    except Exception:
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        except Exception:
            return {"input_ids": [], "labels": [], "attention_mask": []}

    if ids is None or len(ids) == 0:
        return {"input_ids": [], "labels": [], "attention_mask": []}

    if assistant_mask is not None and len(assistant_mask) == len(ids):
        labels = [tok if m else -100 for tok, m in zip(ids, assistant_mask)]
    else:
        labels = mask_assistant_only(ids, tokenizer)

    ids, labels = truncate(ids, labels, max_length)
    if len(ids) == 0:
        return {"input_ids": [], "labels": [], "attention_mask": []}
    if all(x == -100 for x in labels):
        return {"input_ids": [], "labels": [], "attention_mask": []}

    attn = [1] * len(ids)
    return {"input_ids": ids, "labels": labels, "attention_mask": attn}


def truncate(ids: List[int], labels: List[int], max_length: int):
    if len(ids) > max_length:
        ids = ids[:max_length]
        labels = labels[:max_length]
    return ids, labels


def _flat(v):
    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
        return v[0]
    return v


def causal_lm_collator(features: list[dict], pad_token_id: int) -> dict:
    features = [f for f in features if len(f.get("input_ids", [])) > 0]
    if not features:
        raise ValueError("Empty batch after filtering invalid features")

    for f in features:
        f["input_ids"] = _flat(f["input_ids"])
        f["attention_mask"] = _flat(f["attention_mask"])
        f["labels"] = _flat(f["labels"])

    max_len = max(len(f["input_ids"]) for f in features)

    input_ids, attention_mask, labels = [], [], []
    for f in features:
        n = len(f["input_ids"])
        pad = max_len - n
        input_ids.append(f["input_ids"] + [pad_token_id] * pad)
        attention_mask.append(f["attention_mask"] + [0] * pad)
        labels.append(f["labels"] + [-100] * pad)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def maybe_push_artifacts(
    *,
    model,
    tokenizer,
    adapter_repo_id: str | None = None,
    merged_repo_id: str | None = None,
) -> None:
    if adapter_repo_id:
        model.push_to_hub(adapter_repo_id)
        tokenizer.push_to_hub(adapter_repo_id)

    if merged_repo_id:
        merged = model.merge_and_unload()
        merged.push_to_hub(merged_repo_id)
        tokenizer.push_to_hub(merged_repo_id)
