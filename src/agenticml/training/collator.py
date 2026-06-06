from __future__ import annotations

import torch


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
