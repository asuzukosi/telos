"""verify chatml marker rows match seed-word mean pooling (read-only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from agenticml.model_init.initialize_chatml_embeddings import (
    CHATML_TOKEN_SEEDS,
    _chatml_seed_token_ids,
)
from agenticml.model_init.verify_agenticml_embeddings import _model_embedding_weights
from agenticml.tokenizer_helpers import single_token_id


@dataclass
class ChatMLMarkerEmbeddingReport:
    marker: str
    target_id: int
    seed_ids: list[int]
    n_seeds: int
    embed_norm: float
    lm_head_norm: float
    seed_mean_embed_norm: float
    embed_max_abs_err: float
    lm_head_max_abs_err: float


def verify_chatml_embedding_init(
    model,
    tokenizer: PreTrainedTokenizerBase,
    *,
    atol: float = 0.05,
    min_norm_floor: float = 0.15,
    max_norm_ratio: float = 3.0,
) -> tuple[list[ChatMLMarkerEmbeddingReport], list[str]]:
    """check chatml marker rows equal mean of seed rows."""
    embed, lm_head = _model_embedding_weights(model)
    reports: list[ChatMLMarkerEmbeddingReport] = []
    errors: list[str] = []

    if embed.shape[0] != len(tokenizer):
        errors.append(
            f"vocab size mismatch: embed={embed.shape[0]} tok={len(tokenizer)}"
        )
        return reports, errors

    for marker, seed_words in CHATML_TOKEN_SEEDS.items():
        marker_id = single_token_id(tokenizer, marker)
        if marker_id is None or marker_id == tokenizer.unk_token_id:
            errors.append(f"{marker}: not in tokenizer vocab")
            continue

        seed_ids = _chatml_seed_token_ids(tokenizer, seed_words)
        if not seed_ids:
            errors.append(f"{marker}: no seed token ids")
            continue

        expected_embed = embed[seed_ids].float().mean(dim=0)
        expected_head = lm_head[seed_ids].float().mean(dim=0)
        actual_embed = embed[marker_id].float()
        actual_head = lm_head[marker_id].float()

        embed_err = (actual_embed - expected_embed).abs().max().item()
        head_err = (actual_head - expected_head).abs().max().item()
        embed_norm = actual_embed.norm().item()
        head_norm = actual_head.norm().item()
        seed_mean_norm = expected_embed.norm().item()

        reports.append(
            ChatMLMarkerEmbeddingReport(
                marker=marker,
                target_id=marker_id,
                seed_ids=seed_ids,
                n_seeds=len(seed_ids),
                embed_norm=embed_norm,
                lm_head_norm=head_norm,
                seed_mean_embed_norm=seed_mean_norm,
                embed_max_abs_err=embed_err,
                lm_head_max_abs_err=head_err,
            )
        )

        if embed_err > atol:
            errors.append(f"{marker}: embed row differs from seed mean (max err {embed_err:.6f})")
        if head_err > atol:
            errors.append(f"{marker}: lm_head row differs from seed mean (max err {head_err:.6f})")
        if embed_norm < min_norm_floor:
            errors.append(
                f"{marker}: embed norm {embed_norm:.4f} below floor {min_norm_floor}"
            )
        if head_norm < min_norm_floor:
            errors.append(
                f"{marker}: lm_head norm {head_norm:.4f} below floor {min_norm_floor}"
            )
        if seed_mean_norm > 0:
            ratio = embed_norm / seed_mean_norm
            if ratio > max_norm_ratio or ratio < 1.0 / max_norm_ratio:
                errors.append(
                    f"{marker}: embed/seed-mean norm ratio {ratio:.3f} "
                    f"outside [1/{max_norm_ratio}, {max_norm_ratio}]"
                )

    return reports, errors


def print_chatml_verification_report(
    reports: list[ChatMLMarkerEmbeddingReport],
    errors: list[str],
) -> None:
    print(
        f"{'marker':<28} {'id':>7} {'seeds':>5} "
        f"{'embed_norm':>10} {'head_norm':>10} {'seed_norm':>10} {'embed_err':>10}"
    )
    for r in reports:
        print(
            f"{r.marker:<28} {r.target_id:>7} {r.n_seeds:>5} "
            f"{r.embed_norm:>10.4f} {r.lm_head_norm:>10.4f} {r.seed_mean_embed_norm:>10.4f} "
            f"{r.embed_max_abs_err:>10.6f}"
        )
    if errors:
        print("\nverification failures:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\nverification passed (all markers match seed means, norms ok)")


def run_verify_chatml_embeddings(model_id: str) -> None:
    print(f"loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.chat_template is None:
        raise RuntimeError(
            f"{model_id} tokenizer has no chat_template; expected chatml init checkpoint"
        )

    print(f"loading model weights: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    embed, lm_head = _model_embedding_weights(model)
    print(f"embed_tokens device: {embed.device}, dtype: {embed.dtype}")
    print(f"lm_head device:      {lm_head.device}, dtype: {lm_head.dtype}")

    reports, errors = verify_chatml_embedding_init(model, tokenizer)
    print_chatml_verification_report(reports, errors)
    if errors:
        raise RuntimeError(f"chatml embedding verification failed ({len(errors)} issues)")
