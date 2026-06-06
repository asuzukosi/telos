"""initialize chatml special-token embeddings via mean-pooling of seed tokens."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# chatml markers in the instruct tokenizer; rows are untrained in the base model.
CHATML_TOKEN_SEEDS: dict[str, list[str]] = {
    "<|start_header_id|>": ["start", "begin", "role", "header"],
    "<|end_header_id|>": ["end", "stop", "header", "close"],
    "<|eot_id|>": ["end", "stop", "done", "finish"],
    "<|begin_of_text|>": ["begin", "start", "text"],
    "<|end_of_text|>": ["end", "stop", "text"],
    "<|python_tag|>": ["python", "tool", "call", "function"],
}


def _mean_pool_embedding(token_ids: list[int], weight: torch.Tensor) -> torch.Tensor:
    rows = weight[token_ids]
    return rows.mean(dim=0)


def initialize_chatml_embeddings(model, instruct_tokenizer) -> None:
    """in-place mean-pool of seed rows into chatml marker rows in embed_tokens and lm_head."""
    embed = model.get_input_embeddings().weight
    lm_head = model.get_output_embeddings().weight
    assert embed.shape[0] == len(instruct_tokenizer), (
        f"vocab size mismatch: embed={embed.shape[0]} tok={len(instruct_tokenizer)}"
    )
    print(f"vocab size ok: {embed.shape[0]}")

    print("\ninitializing chatml token rows via mean-pool:")
    with torch.no_grad():
        for marker, seed_words in CHATML_TOKEN_SEEDS.items():
            marker_id = instruct_tokenizer.convert_tokens_to_ids(marker)
            if marker_id is None or marker_id == instruct_tokenizer.unk_token_id:
                print(f"  skip {marker}: not in vocab")
                continue

            seed_ids: list[int] = []
            for word in seed_words:
                ids = instruct_tokenizer.encode(word, add_special_tokens=False)
                seed_ids.extend(ids)

            if not seed_ids:
                print(f"  skip {marker}: no seed ids")
                continue

            new_embed = _mean_pool_embedding(seed_ids, embed)
            new_head = _mean_pool_embedding(seed_ids, lm_head)

            embed[marker_id] = new_embed.to(embed.dtype)
            lm_head[marker_id] = new_head.to(lm_head.dtype)

            print(
                f"  ok   {marker:<24} id={marker_id:>6} "
                f"seeds={len(seed_ids):>2} "
                f"embed_norm={new_embed.norm().item():.3f} "
                f"head_norm={new_head.norm().item():.3f}"
            )


def run_initialize_chatml_embeddings(
    base_model_id: str,
    *,
    instruct_tokenizer_id: str,
    output_dir: str | Path,
    repo_id: str | None = None,
    private: bool = False,
) -> None:
    print(f"loading base model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"loading instruct tokenizer: {instruct_tokenizer_id}")
    instruct_tok = AutoTokenizer.from_pretrained(instruct_tokenizer_id)

    initialize_chatml_embeddings(model, instruct_tok)

    out = Path(output_dir)
    print(f"\nsaving to {out}")
    model.save_pretrained(out)
    instruct_tok.save_pretrained(out)

    if not repo_id:
        print("done (no --repo-id; skipped hub push).")
        return

    print(f"\npushing to huggingface hub: {repo_id}")
    commit_message = (
        f"chatml embedding init from base {base_model_id} "
        f"(tokenizer {instruct_tokenizer_id})"
    )
    model.push_to_hub(repo_id, commit_message=commit_message, private=private)
    instruct_tok.push_to_hub(repo_id, commit_message=commit_message, private=private)
    print("pushed.")
    print("done.")
