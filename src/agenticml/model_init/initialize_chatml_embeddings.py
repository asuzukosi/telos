"""initialize chatml special-token embeddings via mean-pooling of seed tokens."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from agenticml.constants import DEFAULT_CHATML_INSTRUCT_TOKENIZER
from agenticml.tokenizer_helpers import single_token_id

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


def _chatml_seed_token_ids(tokenizer: PreTrainedTokenizerBase, seed_words: list[str]) -> list[int]:
    seed_ids: list[int] = []
    for word in seed_words:
        seed_ids.extend(tokenizer.encode(word, add_special_tokens=False))
    return seed_ids


def load_instruct_tokenizer(
    instruct_tokenizer_id: str = DEFAULT_CHATML_INSTRUCT_TOKENIZER,
) -> PreTrainedTokenizerBase:
    """load llama instruct tokenizer for chatml (base model tokenizer has no chat_template)."""
    return AutoTokenizer.from_pretrained(instruct_tokenizer_id)


def initialize_chatml_embeddings(model, tokenizer: PreTrainedTokenizerBase) -> None:
    """in-place mean-pool of seed rows into chatml marker rows in embed_tokens and lm_head."""
    embed = model.get_input_embeddings().weight
    lm_head = model.get_output_embeddings().weight
    assert embed.shape[0] == len(tokenizer), (
        f"vocab size mismatch: embed={embed.shape[0]} tok={len(tokenizer)}"
    )
    print(f"vocab size ok: {embed.shape[0]}")

    print("\ninitializing chatml token rows via mean-pool:")
    with torch.no_grad():
        for marker, seed_words in CHATML_TOKEN_SEEDS.items():
            marker_id = single_token_id(tokenizer, marker)
            if marker_id is None or marker_id == tokenizer.unk_token_id:
                print(f"  skip {marker}: not in vocab")
                continue

            seed_ids = _chatml_seed_token_ids(tokenizer, seed_words)
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
    instruct_tokenizer_id: str = DEFAULT_CHATML_INSTRUCT_TOKENIZER,
    output_dir: str | Path,
    repo_id: str | None = None,
) -> None:
    # model weights: embed_tokens / lm_head rows are updated in place.
    print(f"loading base model weights: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # replace base tokenizer with llama instruct (chat_template required for chatml training).
    print(f"replacing base tokenizer with instruct tokenizer: {instruct_tokenizer_id}")
    tokenizer = load_instruct_tokenizer(instruct_tokenizer_id)

    print("\ninitializing chatml marker embeddings...")
    initialize_chatml_embeddings(model, tokenizer)

    out = Path(output_dir)
    print(f"\nsaving to {out}")
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    if not repo_id:
        print("done (no --repo-id; skipped hub push).")
        return

    print(f"\npushing to huggingface hub: {repo_id}")
    commit_message = (
        f"chatml embedding init from base {base_model_id} "
        f"(tokenizer {instruct_tokenizer_id})"
    )
    model.push_to_hub(repo_id, commit_message=commit_message)  # type: ignore[call-arg]
    tokenizer.push_to_hub(repo_id, commit_message=commit_message)  # type: ignore[call-arg]
    print("pushed.")
    print("done.")
