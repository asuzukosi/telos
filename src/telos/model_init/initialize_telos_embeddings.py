"""initialize telos reserved-token embeddings from semantically related tokens."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# map each telos marker to seed token strings to average.
TELOS_SEED_TOKENS: dict[str, list[str]] = {
    "<|goal|>": ["goal", "objective", "purpose", "aim"],
    "<|mission|>": ["mission", "task", "instruction", "assignment", "problem"],
    "<|obs|>": ["observation", "context", "environment", "situation"],
    "<|belief|>": ["belief", "state", "knowledge", "assumption"],
    "<|plan|>": ["plan", "strategy", "approach", "method"],
    "<|think|>": ["think", "reasoning", "thought", "reflection"],
    "<|action|>": ["action", "call", "tool", "command", "invocation", "function"],
    "<|end|>": ["end", "stop", "done", "complete", "finish", "terminate"],
    "<|result|>": ["result", "output", "response", "outcome"],
    "<|feedback|>": ["feedback", "update", "progress", "comment"],
    "<|reward|>": ["reward", "score", "bonus", "credit"],
}

# telos marker -> reserved-token slot index in llama-3.1 vocabulary.
TELOS_RESERVED_SLOT: dict[str, int] = {
    "<|goal|>": 0,
    "<|mission|>": 1,
    "<|obs|>": 2,
    "<|belief|>": 3,
    "<|plan|>": 4,
    "<|think|>": 5,
    "<|action|>": 6,
    "<|end|>": 7,
    "<|result|>": 8,
    "<|feedback|>": 9,
    "<|reward|>": 10,
}


def _seed_token_ids(tokenizer, words: list[str]) -> list[int]:
    """encode seed words and return single-token ids only."""
    out: list[int] = []
    for w in words:
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        if len(ids) == 1:
            out.append(ids[0])
        else:
            print(f"  warning: {w!r} tokenized to {len(ids)} tokens, skipping")
    if not out:
        first = tokenizer.encode(" " + words[0], add_special_tokens=False)
        out = [first[0]]
        print("  fallback: using first sub-word token only")
    return out


def initialize_telos_embeddings(model, tokenizer) -> None:
    """in-place modification of embed_tokens and lm_head rows."""
    if model.config.tie_word_embeddings:
        raise RuntimeError(
            "this script assumes untied embeddings; llama-3.1 has tie_word_embeddings=False. "
            "got tie_word_embeddings=True - the script would need a different code path."
        )

    embed = model.get_input_embeddings().weight
    lm_head = model.get_output_embeddings().weight
    reserved_slot_base = tokenizer.convert_tokens_to_ids("<|reserved_special_token_0|>")

    print(f"embed_tokens device: {embed.device}, dtype: {embed.dtype}")
    print(f"lm_head device:      {lm_head.device}, dtype: {lm_head.dtype}")
    print(f"reserved_slot_0 base id: {reserved_slot_base}")

    for marker, seeds in TELOS_SEED_TOKENS.items():
        slot = TELOS_RESERVED_SLOT[marker]
        reserved_name = f"<|reserved_special_token_{slot}|>"
        target_id = tokenizer.convert_tokens_to_ids(reserved_name)
        print(f"\n{marker} -> {reserved_name} (id={target_id})")

        seed_ids = _seed_token_ids(tokenizer, seeds)
        print(f"  seed token ids: {seed_ids}")

        seed_embed = embed[seed_ids].float().mean(dim=0)
        seed_head = lm_head[seed_ids].float().mean(dim=0)

        with torch.no_grad():
            embed[target_id] = seed_embed.to(embed.dtype)
            lm_head[target_id] = seed_head.to(lm_head.dtype)

        new_norm_embed = embed[target_id].float().norm().item()
        new_norm_head = lm_head[target_id].float().norm().item()
        print(f"  embed norm: {new_norm_embed:.4f}")
        print(f"  lm_head norm: {new_norm_head:.4f}")


def run_initialize_telos_embeddings(
    base_model_id: str,
    *,
    repo_id: str | None = None,
    private: bool = False,
) -> None:
    # tokenizer only: vocab ids for reserved slots and seed words (vocab is unchanged).
    print(f"loading tokenizer: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # model weights: embed_tokens / lm_head rows are updated in place.
    print(f"loading model weights: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("\ninitializing telos marker embeddings...")
    initialize_telos_embeddings(model, tokenizer)

    if not repo_id:
        print("done (no --repo-id; skipped hub push).")
        return

    print(f"\npushing to huggingface hub: {repo_id}")
    commit_message = f"telos embedding init from base {base_model_id}"
    model.push_to_hub(repo_id, commit_message=commit_message, private=private)
    tokenizer.push_to_hub(repo_id, commit_message=commit_message, private=private)
    print("pushed.")
    print("done.")
