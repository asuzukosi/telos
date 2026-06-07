"""typed helpers around huggingface tokenizer apis with imprecise stubs."""
from __future__ import annotations
from collections.abc import Callable
from typing import Any, cast
from transformers import PreTrainedTokenizerBase
from agenticml.constants import END_MARKER_TOKEN_ID


def chat_template_ids(
    tokenizer: PreTrainedTokenizerBase,
    conversation: Any,
    **kwargs: Any,
) -> list[int]:
    """tokenize a conversation via apply_chat_template.

    render the template to wire text, then encode. some tokenizer builds
    truncate apply_chat_template(..., tokenize=True); render-then-encode avoids that.
    """
    add_special_tokens = kwargs.pop("add_special_tokens", True)
    wire = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        **kwargs,
    )
    if not isinstance(wire, str):
        raise TypeError("apply_chat_template with tokenize=False must return str")
    return tokenizer.encode(wire, add_special_tokens=add_special_tokens)


def single_token_id(tokenizer: PreTrainedTokenizerBase, token: str) -> int | None:
    """map one special-token string to a single vocab id.

    in: e.g. "<|eot_id|>". out: token id, or none if missing or ambiguous.
    """
    convert = cast(Callable[[str], int | list[int] | None], tokenizer.convert_tokens_to_ids)
    tid = convert(token)
    return tid if isinstance(tid, int) else None


def require_single_token_id(tokenizer: PreTrainedTokenizerBase, token: str) -> int:
    """like single_token_id, but raises if the token is not in vocab."""
    tid = single_token_id(tokenizer, token)
    if tid is None:
        raise ValueError(f"token not in vocab: {token!r}")
    return tid


def eos_token_id(tokenizer: PreTrainedTokenizerBase) -> int | None:
    """return tokenizer.eos_token_id when set, else none."""
    eos = tokenizer.eos_token_id
    return eos if isinstance(eos, int) else None


def pad_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    """return pad_token_id, falling back to eos_token_id.

    raises if neither is configured.
    """
    pad = tokenizer.pad_token_id
    if isinstance(pad, int):
        return pad
    eos = eos_token_id(tokenizer)
    if isinstance(eos, int):
        return eos
    raise ValueError("tokenizer has no pad_token_id or eos_token_id")


def agenticml_stop_token_ids(tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """stop ids for agenticml generation: <|end|> plus eos when present."""
    ids = [END_MARKER_TOKEN_ID]
    eos = eos_token_id(tokenizer)
    if isinstance(eos, int) and eos not in ids:
        ids.append(eos)
    return ids


def agenticml_pad_and_stops(tokenizer: PreTrainedTokenizerBase) -> tuple[int, list[int]]:
    """pad + stop ids for agenticml backends; pad falls back to <|end|>."""
    try:
        pad = pad_token_id(tokenizer)
    except ValueError:
        pad = END_MARKER_TOKEN_ID
    return pad, agenticml_stop_token_ids(tokenizer)


def chatml_stop_token_ids(tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """stop ids for chatml generation: <|eot_id|>, <|eom_id|>, and eos."""
    ids: list[int] = []
    unk = tokenizer.unk_token_id
    for token in ("<|eot_id|>", "<|eom_id|>"):
        tid = single_token_id(tokenizer, token)
        if tid is not None and tid != unk:
            ids.append(tid)
    eos = eos_token_id(tokenizer)
    if isinstance(eos, int) and eos not in ids:
        ids.append(eos)
    if ids:
        return ids
    if isinstance(eos, int):
        return [eos]
    raise ValueError("tokenizer has no stop token ids")
