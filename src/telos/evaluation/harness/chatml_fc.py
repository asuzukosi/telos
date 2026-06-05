"""parse llama-style chatml function-call json from model text."""

from __future__ import annotations

import json
from typing import Any, Optional

_CHATML_STRIP_TOKENS = ("<|python_tag|>", "<|eot_id|>", "<|eom_id|>")


def strip_chat_generation_tokens(text: str) -> str:
    out = (text or "").strip()
    for tok in _CHATML_STRIP_TOKENS:
        out = out.replace(tok, "")
    return out.strip()


def parse_chatml_fc_call(raw: str) -> Optional[dict[str, Any]]:
    """bare ``{name, parameters}`` json without ``<|python_tag|>`` wrapper."""
    text = strip_chat_generation_tokens(raw)
    if not text.startswith("{"):
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if not (obj.get("name") or obj.get("tool")):
        return None
    return obj
