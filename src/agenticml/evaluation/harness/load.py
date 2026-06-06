"""shared causal lm loading for evaluation harness (cuda:0 + cpu offload)."""

from __future__ import annotations

from typing import Any, Protocol, cast

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel


class CausalLM(Protocol):
    """typed generate surface for causal lms.

    runtime values are ``PreTrainedModel`` instances; this protocol exists because
    huggingface stubs type ``PreTrainedModel.generate`` as a ``Tensor``, not a method.
    """

    def generate(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...


def as_causal_lm(model: PreTrainedModel) -> CausalLM:
    """single cast boundary so call sites can use ``model.generate(...)`` under pyright."""
    return cast(CausalLM, model)


def causal_lm_load_kwargs(dtype: torch.dtype) -> dict[str, Any]:
    """device_map/max_memory for inference; avoids multi-gpu auto split."""
    if not torch.cuda.is_available():
        return {"torch_dtype": dtype, "device_map": "cpu"}
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    # leave headroom for long bfcl chatml prompts + kv during generate
    cap = max(1, int(total_gib - 4))
    return {
        "torch_dtype": dtype,
        "device_map": "auto",
        "max_memory": {0: f"{cap}GiB", "cpu": "100GiB"},
        "offload_buffers": True,
    }


def load_model(model_id: str, dtype: torch.dtype = torch.bfloat16) -> PreTrainedModel:
    """load a merged hf checkpoint for inference."""
    return AutoModelForCausalLM.from_pretrained(model_id, **causal_lm_load_kwargs(dtype))


def model_device(model: PreTrainedModel) -> torch.device:
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device
