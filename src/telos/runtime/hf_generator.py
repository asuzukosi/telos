from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional, Protocol, Union, cast

import torch

from telos.evaluation.harness.load import AdapterMode, load_model, model_device


class _CausalLMGenerate(Protocol):
    """minimal surface for merged / peft causal lm inference."""

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool,
        pad_token_id: int,
        eos_token_id: Union[int, list[int]],
    ) -> torch.Tensor: ...

    def parameters(self) -> Iterator[torch.nn.Parameter]: ...


@dataclass
class HfGenerator:
    model: _CausalLMGenerate

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        dtype: Optional[torch.dtype] = None,
        adapter_mode: Union[AdapterMode, str] = AdapterMode.MERGED,
        adapter_id: Optional[str] = None,
        **_: Any,
    ) -> HfGenerator:
        resolved_dtype = dtype if dtype is not None else torch.bfloat16
        model = load_model(
            model_name_or_path,
            adapter_mode,
            adapter_id=adapter_id,
            dtype=resolved_dtype,
        )
        return cls(cast(_CausalLMGenerate, model))

    def generate(
        self,
        input_ids: list[int],
        *,
        pad_token_id: int,
        eos_token_id: Union[int, list[int]],
        max_new_tokens: int,
        return_full_sequence: bool = False,
    ) -> list[int]:
        device = model_device(self.model)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        inputs = torch.tensor([input_ids], device=device, dtype=torch.long)
        n = inputs.shape[1]
        with torch.inference_mode():
            out = self.model.generate(
                inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
        full = out[0].tolist()
        return full if return_full_sequence else full[n:]

    def __call__(
        self,
        input_ids: list[int],
        eos_token_id: Union[int, list[int]],
        max_new_tokens: int,
        *,
        pad_token_id: Optional[int] = None,
        return_full_sequence: bool = False,
    ) -> list[int]:
        pad = pad_token_id or (eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id)
        return self.generate(
            input_ids,
            pad_token_id=pad,
            eos_token_id=eos_token_id,
            max_new_tokens=max_new_tokens,
            return_full_sequence=return_full_sequence,
        )
