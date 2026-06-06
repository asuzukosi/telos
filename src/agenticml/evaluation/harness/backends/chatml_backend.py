"""chatml backend: messages + HfGenerator multi-step loop."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Optional

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from agenticml.bridge import bridge
from agenticml.constants import TERMINAL_TOOLS
from agenticml.evaluation.harness.backend import BackendRunResult, BackendStepResult
from agenticml.evaluation.harness.backends.common import GenStats, TimedToolRegistry
from agenticml.runtime.hf_generator import HfGenerateFn, HfGenerator
from agenticml.runtime.tools import ToolError, ToolRegistry
from agenticml.tokenizer_helpers import chat_template_ids, chatml_stop_token_ids, pad_token_id
from agenticml.trajectory import Trajectory


def _to_messages(trajectory: Trajectory | list[dict]) -> list[dict]:
    return bridge.coerce_messages(trajectory)


def _parse(text: str):
    parsed = bridge.parse_chatml_generation(text)
    return parsed.tool_call, parsed.text, parsed.stop_reason


def _assistant(call: Optional[dict], text: str, call_id: str) -> dict:
    if call is None:
        return {"role": "assistant", "content": text}
    name, args = bridge.tool_name_args(call)
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}],
    }


@dataclass
class ChatMLBackend:
    tokenizer: PreTrainedTokenizerBase
    generator: HfGenerateFn

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        dtype: Optional[Any] = None,
    ) -> ChatMLBackend:
        import torch

        dt = dtype or torch.bfloat16
        return cls(
            AutoTokenizer.from_pretrained(model_id),
            HfGenerator.from_pretrained(model_id, dtype=dt),
        )

    @property
    def format(self) -> str:
        return "chatml"

    def _generate(self, messages: list[dict], max_new_tokens: int, stats: GenStats) -> str:
        ids = chat_template_ids(
            self.tokenizer,
            messages,
            add_generation_prompt=True,
        )
        stats.prompt_tokens += len(ids)
        t0 = time.perf_counter()
        gen = self.generator(
            ids,
            chatml_stop_token_ids(self.tokenizer),
            max_new_tokens,
            pad_token_id=pad_token_id(self.tokenizer),
        )
        stats.inference_sec += time.perf_counter() - t0
        stats.generated_tokens += len(gen)
        return self.tokenizer.decode(gen, skip_special_tokens=False)

    def step(
        self,
        trajectory: Trajectory | list[dict],
        tools: Optional[list[dict]] = None,
        *,
        max_new_tokens: int = 512,
        strict: bool = True,
    ) -> BackendStepResult:
        del strict
        messages = bridge.inject_tool_schemas(_to_messages(trajectory), tools or [])
        stats = GenStats()
        raw = self._generate(messages, max_new_tokens, stats)
        call, text, stop = _parse(raw)
        if stop.startswith("parse_error"):
            return BackendStepResult(messages=messages, stopped_on=stop, raw_text=raw, prompt_tokens=stats.prompt_tokens, generated_tokens=stats.generated_tokens, inference_sec=stats.inference_sec)
        msg = _assistant(call, text, f"call_{len(messages)}")
        return BackendStepResult(
            messages=messages + [msg], new_messages=[msg], stopped_on=stop, raw_text=raw,
            prompt_tokens=stats.prompt_tokens, generated_tokens=stats.generated_tokens, inference_sec=stats.inference_sec,
        )

    def run(
        self,
        trajectory: Trajectory | list[dict],
        registry: ToolRegistry,
        *,
        max_iterations: int = 10,
        max_new_tokens: int = 512,
        strict: bool = True,
    ) -> BackendRunResult:
        del strict
        messages = bridge.inject_tool_schemas(_to_messages(trajectory), registry.schemas())
        stats = GenStats()
        timed = TimedToolRegistry(registry)
        t0 = time.perf_counter()
        stop, iterations, answer = "max_iterations", 0, None

        for i in range(1, max_iterations + 1):
            iterations = i
            raw = self._generate(messages, max_new_tokens, stats)
            call, text, ps = _parse(raw)
            if ps.startswith("parse_error"):
                stop = ps
                break
            cid = f"call_{len(messages)}"
            messages.append(_assistant(call, text, cid))
            if call is None:
                stop = "terminal_action" if text else "no_action"
                answer = text or None
                break
            name, args = bridge.tool_name_args(call)
            if name in TERMINAL_TOOLS:
                stop = "terminal_action"
                answer = args.get("text") or text if name == "answer" else None
                break
            try:
                payload = {"tool": name, "value": timed.call(name, args)}
            except ToolError as e:
                payload = {"tool": name, "value": str(e)}
            messages.append({"role": "tool", "tool_call_id": cid, "content": json.dumps(payload)})

        return BackendRunResult(
            messages=messages, stopped_on=stop, iterations=iterations, final_answer=answer,
            prompt_tokens=stats.prompt_tokens, generated_tokens=stats.generated_tokens,
            inference_sec=stats.inference_sec, tool_sec=timed.tool_sec, total_sec=time.perf_counter() - t0,
        )
