"""agenticml backend: hub tokenizer + HfGenerator + sdk step/run."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from agenticml.tokenizer_helpers import agenticml_pad_and_stops
from agenticml.evaluation.harness.backend import BackendRunResult, BackendStepResult
from agenticml.evaluation.harness.backends.common import GenStats, TimedToolRegistry, wrap_generate
from agenticml.runtime.hf_generator import HfGenerateFn, HfGenerator
from agenticml.runtime.runtime import run
from agenticml.runtime.tools import ToolRegistry
from agenticml.sdk import step
from agenticml.trajectory import Trajectory


@dataclass
class AgenticMLBackend:
    tokenizer: PreTrainedTokenizerBase
    generator: HfGenerateFn

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        dtype: Optional[Any] = None,
    ) -> AgenticMLBackend:
        import torch

        dt = dtype or torch.bfloat16
        return cls(
            AutoTokenizer.from_pretrained(model_id),
            HfGenerator.from_pretrained(model_id, dtype=dt),
        )

    @property
    def format(self) -> str:
        return "agenticml"

    def _generate(self) -> tuple[Any, GenStats]:
        pad, stops = agenticml_pad_and_stops(self.tokenizer)

        def base(input_ids, _stop, max_new_tokens):
            return self.generator(input_ids, stops, max_new_tokens, pad_token_id=pad)

        return wrap_generate(base)

    def _stats_result(self, stats: GenStats, **kwargs) -> dict[str, Any]:
        return {
            "prompt_tokens": stats.prompt_tokens,
            "generated_tokens": stats.generated_tokens,
            "inference_sec": stats.inference_sec,
            **kwargs,
        }

    def step(
        self,
        trajectory: Trajectory | list[dict],
        *,
        max_new_tokens: int = 512,
        strict: bool = True,
    ) -> BackendStepResult:
        generate, stats = self._generate()
        result = step(
            trajectory,
            tokenizer=self.tokenizer,
            generate=generate,
            max_new_tokens=max_new_tokens,
            strict=strict,
        )
        return BackendStepResult(step=result, **self._stats_result(stats))

    def run(
        self,
        trajectory: Trajectory | list[dict],
        registry: ToolRegistry,
        *,
        max_iterations: int = 10,
        max_new_tokens: int = 512,
        strict: bool = True,
    ) -> BackendRunResult:
        generate, stats = self._generate()
        timed = TimedToolRegistry(registry)
        t0 = time.perf_counter()
        result = run(
            trajectory, timed, tokenizer=self.tokenizer, generate=generate,
            max_iterations=max_iterations, max_new_tokens=max_new_tokens, strict=strict,
        )
        return BackendRunResult(
            run=result,
            stopped_on=result.stopped_on,
            iterations=result.iterations,
            final_answer=result.final_answer,
            tool_sec=timed.tool_sec,
            total_sec=time.perf_counter() - t0,
            **self._stats_result(stats),
        )
