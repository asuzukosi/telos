"""chatml model adapter: ChatMLBackend + run_chatml_swe on mini-swe bash env."""

from __future__ import annotations

from typing import Any

from telos.evaluation.benchmarks.suite import RunContext
from telos.evaluation.benchmarks.swe.env import (
    DEFAULT_MAX_ITERATIONS,
    cleanup_environment,
    environment_for_instance,
)
from telos.evaluation.benchmarks.swe.io import result_row
from telos.evaluation.benchmarks.swe.loop import run_chatml_swe
from telos.evaluation.benchmarks.swe.registry import BashEnvironment, registry_from_env
from telos.evaluation.harness.backends.chatml_backend import ChatMLBackend


def run_one_task(
    backend: ChatMLBackend,
    entry: dict[str, Any],
    ctx: RunContext,
    *,
    env: BashEnvironment | None = None,
    swe_config: dict[str, Any] | None = None,
    max_iterations: int | None = None,
) -> dict[str, Any]:
    owned_env = env is None
    if owned_env:
        env = environment_for_instance(entry, config=swe_config)
    limit = (
        max_iterations
        if max_iterations is not None
        else (ctx.max_iterations or DEFAULT_MAX_ITERATIONS)
    )
    try:
        _, bridge = registry_from_env(env)
        result = run_chatml_swe(
            backend,
            bridge,
            entry,
            max_iterations=limit,
            max_new_tokens=ctx.max_new_tokens,
        )
        return result_row(result, model_id=ctx.model_id, fmt=backend.format)
    finally:
        if owned_env:
            cleanup_environment(env)
