"""run toolbench subset through chatml backend + cached env."""

from __future__ import annotations

from agenticml.evaluation.benchmarks.suite import RunContext
from agenticml.evaluation.benchmarks.toolbench.cache import CachedToolEnv
from agenticml.evaluation.benchmarks.toolbench.convert import trace_messages
from agenticml.evaluation.benchmarks.toolbench.io import result_row
from agenticml.evaluation.benchmarks.toolbench.registry import registry_from_env
from agenticml.evaluation.harness.backends.chatml_backend import ChatMLBackend

MAX_ITERATIONS = 12


def run_one_task(
    backend: ChatMLBackend,
    entry: dict,
    ctx: RunContext,
) -> dict:
    env = CachedToolEnv(entry)
    registry = registry_from_env(env)
    messages = [
        {"role": "system", "content": env.task_description},
        {"role": "user", "content": env.input_description},
    ]
    run = backend.run(
        messages,
        registry,
        max_iterations=MAX_ITERATIONS,
        max_new_tokens=ctx.max_new_tokens,
        strict=False,
    )
    return result_row(
        str(entry["id"]),
        run,
        success=env.check_success() == 1,
        group=str(entry.get("group") or "G1_instruction"),
        query=env.input_description,
        available_tools=env.functions,
        messages=trace_messages(run),
        fmt="chatml",
    )
