"""run toolbench subset through telos backend + cached env."""

from __future__ import annotations

from telos.evaluation.benchmarks.suite import RunContext
from telos.evaluation.benchmarks.toolbench.cache import CachedToolEnv
from telos.evaluation.benchmarks.toolbench.convert import trace_messages
from telos.evaluation.benchmarks.toolbench.io import result_row
from telos.evaluation.benchmarks.toolbench.registry import registry_from_env
from telos.evaluation.harness.backends.telos_backend import TelosBackend
from telos.trajectory import Trajectory

DEFAULT_GOAL = (
    "You are a helpful assistant that uses tools to complete user tasks. "
    "Always finish with the Finish tool when you have a final answer."
)
MAX_ITERATIONS = 12


def run_one_task(
    backend: TelosBackend,
    entry: dict,
    ctx: RunContext,
) -> dict:
    env = CachedToolEnv(entry)
    registry = registry_from_env(env)
    prelude = [
        {"type": "goal", "content": DEFAULT_GOAL},
        {"type": "mission", "content": f"{env.task_description}\n\nuser query: {env.input_description}"},
    ]
    run = backend.run(
        Trajectory(prelude),
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
        fmt="telos",
    )
