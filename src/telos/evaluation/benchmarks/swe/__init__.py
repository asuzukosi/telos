"""swe-bench-lite benchmark package."""

from telos.evaluation.benchmarks.swe.env import (
    DEFAULT_MAX_ITERATIONS,
    REPEAT_COMMAND_LIMIT,
    cleanup_environment,
    default_swe_config,
    environment_for_instance,
)
from telos.evaluation.benchmarks.swe.io import pred_entry, result_row
from telos.evaluation.benchmarks.swe.loop import SweRunResult, run_chatml_swe, run_telos_swe
from telos.evaluation.benchmarks.swe.prelude import instance_to_prelude
from telos.evaluation.benchmarks.swe.registry import SweEnvBridge, registry_from_env
from telos.evaluation.benchmarks.swe import chatml, score, telos
from telos.evaluation.benchmarks.swe.suite import SWEBenchLiteSuite
from telos.evaluation.benchmarks.swe.score import rows_to_task_results as score_rows_to_task_results
from telos.evaluation.benchmarks.swe.subset import (
    SUBSET_IDS,
    SWEBenchLiteSubset,
    load_subset,
    load_subset_ids,
)

__all__ = [
    "DEFAULT_MAX_ITERATIONS",
    "REPEAT_COMMAND_LIMIT",
    "SUBSET_IDS",
    "SWEBenchLiteSubset",
    "SWEBenchLiteSuite",
    "SweEnvBridge",
    "SweRunResult",
    "cleanup_environment",
    "default_swe_config",
    "environment_for_instance",
    "instance_to_prelude",
    "load_subset",
    "load_subset_ids",
    "pred_entry",
    "registry_from_env",
    "result_row",
    "run_chatml_swe",
    "run_telos_swe",
    "chatml",
    "score",
    "score_rows_to_task_results",
    "telos",
]
