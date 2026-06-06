"""run a named benchmark suite."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from agenticml.evaluation.benchmarks.bfcl.suite import BFCLSuite
from agenticml.evaluation.benchmarks.format_validity.evaluate import print_summary
from agenticml.evaluation.benchmarks.format_validity.suite import FormatValiditySuite
from agenticml.evaluation.benchmarks.swe.suite import SWEBenchLiteSuite
from agenticml.evaluation.benchmarks.toolbench.suite import ToolBenchSuite
from agenticml.evaluation.benchmarks.suite import BenchmarkSuite, RunContext
from agenticml.evaluation.harness.task import BenchmarkResult


@dataclass(frozen=True)
class _SuiteConfig:
    factory: Callable[[], BenchmarkSuite]
    default_max_new_tokens: int
    default_num_examples: Optional[int]


SUITES: dict[str, _SuiteConfig] = {
    "bfcl": _SuiteConfig(BFCLSuite, 1024, None),
    "format_validity": _SuiteConfig(FormatValiditySuite, 1024, 100),
    "toolbench": _SuiteConfig(ToolBenchSuite, 1024, None),
    "swe": _SuiteConfig(SWEBenchLiteSuite, 1024, None),
}


def run_suite(
    name: str,
    model_id: str,
    fmt: str,
    *,
    output_dir: Optional[Path] = None,
    num_examples: Optional[int] = None,
    sample_seed: int = 42,
    max_new_tokens: Optional[int] = None,
    max_iterations: Optional[int] = None,
    inject_retry_failure: bool = False,
    run_inference: bool = True,
    run_score: bool = True,
    score_dir: Optional[Path] = None,
) -> BenchmarkResult:
    cfg = SUITES[name]
    ctx = RunContext(
        model_id=model_id,
        format=fmt,
        max_new_tokens=max_new_tokens or cfg.default_max_new_tokens,
        inject_retry_failure=inject_retry_failure,
        max_iterations=max_iterations,
    )
    run_kw: dict[str, Any] = {
        "output_dir": output_dir,
        "num_examples": cfg.default_num_examples if num_examples is None else num_examples,
        "sample_seed": sample_seed,
    }
    if name in ("bfcl", "swe"):
        run_kw.update(run_inference=run_inference, run_score=run_score, score_dir=score_dir)
    result = cfg.factory().run(ctx, **run_kw)
    if name == "format_validity":
        print_summary(result.metrics)
    return result
