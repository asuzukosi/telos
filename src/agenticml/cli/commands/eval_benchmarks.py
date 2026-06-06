from __future__ import annotations

import argparse
import json
from pathlib import Path

from telos.evaluation.benchmarks.run import SUITES, run_suite
from telos.evaluation.harness.load import AdapterMode


def _validate(p: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.adapter_mode == AdapterMode.PEFT and not args.adapter_id:
        p.error(f"--adapter-mode={AdapterMode.PEFT.value!r} requires --adapter-id")
    if args.suite in ("bfcl", "swe"):
        if args.score_only and args.no_score:
            p.error(f"{args.suite}: cannot use --score-only and --no-score together")
        if args.inject_retry_failure and args.score_only:
            p.error(f"{args.suite}: --inject-retry-failure requires inference (omit --score-only)")
        if args.suite == "swe" and args.inject_retry_failure:
            p.error("swe: does not support --inject-retry-failure")
    elif args.suite in ("format_validity", "toolbench"):
        if args.score_only or args.no_score:
            p.error(f"{args.suite}: does not support --score-only or --no-score")
        if args.inject_retry_failure:
            p.error(f"{args.suite}: does not support --inject-retry-failure")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="run benchmark suites")
    p.add_argument("--suite", required=True, choices=list(SUITES))
    p.add_argument("--format", required=True, choices=["telos", "chatml"])
    p.add_argument("--model", required=True)
    p.add_argument("--adapter-mode", type=AdapterMode, default=AdapterMode.MERGED, choices=list(AdapterMode))
    p.add_argument("--adapter-id", default=None)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="result root; envelope -> <output-dir>/<format>/summary.json",
    )
    p.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="subset size (bfcl/toolbench/swe: all pinned ids; format_validity: default 100, -1 for all evaluable)",
    )
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="swe: agent loop limit per instance (default 250)",
    )
    p.add_argument(
        "--inject-retry-failure",
        action="store_true",
        help="bfcl only: inject one simulated tool failure per multi-turn step",
    )
    p.add_argument(
        "--score-only",
        action="store_true",
        help="bfcl/swe: skip inference; score existing results",
    )
    p.add_argument(
        "--no-score",
        action="store_true",
        help="bfcl/swe: inference only; skip upstream grader",
    )
    args = p.parse_args(argv)
    _validate(p, args)

    run_kw: dict = {
        "output_dir": args.output_dir,
        "num_examples": args.num_examples,
        "sample_seed": args.sample_seed,
        "adapter_mode": args.adapter_mode.value,
        "adapter_id": args.adapter_id,
        "max_new_tokens": args.max_new_tokens,
        "max_iterations": args.max_iterations,
        "inject_retry_failure": args.inject_retry_failure,
    }
    if args.suite in ("bfcl", "swe"):
        run_kw["run_inference"] = not args.score_only
        run_kw["run_score"] = not args.no_score

    result = run_suite(args.suite, args.model, args.format, **run_kw)
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
