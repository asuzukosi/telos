from __future__ import annotations

import argparse

from telos.evaluation.benchmarks.run import SUITES
from telos.evaluation.benchmarks.run_all import (
    DEFAULT_CHATML_MODEL,
    DEFAULT_TELLOS_MODEL,
    MATRIX_FORMATS,
    MATRIX_SUITES,
    run_matrix,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="run benchmark matrix: suites × formats on default merged models",
    )
    p.add_argument(
        "--suites",
        nargs="+",
        choices=list(SUITES),
        default=None,
        help=f"default: {' '.join(MATRIX_SUITES)}",
    )
    p.add_argument(
        "--formats",
        nargs="+",
        choices=list(MATRIX_FORMATS),
        default=None,
    )
    p.add_argument("--telos-model", default=DEFAULT_TELLOS_MODEL)
    p.add_argument("--chatml-model", default=DEFAULT_CHATML_MODEL)
    p.add_argument("--num-examples", type=int, default=None)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="print matrix only")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--score-only", action="store_true", help="bfcl/swe: skip inference")
    p.add_argument("--no-score", action="store_true", help="bfcl/swe: skip grader")
    args = p.parse_args(argv)

    if args.score_only and args.no_score:
        p.error("cannot use --score-only and --no-score together")

    run_matrix(
        suites=args.suites,
        formats=args.formats,
        models={"telos": args.telos_model, "chatml": args.chatml_model},
        num_examples=args.num_examples,
        sample_seed=args.sample_seed,
        max_new_tokens=args.max_new_tokens,
        run_inference=not args.score_only,
        run_score=not args.no_score,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
    )


if __name__ == "__main__":
    main()
