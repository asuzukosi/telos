from __future__ import annotations

import sys
from collections.abc import Callable

from agenticml.cli.commands import (
    data_clean_push,
    data_synthetic_gen,
    eval_aggregate_results,
    eval_benchmarks,
    eval_run_all,
    init_embeddings,
    train_on_format,
    verify_embeddings,
)

COMMANDS: dict[str, Callable[[list[str] | None], None]] = {
    "train-on-format": train_on_format.main,
    "eval-benchmarks": eval_benchmarks.main,
    "eval-run-all": eval_run_all.main,
    "eval-aggregate-results": eval_aggregate_results.main,
    "init-embeddings": init_embeddings.main,
    "verify-embeddings": verify_embeddings.main,
    "data-clean-push": data_clean_push.main,
    "data-synthetic-gen": data_synthetic_gen.main,
}


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: agenticml <command> [args...]")
        print("commands:", ", ".join(sorted(COMMANDS)))
        return
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        print(f"unknown command: {name}")
        print("commands:", ", ".join(sorted(COMMANDS)))
        sys.exit(1)
    COMMANDS[name](rest)


if __name__ == "__main__":
    main()
