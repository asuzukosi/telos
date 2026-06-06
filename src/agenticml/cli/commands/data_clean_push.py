from __future__ import annotations

import argparse

from agenticml.dataset_prep.clean_and_push import run_clean_and_push


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="validate, deduplicate, split, and push agenticml trajectory jsonl to huggingface",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="jsonl of generated trajectories (id, frames, domain; optional messages)",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="hf dataset repo id, e.g. your-username/agenticml-trajectories-v1",
    )
    parser.add_argument(
        "--eval-frac",
        type=float,
        default=0.05,
        help="fraction of each domain held out for eval (default 0.05)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="random seed for stratified train/eval split",
    )
    args = parser.parse_args(argv)

    run_clean_and_push(
        args.input,
        args.repo_id,
        eval_frac=args.eval_frac,
        split_seed=args.split_seed,
    )


if __name__ == "__main__":
    main()
