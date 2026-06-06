from __future__ import annotations

import argparse

from telos.dataset_prep.synthetic_gen import run_synthetic_gen


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="generate synthetic telos trajectory jsonl via openrouter",
    )
    parser.add_argument("--target", type=int, default=20_000)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=16_000)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--backoff-s", type=float, default=1.5)
    parser.add_argument(
        "--out",
        default=None,
        help="output jsonl path (default: repo data/generated.jsonl)",
    )
    parser.add_argument("--model", default=None, help="openrouter model id")
    args = parser.parse_args(argv)

    run_synthetic_gen(
        target=args.target,
        batch=args.batch,
        workers=args.workers,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        backoff_s=args.backoff_s,
        out_path=args.out,
        model=args.model or "qwen/qwen3.5-plus-20260420",
    )


if __name__ == "__main__":
    main()
