from __future__ import annotations

import argparse

from telos.model_init.initialize_telos_embeddings import run_initialize_telos_embeddings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="initialize telos reserved-token embeddings and optionally push to the hub",
    )
    parser.add_argument(
        "--base-model",
        default="meta-llama/Llama-3.1-8B",
        help="huggingface model id to extend (e.g. Llama-3.1-70B-base)",
    )
    parser.add_argument(
        "--repo-id",
        default="kosiasuzu/telos-agent-llama-3.1-8b-init",
        help="if set, push model and tokenizer to this hub repo after init",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="mark the hub repo private when pushing",
    )
    args = parser.parse_args(argv)

    run_initialize_telos_embeddings(
        args.base_model,
        repo_id=args.repo_id or None,
        private=args.private,
    )


if __name__ == "__main__":
    main()
