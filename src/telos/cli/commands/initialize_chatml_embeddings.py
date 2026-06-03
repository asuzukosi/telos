from __future__ import annotations

import argparse

from telos.model_init.initialize_chatml_embeddings import run_initialize_chatml_embeddings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="initialize chatml special-token embeddings, save locally, optionally push to hub",
    )
    parser.add_argument(
        "--base-model",
        default="meta-llama/Llama-3.1-8B",
        help="huggingface base model id (embedding table sized for instruct vocab)",
    )
    parser.add_argument(
        "--instruct-tokenizer",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="tokenizer with chatml tokens in vocab",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/chatml-base-init",
        help="local directory for model + tokenizer after init",
    )
    parser.add_argument(
        "--repo-id",
        default="kosiasuzu/chatml-agent-llama-3.1-8b-init",
        help="if set, push model and tokenizer to this hub repo after save",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="mark the hub repo private when pushing",
    )
    args = parser.parse_args(argv)

    run_initialize_chatml_embeddings(
        args.base_model,
        instruct_tokenizer_id=args.instruct_tokenizer,
        output_dir=args.output_dir,
        repo_id=args.repo_id or None,
        private=args.private,
    )


if __name__ == "__main__":
    main()
