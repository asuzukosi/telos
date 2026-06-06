from __future__ import annotations

import argparse

from agenticml.constants import DEFAULT_CHATML_INSTRUCT_TOKENIZER
from agenticml.model_init.initialize_agenticml_embeddings import run_initialize_agenticml_embeddings
from agenticml.model_init.initialize_chatml_embeddings import run_initialize_chatml_embeddings
from agenticml.training.types import TrainingFormat

_BASE_MODEL = "meta-llama/Llama-3.1-8B"

_INIT_DEFAULTS: dict[TrainingFormat, dict[str, str]] = {
    TrainingFormat.AGENTICML: {
        "repo_id": "kosiasuzu/agenticml-agent-llama-3.1-8b-init",
        "output_dir": "outputs/agenticml-base-init",
    },
    TrainingFormat.CHATML: {
        "repo_id": "kosiasuzu/chatml-agent-llama-3.1-8b-init",
        "output_dir": "outputs/chatml-base-init",
    },
}


def _apply_format_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = _INIT_DEFAULTS[TrainingFormat(args.format)]
    if args.repo_id is None:
        args.repo_id = defaults["repo_id"]
    if args.output_dir is None:
        args.output_dir = defaults["output_dir"]
    return args


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="initialize agenticml or chatml marker embeddings and optionally push to hub",
    )
    parser.add_argument(
        "--format",
        choices=[f.value for f in TrainingFormat],
        required=True,
        help="agenticml: reserved slots + agentic template; chatml: instruct tokenizer + marker rows",
    )
    parser.add_argument(
        "--base-model",
        default=_BASE_MODEL,
        help="huggingface base model id for weights",
    )
    parser.add_argument(
        "--instruct-tokenizer",
        default=DEFAULT_CHATML_INSTRUCT_TOKENIZER,
        help="chatml only: llama instruct tokenizer (replaces base tokenizer)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="local directory for model + tokenizer after init",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="if set, push model and tokenizer to this hub repo after init",
    )
    args = _apply_format_defaults(parser.parse_args(argv))
    fmt = TrainingFormat(args.format)

    if fmt is TrainingFormat.AGENTICML:
        run_initialize_agenticml_embeddings(
            args.base_model,
            output_dir=args.output_dir,
            repo_id=args.repo_id or None,
        )
        return

    run_initialize_chatml_embeddings(
        args.base_model,
        instruct_tokenizer_id=args.instruct_tokenizer,
        output_dir=args.output_dir,
        repo_id=args.repo_id or None,
    )


if __name__ == "__main__":
    main()
