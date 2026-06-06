from __future__ import annotations

import argparse
import sys

from agenticml.constants import DEFAULT_BASE_MODEL, DEFAULT_CHATML_BASE_MODEL
from agenticml.model_init.verify_agenticml_embeddings import run_verify_agenticml_embeddings
from agenticml.model_init.verify_chatml_embeddings import run_verify_chatml_embeddings
from agenticml.training.types import TrainingFormat

_VERIFY_DEFAULTS: dict[TrainingFormat, str] = {
    TrainingFormat.AGENTICML: DEFAULT_BASE_MODEL,
    TrainingFormat.CHATML: DEFAULT_CHATML_BASE_MODEL,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="verify agenticml or chatml init checkpoint embedding rows match seed-word means",
    )
    parser.add_argument(
        "--format",
        choices=[f.value for f in TrainingFormat],
        required=True,
        help="agenticml: reserved slots; chatml: instruct tokenizer marker rows",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="checkpoint to verify (default: format init hub model)",
    )
    args = parser.parse_args(argv)
    fmt = TrainingFormat(args.format)
    model_id = args.model or _VERIFY_DEFAULTS[fmt]

    try:
        if fmt is TrainingFormat.AGENTICML:
            run_verify_agenticml_embeddings(model_id)
        else:
            run_verify_chatml_embeddings(model_id)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
