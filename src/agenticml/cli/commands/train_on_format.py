from __future__ import annotations

import argparse

from agenticml.constants import (
    DEFAULT_BASE_MODEL,
    DEFAULT_CHATML_BASE_MODEL,
    DEFAULT_TRAJECTORY_DATASET,
)
from agenticml.training import RunConfig, TrainingFormat, TrainingMode, run_supervised_train

_FORMAT_DEFAULTS: dict[TrainingFormat, dict[str, str]] = {
    TrainingFormat.AGENTICML: {
        "model_id": DEFAULT_BASE_MODEL,
        "output_dir": "outputs/agenticml",
        "run_name": "agenticml-llama-3.1-8b",
    },
    TrainingFormat.CHATML: {
        "model_id": DEFAULT_CHATML_BASE_MODEL,
        "output_dir": "outputs/chatml",
        "run_name": "chatml-llama-3.1-8b",
    },
}


def _apply_format_defaults(args: argparse.Namespace) -> argparse.Namespace:
    fmt = TrainingFormat(args.format)
    defaults = _FORMAT_DEFAULTS[fmt]
    if args.model_id is None:
        args.model_id = defaults["model_id"]
    if args.output_dir is None:
        args.output_dir = defaults["output_dir"]
    if args.run_name is None:
        args.run_name = defaults["run_name"]
    if args.dataset is None:
        args.dataset = DEFAULT_TRAJECTORY_DATASET
    return args


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--format",
        choices=[f.value for f in TrainingFormat],
        required=True,
        help="agenticml: frames column; chatml: messages column",
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="eval")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--project", default="agenticml-agent")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-eval", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=[m.value for m in TrainingMode],
        default=TrainingMode.LORA.value,
        help="lora: peft adapters (default); full: full-weight fine-tuning",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="override mode default (lora: 2e-4, full: 2e-5)",
    )
    parser.add_argument(
        "--hub-repo-id",
        default="",
        help="merged/full weights repo; lora also pushes adapters to <repo>-adapter",
    )
    args = _apply_format_defaults(parser.parse_args(argv))

    cfg = RunConfig(
        model_id=args.model_id,
        dataset_id=args.dataset,
        train_split=args.train_split,
        eval_split=args.eval_split,
        output_dir=args.output_dir,
        project=args.project,
        run_name=args.run_name,
        mode=TrainingMode(args.mode),
        max_length=args.max_length,
        learning_rate=args.learning_rate,
    )
    hub_repo_id = args.hub_repo_id or None
    training_format = TrainingFormat(args.format)

    run_supervised_train(
        cfg,
        prompt_field=training_format.prompt_field,
        limit_train=args.limit_train,
        limit_eval=args.limit_eval,
        hub_repo_id=hub_repo_id,
    )


if __name__ == "__main__":
    main()
