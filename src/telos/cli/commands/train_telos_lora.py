from __future__ import annotations

import argparse

from telos.training.lora_common import RunConfig
from telos.training.telos_lora import run_telos_lora_train


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="kosiasuzu/telos-agent-llama-3.1-8b-init")
    parser.add_argument("--dataset", default="kosiasuzu/telos-agent-trajectory-dataset")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="eval")
    parser.add_argument("--output-dir", default="outputs/telos-lora")
    parser.add_argument("--project", default="telos-agent")
    parser.add_argument("--run-name", default="telos-llama-3.1-8b-lora")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-eval", type=int, default=0)
    parser.add_argument(
        "--adapter-repo-id",
        default="",
        help="if set, push the LoRA adapter to this HF repo after training",
    )
    parser.add_argument(
        "--merged-repo-id",
        default="",
        help="if set, merge adapter into base and push to this HF repo",
    )
    args = parser.parse_args(argv)

    cfg = RunConfig(
        model_id=args.model_id,
        dataset_id=args.dataset,
        train_split=args.train_split,
        eval_split=args.eval_split,
        output_dir=args.output_dir,
        project=args.project,
        run_name=args.run_name,
        max_length=args.max_length,
    )
    run_telos_lora_train(
        cfg,
        limit_train=args.limit_train,
        limit_eval=args.limit_eval,
        adapter_repo_id=args.adapter_repo_id or None,
        merged_repo_id=args.merged_repo_id or None,
    )


if __name__ == "__main__":
    main()
