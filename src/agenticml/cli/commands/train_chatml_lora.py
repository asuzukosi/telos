from __future__ import annotations

import argparse

from telos.training.lora_common import RunConfig
from telos.training.chatml_lora import run_chatml_lora_train


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="kosiasuzu/chatml-agent-llama-3.1-8b-init")
    ap.add_argument(
        "--tokenizer-id",
        default="",
        help="defaults to model-id (chatml-base-init carries the Instruct tokenizer)",
    )
    ap.add_argument("--dataset", default="kosiasuzu/telos-agent-trajectory-dataset")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval-split", default="eval")
    ap.add_argument("--output-dir", default="outputs/chatml-lora")
    ap.add_argument("--project", default="telos-agent")
    ap.add_argument("--run-name", default="chatml-llama-3.1-8b-lora")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-eval", type=int, default=0)
    ap.add_argument(
        "--adapter-repo-id",
        default="",
        help="if set, push the LoRA adapter to this HF repo after training",
    )
    ap.add_argument(
        "--merged-repo-id",
        default="",
        help="if set, merge adapter into base and push to this HF repo",
    )
    args = ap.parse_args(argv)

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
    run_chatml_lora_train(
        cfg,
        tokenizer_id=args.tokenizer_id,
        limit_train=args.limit_train,
        limit_eval=args.limit_eval,
        adapter_repo_id=args.adapter_repo_id or None,
        merged_repo_id=args.merged_repo_id or None,
    )


if __name__ == "__main__":
    main()
