from __future__ import annotations

import argparse
from pathlib import Path

from telos.evaluation.format_validity import evaluate


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--format", required=True, choices=["telos", "chatml"])
    p.add_argument(
        "--model",
        required=True,
        help="base model id or path (merged mode: fused checkpoint; peft mode: base weights)",
    )
    p.add_argument(
        "--adapter-mode",
        default="merged",
        choices=["merged", "peft"],
    )
    p.add_argument(
        "--adapter-id",
        default=None,
        help="lora adapter repo or path (required when --adapter-mode=peft)",
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="eval")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    args = p.parse_args(argv)

    if args.adapter_mode == "peft" and not args.adapter_id:
        p.error("--adapter-mode='peft' requires --adapter-id")

    evaluate(
        model_id=args.model,
        dataset_id=args.dataset,
        split=args.split,
        fmt=args.format,
        output_path=args.output,
        adapter_mode=args.adapter_mode,
        adapter_id=args.adapter_id,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
