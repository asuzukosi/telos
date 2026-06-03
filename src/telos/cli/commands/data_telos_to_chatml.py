from __future__ import annotations

import argparse

from telos.dataset_prep.telos_to_chatml import convert_jsonl


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="convert telos frames to chatml messages in each jsonl record",
    )
    parser.add_argument("--input", required=True, help="input jsonl path")
    parser.add_argument("--output", required=True, help="output jsonl path")
    args = parser.parse_args(argv)
    convert_jsonl(args.input, args.output)


if __name__ == "__main__":
    main()
