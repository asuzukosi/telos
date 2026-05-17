"""
format-validity evaluation for telos and chatml LoRA-trained models.

for each example in the eval split, the model is given the runtime prelude
(everything up to where the model should start generating) and asked to
complete the trajectory. the completion is then parsed and validated by
the format's own rules. pass/fail is recorded with a reason and aggregated
by domain.

Usage:
    # if the model checkpoint is a merged model (adapter already fused into
    # base weights, single self-contained checkpoint):
    python -m scripts.format_validity_eval \
        --format telos \
        --model kosiasuzu/telos-llama-3.1-8b-lora-merged \
        --adapter-mode merged \
        --dataset kosiasuzu/telos-agent-trajectory-dataset \
        --split eval \
        --output results/telos_format_validity.json

    # if the checkpoint is a LoRA adapter that needs to be loaded on top of
    # a base model explicitly:
    python -m scripts.format_validity_eval \
        --format chatml \
        --base-model meta-llama/Llama-3.1-8B \
        --model kosiasuzu/chatml-llama-3.1-8b-lora \
        --adapter-mode peft \
        --dataset kosiasuzu/telos-agent-trajectory-dataset \
        --split eval \
        --output results/chatml_format_validity.json

outputs a json file with per-example results and an aggregate summary,
plus a printed table.
"""

from __future__ import annotations
import argparse
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from telos.frames import parse as telos_parse, render as telos_render, Frame
from telos.constants import FrameType
from telos.validators import validate as telos_validate


def load_model_and_tokenizer(
    model_id: str,
    adapter_mode: str,
    base_model_id: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
):
    """load a model in one of two modes.
    args:
      model_id:      HF repo id or local path for the model to evaluate.
      adapter_mode:  "merged" (model_id is a self-contained checkpoint
                     with the adapter already fused) or "peft" (model_id is
                     a LoRA adapter to load on top of base_model_id).
      base_model_id: required when adapter_mode == "peft". the base model
                     the adapter sits on top of.
      dtype:         the torch dtype for model weights.
    returns:
      (model, tokenizer) ready for inference
    """
    if adapter_mode == "merged":
        # self-contained checkpoint - load directly
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
        )
    elif adapter_mode == "peft":
        if not base_model_id:
            raise ValueError(
                "adapter_mode='peft' requires --base-model to specify the "
                "base checkpoint to load the LoRA adapter on top of."
            )
        # import here so the script still runs in 'merged' mode without peft
        # installed.
        try:
            from peft import PeftModel
        except ImportError as e:
            raise ImportError(
                "adapter_mode='peft' requires the 'peft' package. "
                "Install with: pip install peft"
            ) from e

        # Load the base, then attach the adapter.
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=dtype,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, model_id)
        # note: NOT calling .merge_and_unload() - that would merge the adapter
        # into base weights in-memory, doubling peak VRAM. Inference works fine
        # with the adapter attached and is only marginally slower.
    else:
        raise ValueError(
            f"adapter_mode must be 'merged' or 'peft', got: {adapter_mode!r}"
        )

    return model, tokenizer


@dataclass
class ExampleResult:
    """per-example evaluation result."""
    id: str
    domain: str
    parsed_ok: bool
    structurally_valid: bool
    parse_error: Optional[str] = None
    validation_errors: list[str] = None
    num_generated_tokens: int = 0
    generated_text_preview: str = ""

    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []


def build_telos_prompt(frames: list[dict]) -> str:
    """
    build the input prompt for a telos eval example.
    takes everything up to (but not including) the first model-owned frame.
    the model is then expected to generate the model block(s) and any
    intermediate runtime/model alternations needed to reach a terminal action.
    """
    # find the first model-owned frame to know where to cut
    model_owned_types = {"belief", "plan", "think", "action"}
    cut_idx = None
    for i, f in enumerate(frames):
        if f.get("type") in model_owned_types:
            cut_idx = i
            break

    if cut_idx is None:
        # no model frames in this example - degenerate, skip
        return ""

    prelude_frames_dicts = frames[:cut_idx]
    # convert dicts to Frame objects for rendering
    prelude_frames = [
        Frame(type=FrameType(f["type"]), content=f["content"])
        for f in prelude_frames_dicts
        if f["type"] != "end"  # safety: drop any stray end frames from old data
    ]
    return telos_render(prelude_frames)


def build_chatml_prompt(messages: list[dict], tokenizer) -> str:
    """build the input prompt for a chatml eval example.
    takes the system + user messages and any leading conversation up to
    where the first assistant turn should appear. uses the model's
    chat template with add_generation_prompt=True.
    """
    # cut at the first assistant message
    cut_idx = None
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            cut_idx = i
            break

    if cut_idx is None:
        return ""

    prelude_messages = messages[:cut_idx]
    # use the model's chat template; rely on it being a llama-3.1 chat template
    # so add_generation_prompt produces the assistant header.
    return tokenizer.apply_chat_template(
        prelude_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

def generate_completion(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 1024,
    stop_token_ids: Optional[list[int]] = None,
) -> tuple[str, int]:
    """generate a completion from the model.
    returns (generated_text, num_generated_tokens). the generated_text
    is decoded with the telos marker substitutions if applicable - the
    caller is responsible for any post-decoding aliasing.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for reproducibility
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=stop_token_ids or tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    return generated_text, len(generated_ids)



def check_telos_output(
    prelude_frames: list[dict],
    generated_text: str,
) -> tuple[bool, bool, Optional[str], list[str]]:
    """check whether the generated text is a valid telos completion.
    returns (parsed_ok, structurally_valid, parse_error, validation_errors).
    """
    # parse the generated text as a partial trajectory. strict=False because
    # the model may legitimately emit anything; want to know whether the
    # parser can recover frames, not whether ownership is perfect.
    try:
        generated_frames = telos_parse(generated_text, strict=False)
    except Exception as e:
        return False, False, f"parse failure: {e}", []

    # assemble the full trajectory: prelude + generated
    full_frames = []
    for f in prelude_frames:
        if f.get("type") == "end":
            continue
        full_frames.append(Frame(type=FrameType(f["type"]), content=f["content"]))
    full_frames.extend(generated_frames)

    # run validator
    try:
        violations = telos_validate([f.__dict__ if hasattr(f, "__dict__") else f for f in full_frames])
    except Exception as e:
        return True, False, None, [f"validator crashed: {e}"]

    validation_errors = [
        f"[{v.rule}] frame {v.frame_index}: {v.message}" if hasattr(v, "rule")
        else str(v)
        for v in violations
    ]
    structurally_valid = len(validation_errors) == 0

    return True, structurally_valid, None, validation_errors


# llama-3.1 tool-call format: <|python_tag|>{json}<|eom_id|> or <|eot_id|>
_TOOL_CALL_RE = re.compile(
    r"<\|python_tag\|>(.+?)<\|(?:eom_id|eot_id)\|>",
    re.DOTALL,
)
_ASSISTANT_TEXT_RE = re.compile(
    r"(?:<\|start_header_id\|>assistant<\|end_header_id\|>)?\s*(.*?)<\|(?:eot_id|eom_id)\|>",
    re.DOTALL,
)


def check_chatml_output(generated_text: str) -> tuple[bool, bool, Optional[str], list[str]]:
    """check whether the generated text is a valid chatml+tools completion.
    returns (parsed_ok, structurally_valid, parse_error, validation_errors).
    validity for chatml means:
      - the output terminates with <|eot_id|> or <|eom_id|> (model knows to stop)
      - if there's a <|python_tag|> tool call, the JSON parses
      - the output contains either non-empty text or a tool call (not both empty)
    """
    errors: list[str] = []

    # check termination - if there's no eot/eom marker, the model ran to max_tokens
    if not re.search(r"<\|(?:eot_id|eom_id)\|>", generated_text):
        errors.append("no stop token emitted - model ran to max_new_tokens")
        return False, False, "missing stop token", errors

    # check for tool calls
    tool_match = _TOOL_CALL_RE.search(generated_text)
    has_tool_call = False
    if tool_match:
        tool_json = tool_match.group(1).strip()
        try:
            parsed_call = json.loads(tool_json)
            if not isinstance(parsed_call, dict):
                errors.append("tool call payload is not a JSON object")
            elif "name" not in parsed_call:
                errors.append("tool call missing 'name' field")
            else:
                has_tool_call = True
        except json.JSONDecodeError as e:
            return True, False, None, [f"tool call JSON invalid: {e.msg}"]

    # check assistant content
    text_match = _ASSISTANT_TEXT_RE.search(generated_text)
    has_text = bool(text_match and text_match.group(1).strip())

    # either tool call or text content - not both empty
    if not has_tool_call and not has_text:
        errors.append("generation has no tool call and no text content")
        return True, False, None, errors

    return True, len(errors) == 0, None, errors


def evaluate(
    model_id: str,
    dataset_id: str,
    split: str,
    fmt: str,
    output_path: Path,
    *,
    adapter_mode: str = "merged",
    base_model_id: Optional[str] = None,
    limit: Optional[int] = None,
    max_new_tokens: int = 1024,
) -> None:
    """run format-validity evaluation.

    args:
      model_id:       HF repo id or local path
      dataset_id:     HF dataset repo id
      split:          dataset split name (typically "eval")
      fmt:            "telos" or "chatml"
      output_path:    where to write the per-example results json
      adapter_mode:   "merged" (single checkpoint) or "peft" (adapter on base)
      base_model_id:  required when adapter_mode == "peft"
      limit:          if set, only evaluate the first n examples
      max_new_tokens: generation length cap
    """
    print(f"loading model from {model_id} (adapter_mode={adapter_mode})...")
    model, tokenizer = load_model_and_tokenizer(
        model_id=model_id,
        adapter_mode=adapter_mode,
        base_model_id=base_model_id,
    )
    model.eval()

    # set up stop tokens per format
    if fmt == "telos":
        # stop on <|end|> - it's token 128015 on llama-3.1
        # if the tokenizer doesn't have it, fall back to eos
        try:
            end_id = tokenizer.convert_tokens_to_ids("<|end|>")
            if end_id == tokenizer.unk_token_id:
                end_id = tokenizer.convert_tokens_to_ids("<|reserved_special_token_7|>")
            stop_ids = [end_id, tokenizer.eos_token_id]
        except Exception:
            stop_ids = [tokenizer.eos_token_id]
    else:
        # chatml: stop on eot_id or eom_id
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        eom = tokenizer.convert_tokens_to_ids("<|eom_id|>")
        stop_ids = [t for t in [eot, eom, tokenizer.eos_token_id] if t is not None]

    print(f"loading dataset {dataset_id} split={split}...")
    ds = load_dataset(dataset_id, split=split)

    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    print(f"evaluating {len(ds)} examples in {fmt} format...")

    results: list[ExampleResult] = []
    start_time = time.time()

    for i, row in enumerate(ds):
        example_id = row["id"]
        domain = row["domain"]

        # load the appropriate format's frames
        if fmt == "telos":
            frames = json.loads(row["frames"])
            prompt = build_telos_prompt(frames)
            if not prompt:
                continue
        else:
            messages = json.loads(row["messages"])
            prompt = build_chatml_prompt(messages, tokenizer)
            if not prompt:
                continue

        # generate
        try:
            generated_text, num_tokens = generate_completion(
                model, tokenizer, prompt,
                max_new_tokens=max_new_tokens,
                stop_token_ids=stop_ids,
            )
        except Exception as e:
            results.append(ExampleResult(
                id=example_id,
                domain=domain,
                parsed_ok=False,
                structurally_valid=False,
                parse_error=f"generation error: {e}",
                generated_text_preview="",
            ))
            continue

        # check format validity
        if fmt == "telos":
            prelude = json.loads(row["frames"])
            cut_idx = next(
                (i for i, f in enumerate(prelude)
                 if f.get("type") in {"belief", "plan", "think", "action"}),
                len(prelude)
            )
            parsed_ok, valid, parse_err, val_errs = check_telos_output(
                prelude[:cut_idx], generated_text
            )
        else:
            parsed_ok, valid, parse_err, val_errs = check_chatml_output(generated_text)

        results.append(ExampleResult(
            id=example_id,
            domain=domain,
            parsed_ok=parsed_ok,
            structurally_valid=valid,
            parse_error=parse_err,
            validation_errors=val_errs,
            num_generated_tokens=num_tokens,
            generated_text_preview=generated_text[:200],
        ))

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(ds) - i - 1) / rate
            print(f"  {i + 1}/{len(ds)}  rate={rate:.1f} ex/s  eta={eta:.0f}s")

    # aggregate
    print("\nAggregating results...")
    summary = aggregate(results)

    # write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump({
            "model": model_id,
            "dataset": dataset_id,
            "split": split,
            "format": fmt,
            "num_examples": len(results),
            "summary": summary,
            "results": [asdict(r) for r in results],
        }, f, indent=2)

    print(f"\nResults written to {output_path}\n")
    print_summary(summary)


def aggregate(results: list[ExampleResult]) -> dict[str, Any]:
    """aggregate per-example results into summary statistics."""
    n = len(results)
    if n == 0:
        return {"n": 0}

    n_parsed = sum(1 for r in results if r.parsed_ok)
    n_valid = sum(1 for r in results if r.structurally_valid)

    # per-domain breakdown
    by_domain: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "parsed": 0, "valid": 0})
    for r in results:
        by_domain[r.domain]["n"] += 1
        if r.parsed_ok:
            by_domain[r.domain]["parsed"] += 1
        if r.structurally_valid:
            by_domain[r.domain]["valid"] += 1

    # common failure modes
    error_counts: dict[str, int] = defaultdict(int)
    for r in results:
        if r.parse_error:
            # bucket by error type prefix
            key = r.parse_error.split(":")[0][:50]
            error_counts[f"parse: {key}"] += 1
        for ve in r.validation_errors:
            key = ve.split(":")[0][:50]
            error_counts[f"validation: {key}"] += 1

    return {
        "n": n,
        "parse_rate": n_parsed / n,
        "valid_rate": n_valid / n,
        "by_domain": dict(by_domain),
        "top_failures": dict(
            sorted(error_counts.items(), key=lambda x: -x[1])[:10]
        ),
    }


def print_summary(summary: dict[str, Any]) -> None:
    """print a human-readable summary of the aggregate."""
    if summary.get("n", 0) == 0:
        print("no examples evaluated.")
        return

    n = summary["n"]
    print(f"examples evaluated: {n}")
    print(f"parse rate:         {summary['parse_rate']:.1%}  ({int(summary['parse_rate'] * n)}/{n})")
    print(f"structural valid:   {summary['valid_rate']:.1%}  ({int(summary['valid_rate'] * n)}/{n})")
    print()

    print("by domain:")
    rows = []
    for domain, d in sorted(summary["by_domain"].items()):
        rows.append((
            domain,
            d["n"],
            d["parsed"] / d["n"] if d["n"] else 0.0,
            d["valid"] / d["n"] if d["n"] else 0.0,
        ))
    print(f"  {'domain':<20} {'n':>6} {'parsed':>10} {'valid':>10}")
    for domain, n_d, p_rate, v_rate in rows:
        print(f"  {domain:<20} {n_d:>6} {p_rate:>9.1%} {v_rate:>9.1%}")
    print()

    if summary.get("top_failures"):
        print("top failure modes:")
        for failure, count in summary["top_failures"].items():
            print(f"  {count:>5}  {failure}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--format", required=True, choices=["telos", "chatml"])
    p.add_argument("--model", required=True,
                   help="hf model id or local path. in 'merged' mode, this is the full checkpoint. in 'peft' mode, this is the lora adapter loaded on top of --base-model.")
    p.add_argument("--adapter-mode", default="merged", choices=["merged", "peft"],
                   help="'merged' (default): --model is a self-contained checkpoint with the adapter already fused. 'peft': --model is a lora adapter loaded on top of --base-model.")
    p.add_argument("--base-model", default=None,
                   help="base model id (required when --adapter-mode='peft'). typically meta-llama/Llama-3.1-8B for chatml or kosiasuzu/telos-agent-llama-3.1-8b-init for telos.")
    p.add_argument("--dataset", required=True,
                   help="hf dataset id (e.g. kosiasuzu/telos-agent-trajectory-dataset)")
    p.add_argument("--split", default="eval")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--limit", type=int, default=None,
                   help="if set, evaluate only the first n examples")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    args = p.parse_args()

    if args.adapter_mode == "peft" and not args.base_model:
        p.error("--adapter-mode='peft' requires --base-model")

    evaluate(
        model_id=args.model,
        dataset_id=args.dataset,
        split=args.split,
        fmt=args.format,
        output_path=args.output,
        adapter_mode=args.adapter_mode,
        base_model_id=args.base_model,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()