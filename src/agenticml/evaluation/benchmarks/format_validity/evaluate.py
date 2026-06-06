"""format-validity helpers: generation, parsing, and metrics for FormatValiditySuite."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import torch
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from agenticml.bridge import bridge
from agenticml.evaluation.harness.load import as_causal_lm, model_device
from agenticml.evaluation.harness.task import TaskResult, TaskTiming, TaskTokens
from agenticml.agentic_template import parse_reserved_wire
from agenticml.trajectory import Trajectory
from agenticml.tokenizer_helpers import (
    agenticml_stop_token_ids,
    chat_template_ids,
    chatml_stop_token_ids,
    pad_token_id,
)
from agenticml.validators import validate

AGENTICML_MODEL_TYPES = frozenset({"belief", "plan", "think", "action"})


@dataclass
class ValidityResult:
    id: str
    domain: str
    parsed_ok: bool
    structurally_valid: bool
    parse_error: Optional[str] = None
    validation_errors: list[str] = field(default_factory=list)
    num_generated_tokens: int = 0
    generated_text_preview: str = ""


@dataclass(frozen=True)
class _FormatSpec:
    load_tokenizer: Callable[[str], Any]
    build_input_ids: Callable[[dict, Any], list[int]]
    decode_output: Callable[[Any, list[int]], str]
    pad_token_id: Callable[[Any], int]
    check_output: Callable[[dict, str], tuple[bool, bool, Optional[str], list[str]]]
    stop_token_ids: Callable[[Any], list[int]]


def _load_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_id)


def _loads_field(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _row_evaluable(row: dict, fmt: str) -> bool:
    """row has the columns and content needed to run generation for this format."""
    try:
        if fmt == "agenticml":
            if "frames" not in row:
                return False
            frames = _loads_field(row["frames"])
            if not isinstance(frames, list) or not frames:
                return False
            cut = _agenticml_cut_index(frames)
            return cut < len(frames)
        if "messages" not in row:
            return False
        messages = _loads_field(row["messages"])
        if not isinstance(messages, list) or not messages:
            return False
        return any(m.get("role") == "assistant" for m in messages)
    except (json.JSONDecodeError, TypeError, KeyError):
        return False


def _agenticml_cut_index(frames: list[dict]) -> int:
    return next(
        (i for i, f in enumerate(frames) if f.get("type") in AGENTICML_MODEL_TYPES),
        len(frames),
    )


def _agenticml_input_ids(row: dict, tokenizer: PreTrainedTokenizerBase) -> list[int]:
    frames = _loads_field(row["frames"])
    cut = _agenticml_cut_index(frames)
    if cut >= len(frames):
        return []
    return chat_template_ids(
        tokenizer,
        Trajectory(frames[:cut]).to_dict(),
        add_generation_prompt=False,
        add_special_tokens=False,
    )


def _chatml_input_ids(row: dict, tokenizer: PreTrainedTokenizerBase) -> list[int]:
    messages = _loads_field(row["messages"])
    cut = next(
        (i for i, m in enumerate(messages) if m.get("role") == "assistant"),
        len(messages),
    )
    if cut >= len(messages):
        return []
    return chat_template_ids(
        tokenizer,
        messages[:cut],
        add_generation_prompt=True,
    )


def _agenticml_check(row: dict, generated_text: str) -> tuple[bool, bool, Optional[str], list[str]]:
    frames = _loads_field(row["frames"])
    try:
        generated_frames = parse_reserved_wire(generated_text, strict=False)
    except Exception as e:
        return False, False, f"parse failure: {e}", []

    cut = _agenticml_cut_index(frames)
    full = Trajectory(frames[:cut]).to_frames()
    full.extend(generated_frames)

    try:
        violations = validate(full, allow_unresolved_actions_at_end=True)
    except Exception as e:
        return True, False, None, [f"validator crashed: {e}"]

    errs = [
        f"[{v.rule}] frame {v.frame_index}: {v.message}" if hasattr(v, "rule") else str(v)
        for v in violations
    ]
    return True, len(errs) == 0, None, errs


def _chatml_check(_row: dict, generated_text: str) -> tuple[bool, bool, Optional[str], list[str]]:
    errors: list[str] = []
    if not re.search(r"<\|(?:eot_id|eom_id)\|>", generated_text):
        return False, False, "missing stop token", ["no stop token emitted"]

    parsed = bridge.parse_chatml_generation(generated_text)
    if parsed.stop_reason.startswith("parse_error"):
        return True, False, None, [parsed.stop_reason]

    has_tool = parsed.tool_call is not None
    has_text = bool(parsed.text.strip())
    if has_tool:
        if not isinstance(parsed.tool_call, dict):
            errors.append("tool call payload is not a JSON object")
        elif "name" not in parsed.tool_call:
            errors.append("tool call missing 'name' field")
    if not has_tool and not has_text:
        errors.append("generation has no tool call and no text content")
        return True, False, None, errors

    return True, len(errors) == 0, None, errors


FORMAT_SPECS: dict[str, _FormatSpec] = {
    "agenticml": _FormatSpec(
        _load_tokenizer,
        _agenticml_input_ids,
        lambda tokenizer, token_ids: tokenizer.decode(token_ids),
        pad_token_id,
        _agenticml_check,
        agenticml_stop_token_ids,
    ),
    "chatml": _FormatSpec(
        _load_tokenizer,
        _chatml_input_ids,
        lambda tokenizer, token_ids: tokenizer.decode(token_ids, skip_special_tokens=False),
        pad_token_id,
        _chatml_check,
        chatml_stop_token_ids,
    ),
}


def _generate_completion(
    model: PreTrainedModel,
    spec: _FormatSpec,
    tokenizer: PreTrainedTokenizerBase,
    input_ids: list[int],
    *,
    max_new_tokens: int = 1024,
    stop_token_ids: Optional[list[int]] = None,
) -> tuple[str, int]:
    device = model_device(model)
    inputs = torch.tensor([input_ids], device=device, dtype=torch.long)
    input_len = inputs.shape[1]
    pad_id = spec.pad_token_id(tokenizer)
    default_eos = stop_token_ids[0] if stop_token_ids else pad_id
    with torch.no_grad():
        out = as_causal_lm(model).generate(
            inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=stop_token_ids or default_eos,
        )
    gen_ids = out[0][input_len:].tolist()
    return spec.decode_output(tokenizer, gen_ids), len(gen_ids)


def eval_row(
    row: dict,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    spec: _FormatSpec,
    stop_ids: list[int],
    max_new_tokens: int,
) -> Optional[tuple[ValidityResult, int, float]]:
    input_ids = spec.build_input_ids(row, tokenizer)
    if not input_ids:
        return None

    prompt_tokens = len(input_ids)
    t0 = time.perf_counter()
    try:
        text, n_gen = _generate_completion(
            model,
            spec,
            tokenizer,
            input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_ids,
        )
    except Exception as e:
        infer = time.perf_counter() - t0
        return (
            ValidityResult(
                id=row["id"],
                domain=row["domain"],
                parsed_ok=False,
                structurally_valid=False,
                parse_error=f"generation error: {e}",
            ),
            prompt_tokens,
            infer,
        )

    parsed_ok, valid, parse_err, val_errs = spec.check_output(row, text)
    infer = time.perf_counter() - t0
    return (
        ValidityResult(
            id=row["id"],
            domain=row["domain"],
            parsed_ok=parsed_ok,
            structurally_valid=valid,
            parse_error=parse_err,
            validation_errors=val_errs,
            num_generated_tokens=n_gen,
            generated_text_preview=text[:200],
        ),
        prompt_tokens,
        infer,
    )


def task_result(res: ValidityResult, *, prompt: int, infer_sec: float) -> TaskResult:
    return TaskResult(
        task_id=res.id,
        domain=res.domain,
        success=res.structurally_valid,
        metrics={"parsed_ok": res.parsed_ok, "structurally_valid": res.structurally_valid},
        timing=TaskTiming(inference_sec=infer_sec, total_sec=infer_sec),
        tokens=TaskTokens(prompt_tokens=prompt, generated_tokens=res.num_generated_tokens),
        detail={
            "parse_error": res.parse_error,
            "validation_errors": res.validation_errors,
            "generated_text_preview": res.generated_text_preview,
        },
    )


def suite_metrics(tasks: list[TaskResult]) -> dict[str, Any]:
    """format-validity rates (merged into benchmark metrics by run_tasks)."""
    n = len(tasks)
    if n == 0:
        return {"n": 0, "parse_rate": 0.0, "valid_rate": 0.0}
    parsed = sum(1 for t in tasks if t.metrics.get("parsed_ok"))
    valid = sum(1 for t in tasks if t.metrics.get("structurally_valid"))
    by_domain: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "parsed": 0, "valid": 0})
    error_counts: dict[str, int] = defaultdict(int)
    for t in tasks:
        by_domain[t.domain]["n"] += 1
        by_domain[t.domain]["parsed"] += int(t.metrics.get("parsed_ok") is True)
        by_domain[t.domain]["valid"] += int(t.metrics.get("structurally_valid") is True)
        if t.detail.get("parse_error"):
            k = f"parse: {str(t.detail['parse_error']).split(':')[0][:50]}"
            error_counts[k] += 1
        for ve in t.detail.get("validation_errors") or []:
            error_counts[f"validation: {ve.split(':')[0][:50]}"] += 1
    return {
        "parse_rate": parsed / n,
        "valid_rate": valid / n,
        "by_domain": dict(by_domain),
        "top_failures": dict(sorted(error_counts.items(), key=lambda x: -x[1])[:10]),
    }


def prepare_eval_dataset(ds, fmt: str, num_examples: int, seed: int):
    """filter to evaluable rows, then sample (num_examples < 0 = full split)."""
    n_raw = len(ds)
    ds_ok = ds.filter(lambda row: _row_evaluable(row, fmt))
    n_ok = len(ds_ok)
    if n_ok == 0:
        raise ValueError(
            f"no evaluable rows for format {fmt!r} (need frames with model blocks, "
            f"or messages with an assistant turn)"
        )
    if n_ok < n_raw:
        print(f"filtered: {n_ok} / {n_raw} rows have a {fmt} generation target")

    if num_examples < 0:
        return ds_ok, n_raw, n_ok, n_ok

    k = min(num_examples, n_ok)
    if k >= n_ok:
        return ds_ok, n_raw, n_ok, n_ok

    indices = random.Random(seed).sample(range(n_ok), k)
    return ds_ok.select(indices), n_raw, n_ok, k


def print_summary(summary: dict[str, Any]) -> None:
    n = summary.get("n", 0)
    if n == 0:
        print("no examples evaluated.")
        return
    print(f"examples evaluated: {n}")
    print(f"parse rate:         {summary['parse_rate']:.1%}  ({int(summary['parse_rate'] * n)}/{n})")
    print(f"structural valid:   {summary['valid_rate']:.1%}  ({int(summary['valid_rate'] * n)}/{n})\n")
    print("by domain:")
    print(f"  {'domain':<20} {'n':>6} {'parsed':>10} {'valid':>10}")
    for domain, d in sorted(summary["by_domain"].items()):
        dn = d["n"]
        print(
            f"  {domain:<20} {dn:>6} "
            f"{d['parsed'] / dn:>9.1%} {d['valid'] / dn:>9.1%}"
        )
    if summary.get("top_failures"):
        print("\ntop failure modes:")
        for failure, count in summary["top_failures"].items():
            print(f"  {count:>5}  {failure}")


