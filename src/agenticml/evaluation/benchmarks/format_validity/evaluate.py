"""format-validity helpers: generation, parsing, and metrics for FormatValiditySuite."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, cast

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from telos.evaluation.harness.load import model_device
from telos.evaluation.harness.task import TaskResult, TaskTiming, TaskTokens
from telos.frames import parse as telos_parse, render as telos_render
from telos.tokenizer import TelosTokenizer
from telos.trajectory import Trajectory
from telos.validators import validate_for_model_generation

TELOS_MODEL_TYPES = frozenset({"belief", "plan", "think", "action"})
_TOOL_CALL_RE = re.compile(r"<\|python_tag\|>(.+?)<\|(?:eom_id|eot_id)\|>", re.DOTALL)
_ASSISTANT_TEXT_RE = re.compile(
    r"(?:<\|start_header_id\|>assistant<\|end_header_id\|>)?\s*(.*?)<\|(?:eot_id|eom_id)\|>",
    re.DOTALL,
)


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


def _telos_load_tokenizer(model_id: str) -> TelosTokenizer:
    return TelosTokenizer.from_pretrained(model_id)


def _chatml_load_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_id)


def _loads_field(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _row_evaluable(row: dict, fmt: str) -> bool:
    """row has the columns and content needed to run generation for this format."""
    try:
        if fmt == "telos":
            if "frames" not in row:
                return False
            frames = _loads_field(row["frames"])
            if not isinstance(frames, list) or not frames:
                return False
            cut = _telos_cut_index(frames)
            return cut < len(frames)
        if "messages" not in row:
            return False
        messages = _loads_field(row["messages"])
        if not isinstance(messages, list) or not messages:
            return False
        return any(m.get("role") == "assistant" for m in messages)
    except (json.JSONDecodeError, TypeError, KeyError):
        return False


def _telos_cut_index(frames: list[dict]) -> int:
    return next(
        (i for i, f in enumerate(frames) if f.get("type") in TELOS_MODEL_TYPES),
        len(frames),
    )


def _telos_prelude_frames(frames: list[dict], cut: int) -> list:
    """dataset frames use short type names (goal); Trajectory coerces to FrameType."""
    prelude_dicts = [f for f in frames[:cut] if f.get("type") != "end"]
    return Trajectory(prelude_dicts).to_frames()


def _telos_input_ids(row: dict, tt: TelosTokenizer) -> list[int]:
    frames = _loads_field(row["frames"])
    cut = _telos_cut_index(frames)
    if cut >= len(frames):
        return []
    return tt.encode(telos_render(_telos_prelude_frames(frames, cut)))


def _telos_decode_output(tt: TelosTokenizer, token_ids: list[int]) -> str:
    return tt.decode(token_ids)


def _telos_pad_token_id(tt: TelosTokenizer) -> int:
    hf = tt.hf
    pad = hf.pad_token_id
    if isinstance(pad, int):
        return pad
    eos = hf.eos_token_id
    if isinstance(eos, int):
        return eos
    raise ValueError("tokenizer has no pad_token_id or eos_token_id")


def _chatml_input_ids(row: dict, tokenizer: PreTrainedTokenizerBase) -> list[int]:
    messages = _loads_field(row["messages"])
    cut = next(
        (i for i, m in enumerate(messages) if m.get("role") == "assistant"),
        len(messages),
    )
    if cut >= len(messages):
        return []
    encoded = tokenizer.apply_chat_template(
        messages[:cut],
        tokenize=True,
        add_generation_prompt=True,
    )
    return list(cast(list[int], encoded))


def _chatml_decode_output(tokenizer: PreTrainedTokenizerBase, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def _chatml_pad_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    pad = tokenizer.pad_token_id
    if isinstance(pad, int):
        return pad
    eos = tokenizer.eos_token_id
    if isinstance(eos, int):
        return eos
    raise ValueError("tokenizer has no pad_token_id or eos_token_id")


def _telos_stop_ids(tt: TelosTokenizer) -> list[int]:
    ids = [tt.end_id]
    eos = tt.hf.eos_token_id
    if isinstance(eos, int) and eos not in ids:
        ids.append(eos)
    return ids


def _chatml_stop_ids(tokenizer: PreTrainedTokenizerBase) -> list[int]:
    ids: list[int] = []
    unk = tokenizer.unk_token_id
    convert_tokens_to_ids = cast(Callable[[str], int], tokenizer.convert_tokens_to_ids)
    for token in ("<|eot_id|>", "<|eom_id|>"):
        raw_tid = convert_tokens_to_ids(token)
        if isinstance(raw_tid, int) and raw_tid != unk:
            ids.append(raw_tid)
    eos = tokenizer.eos_token_id
    if isinstance(eos, int) and eos not in ids:
        ids.append(eos)
    if ids:
        return ids
    if isinstance(eos, int):
        return [eos]
    raise ValueError("tokenizer has no stop token ids")


def _telos_check(row: dict, generated_text: str) -> tuple[bool, bool, Optional[str], list[str]]:
    frames = _loads_field(row["frames"])
    try:
        generated_frames = telos_parse(generated_text, strict=False)
    except Exception as e:
        return False, False, f"parse failure: {e}", []

    cut = _telos_cut_index(frames)
    full = _telos_prelude_frames(frames, cut)
    full.extend(generated_frames)

    try:
        violations = validate_for_model_generation(full)
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

    has_tool = False
    m = _TOOL_CALL_RE.search(generated_text)
    if m:
        try:
            call = json.loads(m.group(1).strip())
            if not isinstance(call, dict):
                errors.append("tool call payload is not a JSON object")
            elif "name" not in call:
                errors.append("tool call missing 'name' field")
            else:
                has_tool = True
        except json.JSONDecodeError as e:
            return True, False, None, [f"tool call JSON invalid: {e.msg}"]

    text_m = _ASSISTANT_TEXT_RE.search(generated_text)
    has_text = bool(text_m and text_m.group(1).strip())
    if not has_tool and not has_text:
        errors.append("generation has no tool call and no text content")
        return True, False, None, errors

    return True, len(errors) == 0, None, errors


FORMAT_SPECS: dict[str, _FormatSpec] = {
    "telos": _FormatSpec(
        _telos_load_tokenizer,
        _telos_input_ids,
        _telos_decode_output,
        _telos_pad_token_id,
        _telos_check,
        _telos_stop_ids,
    ),
    "chatml": _FormatSpec(
        _chatml_load_tokenizer,
        _chatml_input_ids,
        _chatml_decode_output,
        _chatml_pad_token_id,
        _chatml_check,
        _chatml_stop_ids,
    ),
}


def _generate_completion(
    model,
    spec: _FormatSpec,
    tokenizer,
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
        out = model.generate(
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
    model,
    tokenizer,
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


