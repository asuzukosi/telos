"""convert telos trajectory frames to chatml messages in jsonl records."""

from __future__ import annotations

import json
from pathlib import Path

from telos.constants import TERMINAL_TOOLS


def telos_to_chatml(frames: list[dict]) -> list[dict]:
    messages: list[dict] = []
    system_parts: list[str] = []
    pending_reasoning: list[str] = []

    for f in frames:
        t = f["type"]
        c = f["content"]

        if t == "goal":
            system_parts.append(c)
        elif t == "obs":
            system_parts.append(c)
        elif t == "mission":
            if system_parts:
                messages.append({"role": "system", "content": "\n\n".join(system_parts)})
                system_parts = []
            messages.append({"role": "user", "content": c})
        elif t in ("belief", "plan", "think"):
            pending_reasoning.append(c)
        elif t == "action":
            tool = c.get("tool")
            args = {k: v for k, v in c.items() if k != "tool"}
            assistant_content = "\n".join(pending_reasoning) if pending_reasoning else ""
            pending_reasoning = []

            if tool in TERMINAL_TOOLS:
                text = args.get("text") or args.get("reason") or ""
                final_content = (
                    (assistant_content + "\n\n" + text).strip() if assistant_content else text
                )
                messages.append({"role": "assistant", "content": final_content})
            else:
                messages.append({
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": [{
                        "id": f"call_{len(messages)}",
                        "type": "function",
                        "function": {
                            "name": tool,
                            "arguments": json.dumps(args),
                        },
                    }],
                })
        elif t == "result":
            messages.append({
                "role": "tool",
                "tool_call_id": f"call_{len(messages) - 1}",
                "content": json.dumps(c),
            })
        elif t == "feedback":
            messages.append({"role": "user", "content": c})

    return messages


def convert_jsonl(input_path: str | Path, output_path: str | Path) -> None:
    in_path = Path(input_path)
    out_path = Path(output_path)
    with in_path.open() as f_in, out_path.open("w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "frames" not in obj or "id" not in obj or "domain" not in obj:
                print(f"skipping line: {line} (bad format)")
                continue
            try:
                obj["messages"] = telos_to_chatml(obj["frames"])
                f_out.write(json.dumps(obj) + "\n")
            except Exception as e:
                print(f"skipping line: {line} (error: {e})")
