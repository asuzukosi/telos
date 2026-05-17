# scripts/telos_to_chatml.py
import json
from telos.constants import TERMINAL_TOOLS
import sys

def telos_to_chatml(frames: list[dict]) -> list[dict]:
    """convert a telos trajectory to chatml messages."""
    messages = []
    system_parts = []
    pending_reasoning = []  # accumulate belief, plan, and think until next action
    
    for f in frames:
        t = f["type"]
        c = f["content"]
        
        if t == "goal":
            system_parts.append(c)
        elif t == "obs":
            # tool defs typically live in obs - inline them in system for SFT
            # (production chatml separates them, but for SFT they go in system)
            system_parts.append(c)
        elif t == "mission":
            # flush system, then user message
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
                # terminal action - "answer" becomes the final assistant message
                # "fail" becomes a final message explaining the failure
                text = args.get("text") or args.get("reason") or ""
                final_content = (assistant_content + "\n\n" + text).strip() if assistant_content else text
                messages.append({"role": "assistant", "content": final_content})
            else:
                # Non-terminal tool call.
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
        # reward: dropped (no ChatML equivalent)
    
    return messages


def main(in_path, out_path):
    with open(in_path) as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "frames" not in obj or "id" not in obj or "domain" not in obj:
                print(f"skipping line: {line} (bad format)")
                continue
            try:
                messages = telos_to_chatml(obj["frames"])
                obj["messages"] = messages
                f_out.write(json.dumps(obj) + "\n")
            except Exception as e:
                print(f"skipping line: {line} (error: {e})")
                continue

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])