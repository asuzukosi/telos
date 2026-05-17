import json
import sys

TERMINAL_TOOLS = {"answer", "fail"}

def migrate(frames):
    """remove end frames and the dummy terminal result frame."""
    # remove all end frames
    frames = [f for f in frames if f.get("type") != "end"]
    # if the trajectory ends with a terminal-result-after-terminal-action,
    # drop the final result.
    if (len(frames) >= 2
            and frames[-1].get("type") == "result"
            and frames[-1].get("content") in ({"ok": 1, "value": None}, {"ok": 1.0, "value": None}, {"ok": "1", "value": None}, {"ok": "1.0", "value": None})
            and frames[-2].get("type") == "action"
            and isinstance(frames[-2].get("content"), dict)
            and frames[-2]["content"].get("tool") in TERMINAL_TOOLS):
        frames.pop()
    return frames

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
            obj["frames"] = migrate(obj["frames"])
            f_out.write(json.dumps(obj) + "\n")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])