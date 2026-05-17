import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PROMPT_PATH = _REPO_ROOT / "prompts" / "synthetic_data_generation.txt"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

prompt = _PROMPT_PATH.read_text()

# override the "generate now" instruction to limit to 20 for pilot
pilot_instruction = "\n\nFor this run, generate exactly 20 trajectories."

print("making request now")
response = client.chat.completions.create(
    model="qwen/qwen3.5-plus-20260420",
    max_tokens=16000,
    messages=[{"role": "user", "content": prompt + pilot_instruction}],
)

out_path = _REPO_ROOT / "data" / "pilot2.jsonl"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    f.write(response.choices[0].message.content or "")
print("file written")