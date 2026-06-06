"""build telos prelude frames for a swe-bench instance."""

from __future__ import annotations

from typing import Any

from telos.evaluation.benchmarks.swe.common import DEFAULT_GOAL

# aligned with third_party/mini-swe-agent/src/minisweagent/config/benchmarks/swebench.yaml
SWE_MISSION_TEMPLATE = """<pr_description>
Consider the following PR description:
{task}
</pr_description>

<instructions>
You are a software engineer interacting with a computer shell to solve programming tasks.
Make changes to non-test files in /testbed to fix the issue in the PR description.

For each step:
1. Reason about what to do next
2. Issue at least one bash command via the bash tool

Rules:
- Working directory for commands is /testbed
- The repo at /testbed is a local checkout only (no git remotes); do not use git fetch, git pull, or git clone
- Do not modify tests or packaging/config files unless required by the fix
- Each command runs in a fresh subshell (use `cd /testbed && ...` when needed)
- Use non-interactive flags; avoid vi/nano and other interactive tools

Submission (when done):
1. Create patch.txt with only your source-file changes: `git diff -- path/to/file > patch.txt`
2. Verify patch.txt headers show `--- a/` and `+++ b/` paths
3. Submit with this exact command (separate from patch creation):
   `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt`
</instructions>"""


def instance_mission(instance: dict[str, Any]) -> str:
    task = str(instance.get("problem_statement") or "").strip()
    if not task:
        raise ValueError("instance missing problem_statement")
    return SWE_MISSION_TEMPLATE.format(task=task)


def instance_to_prelude(instance: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"type": "goal", "content": DEFAULT_GOAL},
        {"type": "mission", "content": instance_mission(instance)},
    ]


def instance_to_messages(instance: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DEFAULT_GOAL},
        {"role": "user", "content": instance_mission(instance)},
    ]
