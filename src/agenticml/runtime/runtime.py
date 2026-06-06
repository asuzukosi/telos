from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable
from agenticml.constants import FrameType, TERMINAL_TOOLS
from agenticml.frames import Frame
from agenticml.sdk import step, with_tool_obs
from agenticml.trajectory import Trajectory
from agenticml.runtime.tools import ToolError, ToolRegistryLike

GenerateFn = Callable[[list[int], int, int], list[int]]

@dataclass
class RunResult:
    """
    outcome of a run call. 
    fields:
        - trajectory: the full final trajectory
        - stopped_on: why the loop stopped: "terminal_action", "max_iterations", "no_action"
        - iterations: number of step() calls made
        - final_answer: if stopped_on==terminal_action and the tool was answer, this is where the answer text is stored none otherwise
    """
    trajectory: Trajectory
    stopped_on: str
    iterations: int
    final_answer: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "trajectory": self.trajectory.to_dict(),
            "stopped_on": self.stopped_on,
            "iterations": self.iterations,
            "final_answer": self.final_answer,
        }

def run(
    trajectory, 
    registry: ToolRegistryLike,
    *, 
    tokenizer,
    generate: GenerateFn,
    max_iterations: int = 10,
    max_new_tokens: int = 512,
    strict: bool = True,
) -> RunResult:
    """
    run a trajectory through a tool registry.
    args:
        trajectory: the initial trajectory
        registry: the tool registry
        tokenizer: the tokenizer
        generate: the generate function
        max_iterations: the maximum number of iterations to run
    """
    traj = with_tool_obs(
        trajectory if isinstance(trajectory, Trajectory) else Trajectory(trajectory),
        registry.schemas(),
    )

    for iteration in range(1, max_iterations + 1):
        step_result = step(
            traj,
            tokenizer=tokenizer,
            generate=generate,
            max_new_tokens=max_new_tokens,
            strict=strict,
        )
        if step_result.stopped_on.startswith("parse_error"):
            return RunResult(
                trajectory=step_result.trajectory,
                stopped_on=step_result.stopped_on,
                iterations=iteration,
            )
        traj = step_result.trajectory
        # find and execute the tool
        actions = [f for f in step_result.new_frames if f.type == FrameType.ACTION]
        if not actions:
            # we treat this as terminal # NOTE: we may revise this in the future
            return RunResult(
                trajectory=traj,
                stopped_on="no_action",
                iterations=iteration,
            )
        # execute each action in order, append a result for each
        terminal_hit = False
        terminal_answer: Optional[str] = None

        for action_frame in actions:
            tool_name = (action_frame.content or {}).get("tool")
            if tool_name in TERMINAL_TOOLS:
                terminal_hit = True
                if tool_name == "answer":
                    terminal_answer = action_frame.content.get("text")
                
                traj.append(Frame(
                    type=FrameType.RESULT,
                    content={"tool": tool_name, "value": None},
                ))
                break
            # execute the tool.
            args = {k: v for k, v in (action_frame.content or {}).items() if k != "tool"}
            try:
                value = registry.call(str(tool_name), args)
                traj.append(Frame(
                    type=FrameType.RESULT,
                    content={"tool": tool_name, "value": value},
                ))
            except ToolError as e:
                traj.append(Frame(
                    type=FrameType.RESULT,
                    content={"tool": tool_name, "value": str(e)},
                ))
 
        if terminal_hit:
            return RunResult(
                trajectory=traj,
                stopped_on="terminal_action",
                iterations=iteration,
                final_answer=terminal_answer,
            )
        if step_result.stopped_on == "max_tokens":
            return RunResult(
                trajectory=traj,
                stopped_on="max_tokens",
                iterations=iteration,
            )
    return RunResult(
        trajectory=traj,
        stopped_on="max_iterations",
        iterations=max_iterations,
    )