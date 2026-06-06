"""convert agenticml toolbench rows to upstream tooleval answer format."""

from __future__ import annotations

from typing import Any

from agenticml.bridge import bridge
from agenticml.evaluation.benchmarks.toolbench.common import ensure_toolbench_on_path
from agenticml.evaluation.harness.backends.agenticml_backend import BackendRunResult


def _execution_types():
    ensure_toolbench_on_path()
    from toolbench.tooleval.evaluation.dataclass import ExecutionGraph, ExecutionNode

    return ExecutionGraph, ExecutionNode


def _init_graph(eg: Any, ExecutionNode: Any, functions: list[dict], query: str) -> Any:
    init_node = ExecutionNode(
        role="system",
        message=(
            "You are AutoGPT, you can use many tools(functions) to do the following task.\n"
            "First I will give you the task description, and your task start.\n"
            "At each step, you need to give your thought to analyze the status now and what to do next, "
            "with a function call to actually excute your step.\n"
            "After the call, you will get the call result, and you are now in a new state.\n"
            "Then you will analyze your status now, then decide what to do next...\n"
            "After many (Thought-call) pairs, you finally perform the task, then you can give your finial answer.\n"
            "Remember: \n"
            '1.the state change is irreversible, you can\'t go back to one of the former state, if you want to restart the task, say "I give up and restart".\n'
            "2.All the thought is short, at most in 5 sentence.\n"
            "3.You can do more then one trys, so if your plan is to continusly try some conditions, you can do one of the conditions per try.\n"
            "Let's Begin!\n"
            "Task description: You should use functions to help handle the real time user querys. "
            'Remember to ALWAYS call "Finish" function at the end of the task. And the final answer should '
            "contain enough information to show to the user.\n"
            "Specifically, you have access to the following functions: " + str(functions),
        ),
    )
    eg.set_init_node(init_node)
    user_node = ExecutionNode(role="user", message=query)
    eg.add_node(user_node)
    eg[init_node, user_node] = None
    return user_node


def messages_to_execution_graph(
    query: str,
    functions: list[dict[str, Any]],
    messages: list[dict[str, Any]],
) -> Any:
    ExecutionGraph, ExecutionNode = _execution_types()
    eg = ExecutionGraph()
    last_node = _init_graph(eg, ExecutionNode, functions, query)

    index = 0
    while index < len(messages) and messages[index].get("role") in ("system", "user"):
        index += 1

    while index < len(messages):
        message = messages[index]
        role = message.get("role")
        if role in ("system", "user", "function"):
            index += 1
            continue
        if role == "assistant":
            if message.get("tool_calls"):
                tc = message["tool_calls"][0]["function"]
                name = tc["name"]
                arguments = tc["arguments"]
                response = ""
                if index + 1 < len(messages) and messages[index + 1].get("role") == "tool":
                    if name != "Finish":
                        response = messages[index + 1].get("content") or ""
                    index += 1
                node = ExecutionNode(
                    role="tool",
                    message={"name": name, "arguments": arguments, "response": response},
                )
            else:
                node = ExecutionNode(role="assistant", message=message.get("content") or "")
        elif role == "tool":
            index += 1
            continue
        else:
            raise NotImplementedError(f"unknown role {role!r}")

        index += 1
        eg.add_node(node)
        eg[last_node, node] = None
        last_node = node

    return eg.reduce_graph_to_sequence()


def trace_messages(run: BackendRunResult) -> list[dict[str, Any]]:
    if run.messages:
        return list(run.messages)
    if run.run is not None:
        return bridge.trajectory_to_messages(run.run.trajectory)
    return []


def method_name(fmt: str) -> str:
    return "AgenticML_ChatML" if fmt == "chatml" else "AgenticML_CoT"


def row_to_converted(row: dict[str, Any]) -> dict[str, Any] | None:
    query = row.get("query") or ""
    functions = row.get("available_tools")
    messages = row.get("messages")
    if not query or not functions or not messages:
        return None

    fmt = str(row.get("format") or "agenticml")
    eg = messages_to_execution_graph(query, functions, messages)
    final_answer = row.get("final_answer")
    if final_answer is None:
        final_answer = ""

    return {
        "query": query,
        "available_tools": functions,
        "answer": {
            "method": method_name(fmt),
            "total_steps": eg.node_count,
            "final_answer": final_answer,
            "answer_details": eg.convert_to_dict(),
        },
    }


def rows_to_converted_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        converted = row_to_converted(row)
        if converted is not None:
            out[str(row["id"])] = converted
    return out
