from telos.evaluation.benchmarks.toolbench.convert import (
    messages_to_execution_graph,
    row_to_converted,
)
from telos.evaluation.benchmarks.toolbench.score import check_has_hallucination, get_steps, structural_pass


def _finish_row(*, tool_name: str = "search_for_foo", final_answer: str = "done") -> dict:
    functions = [
        {"name": tool_name, "description": "search", "parameters": {"type": "object", "properties": {}}},
        {
            "name": "Finish",
            "description": "finish",
            "parameters": {
                "type": "object",
                "properties": {
                    "return_type": {"type": "string"},
                    "final_answer": {"type": "string"},
                },
            },
        },
    ]
    messages = [
        {"role": "system", "content": "task"},
        {"role": "user", "content": "find foo"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": '{"ok": 1}'},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "Finish",
                        "arguments": (
                            '{"return_type": "give_answer", "final_answer": '
                            f'"{final_answer}"}}'
                        ),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
    ]
    return {
        "id": "1",
        "group": "G1_instruction",
        "format": "chatml",
        "success": True,
        "query": "find foo",
        "available_tools": functions,
        "messages": messages,
        "final_answer": final_answer,
        "steps": 2,
    }


def test_row_to_converted_has_finish_step():
    converted = row_to_converted(_finish_row())
    assert converted is not None
    _steps, final_step = get_steps(converted)
    assert "'name': 'Finish'" in final_step
    assert converted["answer"]["method"] == "Telos_ChatML"


def test_structural_pass_with_valid_trace():
    converted = row_to_converted(_finish_row())
    assert converted is not None
    ok, reason = structural_pass(converted)
    assert ok is True
    assert reason == "structural pass"


def test_hallucination_detects_unknown_tool():
    row = _finish_row()
    row["messages"][2]["tool_calls"][0]["function"]["name"] = "made_up_tool"
    converted = row_to_converted(row)
    assert converted is not None
    ok, reason = structural_pass(converted)
    assert ok is False
    assert reason == "hallucination"
    assert check_has_hallucination(converted["available_tools"], converted["answer"]) is False


def test_messages_to_execution_graph_builds_chain():
    row = _finish_row()
    eg = messages_to_execution_graph(row["query"], row["available_tools"], row["messages"])
    details = eg.convert_to_dict()
    assert details[0]["role"] == "system"
    assert any(
        node.get("role") == "tool" and node.get("message", {}).get("name") == "Finish"
        for node in _walk(details)
    )


def _walk(nodes):
    if isinstance(nodes, dict):
        yield nodes
        for child in nodes.get("next") or []:
            yield from _walk(child)
    elif isinstance(nodes, list):
        for node in nodes:
            yield from _walk(node)
