from agenticml.evaluation.benchmarks.toolbench.convert import (
    messages_to_execution_graph,
    row_to_converted,
)
from agenticml.evaluation.benchmarks.toolbench.score import structural_pass

from tests.evaluation.benchmarks.toolbench.helpers import finish_row


def test_row_to_converted_has_finish_step():
    converted = row_to_converted(finish_row())
    assert converted is not None
    ok, _ = structural_pass(converted)
    assert ok is True
    assert converted["answer"]["method"] == "AgenticML_ChatML"


def test_structural_pass_with_valid_trace():
    converted = row_to_converted(finish_row())
    assert converted is not None
    ok, reason = structural_pass(converted)
    assert ok is True
    assert reason == "structural pass"


def test_hallucination_detects_unknown_tool():
    row = finish_row()
    row["messages"][2]["tool_calls"][0]["function"]["name"] = "made_up_tool"
    converted = row_to_converted(row)
    assert converted is not None
    ok, reason = structural_pass(converted)
    assert ok is False
    assert reason == "hallucination"


def test_messages_to_execution_graph_builds_chain():
    row = finish_row()
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
