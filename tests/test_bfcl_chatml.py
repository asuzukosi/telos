import json

import pytest

from telos.evaluation.benchmarks.bfcl.chatml import entry_turn_messages
from telos.evaluation.benchmarks.bfcl.common import encode_result
from telos.evaluation.benchmarks.bfcl.common import entry_tool_schemas
from telos.evaluation.harness.backends.chatml_backend import _with_tools


def test_entry_turn_messages_inserts_system():
    pytest.importorskip("tree_sitter")
    entry = {
        "id": "simple_python_0",
        "question": [[{"role": "user", "content": "compute area"}]],
        "function": [
            {
                "name": "area",
                "description": "area",
                "parameters": {
                    "type": "dict",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
            }
        ],
    }
    msgs = entry_turn_messages(entry, 0)
    assert msgs[0]["role"] == "system"
    assert "area" in msgs[0]["content"] or "function" in msgs[0]["content"].lower()
    assert any(m["role"] == "user" for m in msgs)


def test_entry_turn_messages_later_turn_unchanged():
    entry = {
        "id": "multi_turn_base_0",
        "question": [
            [{"role": "user", "content": "first"}],
            [{"role": "user", "content": "second"}],
        ],
        "function": [],
    }
    turn1 = entry_turn_messages(entry, 1)
    assert turn1 == [{"role": "user", "content": "second"}]


def test_encode_result_from_call():
    raw = '<|python_tag|>{"name": "fn", "parameters": {"a": 1}}<|eot_id|>'
    call = {"name": "fn", "parameters": {"a": 1}}
    out = encode_result("simple_python_1", raw=raw, call=call)
    parsed = json.loads(out)
    assert parsed["name"] == "fn"
    assert parsed["parameters"] == {"a": 1}


def test_encode_result_irrelevance_empty():
    assert encode_result("irrelevance_0", raw="just text") == "[]"


def test_chatml_skips_duplicate_tool_block_in_prompt():
    pytest.importorskip("tree_sitter")
    from telos.evaluation.benchmarks.bfcl.subset import load_subset

    entry = next(e for e in load_subset().entries if e["id"] == "multi_turn_base_67")
    msgs = entry_turn_messages(entry, 0)
    dup = _with_tools(list(msgs), entry_tool_schemas(entry))
    assert "available tools:" not in msgs[0]["content"]
    assert any("available tools:" in m.get("content", "") for m in dup)


def test_encode_result_passes_json_payload():
    raw = '{"name": "fn", "parameters": {"b": 2}}'
    out = encode_result("simple_python_2", raw=raw)
    assert json.loads(out)["name"] == "fn"
