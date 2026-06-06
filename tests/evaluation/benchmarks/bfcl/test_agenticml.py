import json

from telos.evaluation.benchmarks.bfcl.common import (
    ResultHandler,
    actions_to_result,
    entry_tool_schemas,
    functions_to_schemas,
)
from telos.evaluation.benchmarks.bfcl.telos import entry_to_prelude
from telos.evaluation.benchmarks.common import model_dir_name


def test_model_dir_name():
    assert model_dir_name("org/model") == "org_model"


def test_functions_to_schemas():
    bfcl_fn = [
        {
            "name": "calculate_triangle_area",
            "description": "area of a triangle",
            "parameters": {
                "type": "dict",
                "properties": {
                    "base": {"type": "integer", "description": "base"},
                    "height": {"type": "integer", "description": "height"},
                },
                "required": ["base", "height"],
            },
        }
    ]
    schemas = functions_to_schemas(bfcl_fn)
    assert schemas[0]["name"] == "calculate_triangle_area"
    assert schemas[0]["parameters"]["type"] == "object"
    assert "base" in schemas[0]["parameters"]["properties"]

    entry = {"function": bfcl_fn}
    assert entry_tool_schemas(entry) == schemas


def test_entry_to_prelude():
    entry = {"question": [[{"role": "user", "content": "do the thing"}]]}
    frames = entry_to_prelude(entry, user_content="do the thing")
    assert frames[0]["type"] == "goal"
    assert frames[1] == {"type": "mission", "content": "do the thing"}


def test_actions_to_result_single_call():
    actions = [{"tool": "search", "q": "a", "limit": 3}]
    raw = actions_to_result(actions, "simple_python_1")
    parsed = json.loads(raw)
    assert parsed["name"] == "search"
    assert parsed["parameters"] == {"q": "a", "limit": 3}


def test_actions_to_result_parallel_array():
    actions = [
        {"tool": "a", "x": 1},
        {"tool": "b", "y": 2},
    ]
    raw = actions_to_result(actions, "parallel_1")
    parsed = json.loads(raw)
    assert len(parsed) == 2


def test_actions_to_result_irrelevance_empty():
    raw = actions_to_result([], "irrelevance_3")
    assert raw == "[]"


def test_result_handler_decode_ast():
    import pytest

    pytest.importorskip("tree_sitter")
    from telos.evaluation.benchmarks.bfcl.subset import ensure_bfcl_on_path

    ensure_bfcl_on_path()
    pytest.importorskip("bfcl_eval")

    h = ResultHandler.from_model_id("x/y")
    raw = json.dumps({"name": "fn", "parameters": {"a": 1}})
    assert h.decode_ast(raw) == [{"fn": {"a": 1}}]
    parallel = json.dumps(
        [
            {"name": "a", "parameters": {"x": 1}},
            {"name": "b", "parameters": {"y": 2}},
        ]
    )
    assert len(h.decode_ast(parallel)) == 2
