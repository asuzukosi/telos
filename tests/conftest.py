import json
from pathlib import Path

import pytest


def _write_tool_fixture(data_root: Path) -> dict:
    tool_dir = data_root / "data/toolenv/tools/Demo/demo_tool"
    tool_dir.mkdir(parents=True, exist_ok=True)
    tool_json = {
        "tool_name": "demo_tool",
        "tool_description": "demo tool for tests",
        "api_list": [
            {
                "name": "hello",
                "description": "say hello",
                "required_parameters": [],
                "optional_parameters": [],
            }
        ],
    }
    (data_root / "data/toolenv/tools/Demo/demo_tool.json").write_text(json.dumps(tool_json))
    (tool_dir / "api.py").write_text("def hello():\n    return {'message': 'hi'}\n")
    return {
        "query": "say hi",
        "query_id": "1",
        "api_list": [
            {"category_name": "Demo", "tool_name": "demo_tool", "api_name": "hello"},
        ],
    }


def _write_instruction_fixture(data_root: Path) -> None:
    (data_root / "data/test_instruction").mkdir(parents=True, exist_ok=True)
    (data_root / "data/test_query_ids").mkdir(parents=True, exist_ok=True)
    (data_root / "data/test_instruction/G1_instruction.json").write_text("[]")
    (data_root / "data/test_query_ids/G1_instruction.json").write_text("[]")


@pytest.fixture
def data_root(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "toolbench"
    _write_instruction_fixture(root)
    entry = _write_tool_fixture(root)
    return root, entry
