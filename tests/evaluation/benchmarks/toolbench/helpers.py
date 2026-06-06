"""shared fixtures for toolbench benchmark tests."""


def finish_row(*, tool_name: str = "search_for_foo", final_answer: str = "done") -> dict:
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
