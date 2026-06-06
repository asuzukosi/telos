import json

from telos.evaluation.benchmarks.bfcl.common import ResultHandler, encode_result
from telos.evaluation.harness.chatml_fc import parse_chatml_fc_call, strip_chat_generation_tokens


def test_strip_chat_generation_tokens():
    raw = '{"name": "fn", "parameters": {}}<|eot_id|>'
    assert strip_chat_generation_tokens(raw) == '{"name": "fn", "parameters": {}}'


def test_parse_chatml_fc_call_string_parameters():
    raw = '{"name": "calc_heat_capacity", "parameters": "{\\"temp\\": 298}"}<|eot_id|>'
    call = parse_chatml_fc_call(raw)
    assert call is not None
    assert call["name"] == "calc_heat_capacity"
    out = encode_result("simple_python_0", call=call)
    parsed = json.loads(out)
    assert parsed["parameters"] == {"temp": 298}


def test_decode_ast_strips_eot_and_string_parameters():
    raw = (
        '{"name": "instrument_price.get", "parameters": '
        '"{\\"brand\\": \\"Fender\\", \\"model\\": \\"Strat\\", \\"finish\\": \\"Rosewood\\"}"}'
        "<|eot_id|>"
    )
    handler = ResultHandler.from_model_id("org/model")
    decoded = handler.decode_ast(raw)
    assert decoded == [
        {"instrument_price.get": {"brand": "Fender", "model": "Strat", "finish": "Rosewood"}}
    ]
