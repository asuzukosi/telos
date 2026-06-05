from telos.evaluation.benchmarks.bfcl.common import execution_result_frame


def test_execution_result_frame_ok():
    assert execution_result_frame('{"zipcode": "94016"}') == {
        "ok": 1,
        "value": '{"zipcode": "94016"}',
    }


def test_execution_result_frame_error():
    assert execution_result_frame('{"error": "distance not found in database."}') == {
        "ok": 0,
        "value": '{"error": "distance not found in database."}',
    }
