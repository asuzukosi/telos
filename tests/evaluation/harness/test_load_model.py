from unittest.mock import MagicMock, patch

import torch

from agenticml.evaluation.harness.load import causal_lm_load_kwargs, load_model


def test_causal_lm_load_kwargs_cpu_has_device_map():
    kw = causal_lm_load_kwargs(torch.bfloat16)
    assert "device_map" in kw
    assert "torch_dtype" in kw


@patch("agenticml.evaluation.harness.load.AutoModelForCausalLM.from_pretrained")
def test_load_model_calls_from_pretrained(mock_from_pretrained: MagicMock):
    mock_from_pretrained.return_value = MagicMock()
    load_model("org/merged-model", dtype=torch.float16)
    mock_from_pretrained.assert_called_once()
    args, kwargs = mock_from_pretrained.call_args
    assert args[0] == "org/merged-model"
    assert kwargs["torch_dtype"] == torch.float16
