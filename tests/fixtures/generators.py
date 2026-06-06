"""shared test generators for sdk, runtime, and harness backends."""

from __future__ import annotations


class SdkScriptedGenerator:
    """returns a single preset wire response for sdk step() tests."""

    def __init__(self, response_text: str, *, append_end: bool = True):
        self.response_text = response_text
        self.append_end = append_end

    def __call__(self, input_ids, stop_token_id, max_new_tokens):
        del input_ids
        ids = [ord(c) for c in self.response_text]
        if self.append_end:
            ids.append(stop_token_id)
        if len(ids) > max_new_tokens:
            ids = ids[:max_new_tokens]
        return ids


class HfScriptedGenerator:
    """pops preset responses for multi-step runtime / hf backend tests."""

    def __init__(self, responses: list[str], *, append_stop: bool = True):
        self.responses = list(responses)
        self.append_stop = append_stop
        self.call_count = 0

    def __call__(
        self,
        input_ids,
        eos_token_id,
        max_new_tokens,
        *,
        pad_token_id=None,
        return_full_sequence=False,
        **_kwargs,
    ):
        del pad_token_id
        if not self.responses:
            raise AssertionError("hf scripted generator exhausted")
        text = self.responses.pop(0)
        self.call_count += 1
        ids = [ord(c) for c in text]
        if self.append_stop:
            stop = eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id
            ids.append(stop)
        if len(ids) > max_new_tokens:
            ids = ids[:max_new_tokens]
        if return_full_sequence:
            return list(input_ids) + ids
        return ids
