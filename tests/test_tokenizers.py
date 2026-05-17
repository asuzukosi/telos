"""
test for telos tokenizer

these tests require access to the gated llama-3.1 8b base model on hugginface. if the model cannot be accessed, the tests will be skipped.
"""

import pytest

from telos.constants import (
    DEFAULT_BASE_MODEL,
    END_MARKER,
    END_MARKER_RESERVED_SLOT,
    FrameOwner,
    FrameType,
    TELOS_OWNERS,
    TELOS_TOKEN_MAP,
)


@pytest.fixture(scope="module")
def tokenizer():
    """
    load a tokenizer for the llama-3.1 8b base model. skip the test if the model cannot be accessed.
    """
    try:
        from telos.tokenizer import TelosTokenizer
    except ImportError as e:
        pytest.skip(f"telos.tokenizer not importable: {e}")
    try:
        return TelosTokenizer.from_pretrained(DEFAULT_BASE_MODEL)
    except Exception as e:
        pytest.skip(f"failed to load tokenizer for {DEFAULT_BASE_MODEL}: {e}")


def test_token_map_has_ten_frame_entries():
    assert len(TELOS_TOKEN_MAP) == 10


def test_token_map_entries_are_unique():
    slots = [slot for _, slot in TELOS_TOKEN_MAP]
    assert len(slots) == len(set(slots))
    slots = sorted(slot for _, slot in TELOS_TOKEN_MAP)
    assert slots == [0, 1, 2, 3, 4, 5, 6, 8, 9, 10]
    assert END_MARKER_RESERVED_SLOT == 7
    assert END_MARKER_RESERVED_SLOT not in slots

def test_owner_table_covers_all_frame_types():
    """
    test that the owner table covers all frame types.
    """
    for frame_type in FrameType:
        assert frame_type in TELOS_OWNERS
        assert TELOS_OWNERS[frame_type] in [FrameOwner.RUNTIME, FrameOwner.MODEL]

def test_default_base_model_is_llama31_8b():
    assert DEFAULT_BASE_MODEL == "meta-llama/Llama-3.1-8B"


def test_apply_trajectory_template_string_matches_render(tokenizer):
    from telos.frames import action, goal, mission, render
    from telos.trajectory import Trajectory

    frames = [
        goal("hello"),
        mission("task"),
        action({"tool": "answer", "text": "ok"}),
    ]
    tr = Trajectory(frames)
    assert tokenizer.apply_trajectory_template(tr, tokenize=False) == render(frames)


def test_apply_trajectory_template_accepts_frame_dicts(tokenizer):
    from telos.trajectory import Trajectory

    tr = Trajectory([
        {"type": "goal", "content": "x"},
        {"type": "action", "content": {"tool": "answer", "text": "y"}},
    ])
    wire = tokenizer.apply_trajectory_template(tr, tokenize=False)
    assert "<|goal|>x" in wire
    assert END_MARKER in wire


def test_apply_trajectory_template_tokenize_matches_encode(tokenizer):
    from telos.frames import action, goal, mission, render
    from telos.trajectory import Trajectory

    tr = Trajectory([
        goal("a"),
        mission("b"),
        action({"tool": "answer", "text": "z"}),
    ])
    wire = render(tr.to_frames())
    assert tokenizer.apply_trajectory_template(tr, tokenize=True) == tokenizer.encode(wire)


def test_apply_trajectory_template_return_tensors_pt(tokenizer):
    torch = pytest.importorskip("torch")
    from telos.frames import action, goal
    from telos.trajectory import Trajectory

    tr = Trajectory([
        goal("hi"),
        action({"tool": "answer", "text": "ok"}),
    ])
    wire = tokenizer.apply_trajectory_template(tr, tokenize=False)
    t = tokenizer.apply_trajectory_template(
        tr,
        tokenize=True,
        return_tensors="pt",
    )
    assert isinstance(t, torch.Tensor)
    assert t.shape == (1, len(tokenizer.encode(wire)))


def test_each_marker_encodes_to_single_token(tokenizer):
    for telos_name, _slot in TELOS_TOKEN_MAP:
        ids = tokenizer.encode(telos_name.value)
        assert len(ids) == 1, f"expected single token for {telos_name.value}, got {ids}"
    ids = tokenizer.encode(END_MARKER)
    assert len(ids) == 1, f"expected single token for {END_MARKER}, got {ids}"


def test_marker_ids_match_id_of_lookup(tokenizer):
    for telos_name, _slot in TELOS_TOKEN_MAP:
        ids = tokenizer.encode(telos_name.value)
        assert ids[0] == tokenizer.id_of(telos_name.value), (
            f"expected token id {tokenizer.id_of(telos_name.value)} for {telos_name.value}, got {ids[0]}"
        )
    ids = tokenizer.encode(END_MARKER)
    assert ids[0] == tokenizer.id_of(END_MARKER)


def test_end_id_property_matches_explicit_lookup(tokenizer):
    assert tokenizer.end_id == tokenizer.id_of(END_MARKER)


def test_telos_token_ids_returns_all_eleven(tokenizer):
    ids = tokenizer.telos_token_ids()
    assert len(ids) == 11
    for telos_name, _slot in TELOS_TOKEN_MAP:
        assert tokenizer.id_of(telos_name) in ids
    assert tokenizer.id_of(END_MARKER) in ids
 
 
def test_unknown_marker_raises(tokenizer):
    with pytest.raises(KeyError):
        tokenizer.token("<|not_a_telos_marker|>")
 
 
def test_encode_decode_round_trip_simple(tokenizer):
    text = "<|goal|>You are helpful.<|mission|>Answer 2+2.<|end|>"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert decoded == text
 
 
def test_encode_decode_round_trip_with_json(tokenizer):
    text = (
        '<|action|>{"tool":"read_file","path":"main.py"}<|end|>'
        '<|result|>{"ok":1,"value":"hello"}'
    )
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert decoded == text
 
 
def test_encode_does_not_add_bos(tokenizer):
    text = "<|goal|>hi"
    ids = tokenizer.encode(text)
    # The first token should be the <|goal|> ID, not BOS.
    assert ids[0] == tokenizer.id_of("<|goal|>")
 
 
def test_underlying_tokenizer_is_unmodified(tokenizer):
    """Aliasing happens at the string level; the underlying HF tokenizer
    should still know the reserved tokens by their original names."""
    raw_ids = tokenizer.hf.encode(
        "<|reserved_special_token_0|>", add_special_tokens=False
    )
    assert len(raw_ids) == 1
    assert raw_ids[0] == tokenizer.id_of("<|goal|>")
 
 
def test_describe_lists_all_markers(tokenizer):
    text = tokenizer.describe()
    for telos_name, _slot in TELOS_TOKEN_MAP:
        assert telos_name.value in text
    assert END_MARKER in text
 
 
def test_token_metadata_has_correct_owner(tokenizer):
    assert tokenizer.token("<|goal|>").owner == "runtime"
    assert tokenizer.token("<|action|>").owner == "model"
    assert tokenizer.token("<|end|>").owner == "model"
    assert tokenizer.token("<|result|>").owner == "runtime"
 