"""tests for agenticml.validators."""
from agenticml.frames import (
    action,
    belief,
    end,
    feedback,
    goal,
    mission,
    obs,
    result,
    reward,
)
from agenticml.validators import Violation, is_valid, validate


def test_empty_trajectory_is_valid():
    assert validate([]) == []
    assert is_valid([])


def test_minimal_valid_trajectory_answer_only():
    frames = [
        goal("be helpful"),
        mission("answer the question"),
        action({"tool": "answer", "text": "42"}),
    ]
    assert validate(frames) == []


def test_minimal_valid_trajectory_terminal_plus_optional_result():
    frames = [
        goal("be helpful"),
        mission("answer the question"),
        action({"tool": "answer", "text": "42"}),
        result({"tool": "answer", "value": None}),
    ]
    assert validate(frames) == []


def test_typical_multistep_trajectory():
    frames = [
        goal("file assistant"),
        mission("find largest file"),
        action({"tool": "list_dir", "path": "/tmp"}),
        result({"tool": "list_dir", "value": ["a", "b"]}),
        belief("two files"),
        action({"tool": "answer", "text": "done"}),
    ]
    assert is_valid(frames)


def test_obs_after_goal_before_first_model_frame_is_allowed():
    frames = [
        goal("be helpful"),
        obs("tools: read_file, answer"),
        mission("answer"),
        action({"tool": "answer", "text": "ok"}),
    ]
    assert is_valid(frames)


def test_mission_can_appear_again_mid_trajectory():
    frames = [
        goal("g"),
        mission("first task"),
        action({"tool": "answer", "text": "a"}),
        mission("revised task"),
        action({"tool": "answer", "text": "b"}),
    ]
    assert is_valid(frames)


def test_belief_before_mission_violates_prelude():
    frames = [
        belief("premature"),
        goal("g"),
        mission("m"),
        action({"tool": "answer", "text": "x"}),
    ]
    vs = validate(frames)
    assert isinstance(vs, list)


def test_trailing_non_terminal_without_result_is_invalid():
    frames = [
        goal("g"),
        mission("m"),
        action({"tool": "list_dir", "path": "/"}),
    ]
    vs = validate(frames)
    assert any(v.rule == "unresolved_action" for v in vs)


def test_trailing_non_terminal_ok_when_runtime_results_expected():
    frames = [
        goal("g"),
        mission("m"),
        action({"tool": "list_dir", "path": "/"}),
    ]
    assert is_valid(frames, allow_unresolved_actions_at_end=True)


def test_belief_after_non_terminal_action_without_result_is_invalid():
    frames = [
        goal("g"),
        mission("m"),
        action({"tool": "list_dir", "path": "/"}),
        belief("skipped result"),
    ]
    vs = validate(frames)
    assert any(v.rule == "unresolved_action" for v in vs)


def test_trailing_fail_without_result_is_valid():
    frames = [
        goal("g"),
        mission("m"),
        action({"tool": "fail", "reason": "nope"}),
    ]
    assert is_valid(frames)


def test_trailing_action_without_result_is_valid_for_terminal():
    frames = [
        goal("g"),
        mission("m"),
        action({"tool": "answer", "text": "x"}),
    ]
    assert is_valid(frames)


def test_orphan_result_is_caught():
    frames = [
        goal("g"),
        mission("m"),
        result({"tool": "bash", "value": "stray"}),
    ]
    vs = validate(frames)
    rules = [v.rule for v in vs]
    assert "orphan_result" in rules


def test_batched_actions_in_one_block_is_valid():
    frames = [
        goal("g"),
        mission("m"),
        action({"tool": "stat", "path": "a"}),
        action({"tool": "stat", "path": "b"}),
        action({"tool": "stat", "path": "c"}),
        result({"tool": "stat", "value": {"size": 1}}),
        result({"tool": "stat", "value": {"size": 2}}),
        result({"tool": "stat", "value": {"size": 3}}),
        belief("collected three sizes"),
        action({"tool": "answer", "text": "done"}),
        result({"tool": "answer", "value": None}),
    ]
    assert is_valid(frames)


def test_batched_actions_with_too_few_results_is_invalid():
    frames = [
        goal("g"),
        mission("m"),
        action({"tool": "stat", "path": "a"}),
        action({"tool": "stat", "path": "b"}),
        action({"tool": "stat", "path": "c"}),
        result({"tool": "stat", "value": {"size": 1}}),
        result({"tool": "stat", "value": {"size": 2}}),
    ]
    vs = validate(frames)
    assert any(v.rule == "unresolved_action" for v in vs)


def test_batched_non_terminal_then_terminal_one_result_ok():
    frames = [
        goal("g"),
        mission("m"),
        action({"tool": "stat", "path": "a"}),
        action({"tool": "answer", "text": "done"}),
        result({"tool": "stat", "value": {"size": 1}}),
    ]
    assert is_valid(frames)


def test_batched_actions_with_extra_result():
    frames = [
        goal("g"),
        mission("m"),
        action({"tool": "stat", "path": "a"}),
        action({"tool": "stat", "path": "b"}),
        result({"tool": "stat", "value": 1}),
        result({"tool": "stat", "value": 2}),
        result({"tool": "stat", "value": 3}),
    ]
    vs = validate(frames)
    rules = [v.rule for v in vs]
    assert "orphan_result" in rules


def test_reward_before_any_action_is_caught():
    frames = [
        goal("g"),
        mission("m"),
        reward(1.0),
        action({"tool": "answer", "text": "x"}),
        result({"tool": "answer", "value": None}),
    ]
    vs = validate(frames)
    rules = [v.rule for v in vs]
    assert "premature_runtime_frame" in rules


def test_feedback_before_any_action_is_caught():
    frames = [
        goal("g"),
        mission("m"),
        feedback("user clarification"),
        action({"tool": "answer", "text": "x"}),
        result({"tool": "answer", "value": None}),
    ]
    vs = validate(frames)
    rules = [v.rule for v in vs]
    assert "premature_runtime_frame" in rules


def test_violation_str_format():
    v = Violation(rule="orphan_action", frame_index=3, message="no result")
    s = str(v)
    assert "orphan_action" in s
    assert "3" in s
    assert "no result" in s


def test_trajectory_without_goal_is_invalid():
    frames = [
        mission("answer"),
        action({"tool": "answer", "text": "x"}),
        result({"tool": "answer", "value": None}),
    ]
    vs = validate(frames)
    rules = [v.rule for v in vs]
    assert "missing_goal" in rules


def test_obs_before_goal_is_invalid():
    frames = [
        obs("env: linux"),
        goal("g"),
        mission("m"),
        action({"tool": "answer", "text": "x"}),
        result({"tool": "answer", "value": None}),
    ]
    vs = validate(frames)
    rules = [v.rule for v in vs]
    assert "missing_goal" in rules


def test_action_end_result_pattern_is_valid():
    frames = [
        goal("g"),
        mission("m"),
        action({"tool": "list_dir", "path": "/"}),
        end(),
        result({"tool": "list_dir", "value": []}),
        action({"tool": "answer", "text": "done"}),
        end(),
    ]
    assert is_valid(frames)


def test_misplaced_end_before_model_frame_is_invalid():
    frames = [
        goal("g"),
        end(),
        mission("m"),
        action({"tool": "answer", "text": "x"}),
    ]
    vs = validate(frames)
    assert any(v.rule == "misplaced_end" for v in vs)


def test_goal_first_then_obs_is_valid():
    frames = [
        goal("g"),
        obs("env: linux"),
        mission("m"),
        action({"tool": "answer", "text": "x"}),
        result({"tool": "answer", "value": None}),
    ]
    assert is_valid(frames)
