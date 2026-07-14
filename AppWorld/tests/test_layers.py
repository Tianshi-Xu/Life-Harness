from __future__ import annotations

import json
from pathlib import Path

from life_harness_appworld.layers.h2_action_gate import H2ActionGate
from life_harness_appworld.layers.h3_tool_contract import H3ToolContract
from life_harness_appworld.layers.h4_trajectory_monitor import H4TrajectoryMonitor
from life_harness_appworld.layers.h5_skill_guidance import H5SkillGuidance
from life_harness_appworld.runtime_types import GateDecision, ToolCall


ROOT = Path(__file__).parents[1]
POLICY = ROOT / "policies" / "v005"


def function(name: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "description": "base",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def test_h2_blocks_malformed_json_before_execution() -> None:
    gate = H2ActionGate()
    tools = {"app__read": function("app__read", {}, [])}
    decision = gate.realize("1", "app__read", "{bad", tools)
    assert decision.blocked
    assert "valid JSON" in decision.feedback


def test_h2_blocks_unknown_tool_missing_fields_and_extra_fields() -> None:
    gate = H2ActionGate()
    tools = {
        "app__read": function(
            "app__read",
            {"record_id": {"type": "integer"}},
            ["record_id"],
        )
    }
    assert gate.realize("1", "app__invented", {}, tools).blocked
    assert gate.realize("2", "app__read", {}, tools).blocked
    extra = gate.realize("3", "app__read", {"record_id": 1, "force": True}, tools)
    assert extra.blocked
    assert "undeclared" in extra.feedback


def test_h2_repairs_only_unambiguous_representation() -> None:
    gate = H2ActionGate()
    tools = {
        "app__read": function(
            "app__read",
            {"record_id": {"type": "integer"}, "label": {"type": "string"}},
            ["record_id", "label"],
        )
    }
    decision = gate.realize(
        "1", "app__read", {"record_id": "7", "label": "original"}, tools
    )
    assert not decision.blocked
    assert decision.call.name == "app__read"
    assert decision.call.arguments == {"record_id": 7, "label": "original"}
    assert len(decision.repairs) == 1


def test_h2_never_changes_semantic_values_or_blocks_a_valid_repeat() -> None:
    gate = H2ActionGate()
    tools = {
        "app__write": function(
            "app__write",
            {"record_id": {"type": "integer"}, "value": {"type": "string"}},
            ["record_id", "value"],
        )
    }
    call = ToolCall("1", "app__write", {"record_id": 5, "value": "keep-me"})
    first = gate.apply(call, tools)
    repeated = gate.apply(call, tools)
    assert not first.blocked and not repeated.blocked
    assert first.call == call == repeated.call


def test_h2_blocks_literal_access_token_placeholders() -> None:
    gate = H2ActionGate()
    tools = {
        "app__read": function(
            "app__read", {"access_token": {"type": "string"}}, ["access_token"]
        )
    }
    for placeholder in ("YOUR_APP_ACCESS_TOKEN", "app__get_access_token()"):
        decision = gate.apply(
            ToolCall("1", "app__read", {"access_token": placeholder}), tools
        )
        assert decision.blocked
        assert "placeholder" in decision.feedback
    valid = gate.apply(
        ToolCall("2", "app__read", {"access_token": "eyJreal-token"}), tools
    )
    assert not valid.blocked


def test_h2_drops_only_access_token_when_schema_does_not_declare_it() -> None:
    gate = H2ActionGate()
    tools = {
        "app__public_read": function(
            "app__public_read", {"record_id": {"type": "integer"}}, ["record_id"]
        )
    }
    repaired = gate.apply(
        ToolCall(
            "1",
            "app__public_read",
            {"record_id": 7, "access_token": "irrelevant-token"},
        ),
        tools,
    )
    assert not repaired.blocked
    assert repaired.call.arguments == {"record_id": 7}
    assert repaired.repairs == ["removed undeclared `access_token`"]

    other_extra = gate.apply(
        ToolCall("2", "app__public_read", {"record_id": 7, "force": True}), tools
    )
    assert other_extra.blocked
    assert "`force`" in other_extra.feedback


def test_h2_delays_terminal_submission_parallel_with_unobserved_actions() -> None:
    gate = H2ActionGate()
    action = GateDecision(call=ToolCall("1", "app__write", {"value": 1}))
    terminal = GateDecision(
        call=ToolCall("2", "supervisor__complete_task", {"status": "success"})
    )
    delayed = gate.apply_batch([action, terminal], step=5)
    assert not delayed[0].blocked
    assert delayed[1].blocked
    assert "Observe their results first" in delayed[1].feedback
    assert not gate.apply_batch([terminal], step=5)[0].blocked


def test_h2_blocks_premature_complete_task() -> None:
    gate = H2ActionGate({"min_steps_before_complete": 3, "min_steps_before_fail": 10})
    fail = GateDecision(
        call=ToolCall("1", "supervisor__complete_task", {"status": "fail"})
    )
    success = GateDecision(
        call=ToolCall("2", "supervisor__complete_task", {"status": "success"})
    )
    assert gate.apply_batch([fail], step=5)[0].blocked
    assert "doable" in gate.apply_batch([fail], step=5)[0].feedback
    assert gate.apply_batch([fail], step=10)[0].blocked is False
    assert gate.apply_batch([success], step=2)[0].blocked
    assert gate.apply_batch([success], step=3)[0].blocked is False


def test_h3_clones_selected_tools_and_only_adds_parameter_contracts() -> None:
    contract = H3ToolContract(
        {
            "parameter_contracts": {
                "sort_by": "Prefix with + or -.",
            },
            "tool_parameter_contracts": {
                "file_system__": {
                    "access_token": "Use the token returned by login.",
                }
            },
        }
    )
    original = [
        {
            "type": "function",
            "function": function(
                "app__search",
                {"query": {"type": "string"}, "sort_by": {"type": "string"}},
                ["query"],
            ),
        },
        {
            "type": "function",
            "function": function(
                "app__read", {"record_id": {"type": "integer"}}, ["record_id"]
            ),
        },
        {
            "type": "function",
            "function": function(
                "file_system__show_directory",
                {"access_token": {"type": "string"}},
                ["access_token"],
            ),
        },
    ]
    calibrated, augmented = contract.apply(original)
    assert augmented == 2
    assert [item["function"]["name"] for item in calibrated] == [
        "app__search",
        "app__read",
        "file_system__show_directory",
    ]
    assert original[0]["function"]["description"] == "base"
    assert "Prefix with + or -." in calibrated[0]["function"]["description"]
    assert calibrated[1]["function"]["description"] == "base"
    assert "token returned by login" in calibrated[2]["function"]["description"]


def test_h3_calibrates_action_only_terminal_answer_convention() -> None:
    contract = H3ToolContract(
        {
            "tool_parameter_contracts": {
                "supervisor__complete_task": {
                    "answer": "Provide only for a requested textual response; omit for action-only instructions."
                }
            }
        }
    )
    tools = [
        {
            "type": "function",
            "function": function(
                "supervisor__complete_task",
                {"answer": {"type": "string"}},
                [],
            ),
        }
    ]
    calibrated, augmented = contract.apply(tools)
    assert augmented == 1
    description = calibrated[0]["function"]["description"]
    assert "omit for action-only instructions" in description


def test_h3_pagination_and_password_contracts_are_parameter_local() -> None:
    contract = H3ToolContract(
        {
            "parameter_contracts": {
                "password": "Copy exact punctuation.",
                "page_index": "Continue until an empty or short page.",
            }
        }
    )
    tools = [
        {
            "type": "function",
            "function": function(
                "app__search",
                {
                    "password": {"type": "string", "description": "secret"},
                    "page_index": {"type": "integer", "description": "page"},
                },
                ["password"],
            ),
        }
    ]
    calibrated, augmented = contract.apply(tools)
    assert augmented == 1
    description = calibrated[0]["function"]["description"]
    assert "Copy exact punctuation" in description
    assert "empty or short page" in description


def test_h4_duplicate_detection_is_post_execution_advice_only() -> None:
    monitor = H4TrajectoryMonitor({"duplicate_threshold": 3})
    call = ToolCall("1", "app__read", {"record_id": 1})
    output = '{"id": 1}'
    assert monitor.observe(1, 50, [call], [output]) == []
    assert monitor.observe(2, 50, [call], [output]) == []
    interventions = monitor.observe(3, 50, [call], [output])
    assert len(interventions) == 1
    assert "same action" in interventions[0][1]
    assert output == '{"id": 1}'
    assert monitor.observe(4, 50, [call], [output]) == []


def test_h4_compacts_only_oversized_no_action_private_text() -> None:
    monitor = H4TrajectoryMonitor({"max_no_action_history_characters": 20})
    no_action = {
        "role": "assistant",
        "reasoning_content": "r" * 15,
        "content": "c" * 10,
        "tool_calls": [],
    }
    assert monitor.compact_no_action_message(no_action) == 25
    assert no_action["reasoning_content"] == ""
    assert no_action["content"] == ""
    with_action = {
        "role": "assistant",
        "reasoning_content": "r" * 30,
        "content": "",
        "tool_calls": [{"function": {"name": "app__read"}}],
    }
    assert monitor.compact_no_action_message(with_action) == 0
    assert with_action["reasoning_content"] == "r" * 30


def test_h4_detects_abab_from_action_and_observation_statistics() -> None:
    monitor = H4TrajectoryMonitor()
    a = ToolCall("a", "app__read", {"record_id": 1})
    b = ToolCall("b", "app__read", {"record_id": 2})
    monitor.observe(1, 50, [a], ["A"])
    monitor.observe(2, 50, [b], ["B"])
    monitor.observe(3, 50, [a], ["A"])
    interventions = monitor.observe(4, 50, [b], ["B"])
    assert any("A-B-A-B" in message for _, message in interventions)


def test_h4_error_and_budget_feedback_are_bounded() -> None:
    monitor = H4TrajectoryMonitor(
        {"error_streak_threshold": 2, "budget_fraction": 0.8}
    )
    call = ToolCall("1", "app__write", {"record_id": 1})
    monitor.observe(1, 10, [call], ["Execution failed: bad request"])
    interventions = monitor.observe(8, 10, [call], ["Execution failed: bad request"])
    triggers = {trigger for trigger, _ in interventions}
    assert triggers == {"error_streak", "budget"}
    budget_message = next(message for trigger, message in interventions if trigger == "budget")
    assert "Do not submit merely because the budget is low" in budget_message
    assert "verifying completion" in budget_message
    later = monitor.observe(9, 10, [], [], no_tool=False)
    assert all(trigger != "budget" for trigger, _ in later)


def test_h4_gives_one_post_execution_create_conflict_recovery() -> None:
    monitor = H4TrajectoryMonitor()
    call = ToolCall("1", "app__create_record", {"record_id": 1})
    output = 'Execution failed: Response status code is 409: {"message":"already exists"}'
    first = monitor.observe(1, 50, [call], [output])
    assert any(trigger == "create_conflict" for trigger, _ in first)
    assert any("read/list" in message and "update" in message for _, message in first)
    second = monitor.observe(2, 50, [call], [output])
    assert all(trigger != "create_conflict" for trigger, _ in second)


def test_h4_gives_bounded_authentication_recovery_per_app() -> None:
    monitor = H4TrajectoryMonitor({"max_auth_emissions": 3})
    failure = "Execution failed: Response status code is 401: invalid credentials"
    phone = ToolCall("1", "phone__login", {"username": "x", "password": "y"})
    note = ToolCall("2", "simple_note__login", {"username": "x", "password": "y"})
    first = monitor.observe(1, 50, [phone], [failure])
    assert any(trigger == "authentication_failure" for trigger, _ in first)
    assert any("phone" in message and "returned access token" in message for _, message in first)
    # Second emission should have more specific guidance
    second = monitor.observe(2, 50, [phone], [failure])
    assert any(trigger == "authentication_failure" for trigger, _ in second)
    # Third emission should be the strongest
    third = monitor.observe(3, 50, [phone], [failure])
    assert any(trigger == "authentication_failure" for trigger, _ in third)
    # Fourth should be suppressed
    assert all(
        trigger != "authentication_failure"
        for trigger, _ in monitor.observe(4, 50, [phone], [failure])
    )
    # Different app should still get its own emissions
    other_app = monitor.observe(5, 50, [note], [failure])
    assert any(trigger == "authentication_failure" for trigger, _ in other_app)


def test_h5_retrieves_once_without_tool_or_runtime_control(tmp_path: Path) -> None:
    skills_path = tmp_path / "skills.json"
    skills_path.write_text(
        json.dumps(
            [
                {
                    "id": "complete_scan",
                    "title": "Complete scan",
                    "pattern": "all pages records",
                    "tip": "Read every page before aggregating.",
                },
                {
                    "id": "destructive_last",
                    "title": "Destructive last",
                    "pattern": "delete source",
                    "tip": "Delete only after output creation.",
                },
            ]
        ),
        encoding="utf-8",
    )
    guidance = H5SkillGuidance(skills_path, top_k=1)
    selected = guidance.retrieve("Count all records on every page")
    assert [skill.id for skill in selected] == ["complete_scan"]
    block = guidance.format(selected)
    assert "Read every page" in block
    assert "/no_think" not in block
    assert not hasattr(guidance, "required_tool_names")


def test_h5_skill_rejects_unknown_fields(tmp_path: Path) -> None:
    skills_path = tmp_path / "skills.json"
    skills_path.write_text(
        json.dumps(
            [
                {
                    "id": "play_extremum",
                    "title": "Play an extremum",
                    "pattern": "play least played song",
                    "tip": "Perform the action.",
                }
            ]
        ),
        encoding="utf-8",
    )
    guidance = H5SkillGuidance(skills_path, top_k=1)
    assert [skill.id for skill in guidance.retrieve("Play the least-played song")] == [
        "play_extremum"
    ]


def test_layer_interfaces_have_no_cross_layer_control_channels() -> None:
    assert not hasattr(H4TrajectoryMonitor, "compact_output")
    assert not hasattr(H4TrajectoryMonitor, "compact_message_history")
    assert not hasattr(H4TrajectoryMonitor, "latest_loop_signatures")
    assert not hasattr(H5SkillGuidance, "required_tool_names")

    source = (ROOT / "src/life_harness_appworld/agent.py").read_text(encoding="utf-8")
    forbidden_channels = {
        "rescue_calls",
        "failed_action_errors",
        "loop_blocked_signatures",
        "compact_message_history",
        "compact_output",
        "required_tool_names",
    }
    assert not (forbidden_channels & set(source.split()))


def test_v005_policy_contains_no_task_ids_or_app_specific_skills() -> None:
    rows = json.loads((POLICY / "h5_skills.json").read_text(encoding="utf-8"))
    text = json.dumps(rows)
    assert "source_task_ids" not in text
    assert "spotify__" not in text
    assert "file_system__" not in text


def test_v005_h5_retrieves_cross_app_payment_reconciliation() -> None:
    guidance = H5SkillGuidance(POLICY / "h5_skills.json", top_k=1)
    selected = guidance.retrieve(
        "I paid a dinner bill for coworkers and noted shares; request unpaid people on venmo."
    )
    assert [skill.id for skill in selected] == [
        "resolve_people_and_reconcile_payments"
    ]
    assert "Never invent an email" in guidance.format(selected)


def test_v005_h5_retrieves_enumeration_for_aggregate_queries() -> None:
    guidance = H5SkillGuidance(POLICY / "h5_skills.json", top_k=1)
    selected = guidance.retrieve(
        "What is the total count of all songs in my playlists?"
    )
    assert [skill.id for skill in selected] == [
        "enumerate_before_aggregate_or_bulk_write"
    ]
    block = guidance.format(selected)
    assert "completeness" in block


def test_v005_h5_retrieves_enumeration_for_extremum_action() -> None:
    guidance = H5SkillGuidance(POLICY / "h5_skills.json", top_k=1)
    selected = guidance.retrieve(
        "Play the least listened to song from this album."
    )
    assert [skill.id for skill in selected] == [
        "enumerate_before_aggregate_or_bulk_write"
    ]
    block = guidance.format(selected)
    assert "completeness" in block


def test_v005_h5_retrieves_complete_collection_procedure_without_overtrigger() -> None:
    guidance = H5SkillGuidance(POLICY / "h5_skills.json", top_k=1)
    selected = guidance.retrieve(
        "Organize the vacation photographs into subdirectories based on month."
    )
    assert [skill.id for skill in selected] == [
        "classify_complete_collection_before_submit"
    ]
    block = guidance.format(selected)
    assert "complete source collection" in block
    assert guidance.retrieve("Play a recommended song") == []


def test_v005_h5_enumeration_does_not_overtrigger_on_rating() -> None:
    guidance = H5SkillGuidance(POLICY / "h5_skills.json", top_k=1)
    selected = guidance.retrieve(
        "Give every liked item a five-star rating and increase lower ratings."
    )
    assert selected == []


def test_v005_h5_retrieves_payment_reconciliation_for_transaction_tasks() -> None:
    guidance = H5SkillGuidance(POLICY / "h5_skills.json", top_k=1)
    selected = guidance.retrieve(
        "Like and comment on all payments received from friends in the last seven days."
    )
    assert [skill.id for skill in selected] == [
        "resolve_people_and_reconcile_payments"
    ]


def test_v005_h5_keeps_dinner_reconciliation_distinct_from_social_activity() -> None:
    guidance = H5SkillGuidance(POLICY / "h5_skills.json", top_k=1)
    selected = guidance.retrieve(
        "I paid a dinner bill for coworkers and noted shares; request each unpaid colleague."
    )
    assert [skill.id for skill in selected] == [
        "resolve_people_and_reconcile_payments"
    ]
    assert "Read the source note" in guidance.format(selected)


def test_v005_h5_enumeration_covers_cross_container_filter() -> None:
    guidance = H5SkillGuidance(POLICY / "h5_skills.json", top_k=1)
    selected = guidance.retrieve(
        "Remove all songs released after 2021 from my library and playlists."
    )
    assert [skill.id for skill in selected] == [
        "enumerate_before_aggregate_or_bulk_write"
    ]


def test_v005_h5_destructive_export_retrieves_terminal_action_skill() -> None:
    guidance = H5SkillGuidance(POLICY / "h5_skills.json", top_k=1)
    selected = guidance.retrieve(
        "Export an account backup and terminate the account after the backup."
    )
    assert [skill.id for skill in selected] == [
        "destructive_terminal_actions_last"
    ]
    block = guidance.format(selected)
    assert "irreversible terminal action" in block
