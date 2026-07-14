from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

from openai import OpenAI

from appworld import AppWorld
from appworld_agents.code.common.usage_tracker import Usage
from appworld_agents.code.simplified.agent import ExecutionIO, Status
from appworld_agents.code.simplified.function_calling_agent import SimplifiedFunctionCallingAgent

from life_harness_appworld.events import HarnessEventLogger
from life_harness_appworld.layers.h2_action_gate import H2ActionGate
from life_harness_appworld.layers.h3_tool_contract import H3ToolContract
from life_harness_appworld.layers.h4_trajectory_monitor import H4TrajectoryMonitor
from life_harness_appworld.layers.h5_skill_guidance import H5SkillGuidance
from life_harness_appworld.runtime_types import GateDecision, ToolCall


def load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


class LifeHarnessFunctionCallingAgent(SimplifiedFunctionCallingAgent):
    """AppWorld agent with four independent, lifecycle-scoped harness hooks."""

    def __init__(
        self,
        harness_policy_directory: str,
        harness_h2: bool = True,
        harness_h3: bool = True,
        harness_h4: bool = True,
        harness_h5: bool = True,
        harness_h5_top_k: int = 1,
        baseline: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.baseline = baseline
        policy_directory = Path(harness_policy_directory)
        self.harness_h2 = harness_h2
        self.harness_h3 = harness_h3
        self.harness_h4 = harness_h4
        self.harness_h5 = harness_h5
        self.h2 = H2ActionGate(load_json(policy_directory / "h2.json"))
        self.h3 = H3ToolContract(load_json(policy_directory / "h3.json"))
        self.h4 = H4TrajectoryMonitor(load_json(policy_directory / "h4.json"))
        self.h5 = H5SkillGuidance(
            policy_directory / "h5_skills.json", top_k=harness_h5_top_k
        )
        self.h5_block = ""
        self.harness_events = HarnessEventLogger()

        # The main model has no client-side output cap.  vLLM may use all
        # remaining context up to its configured max-model-len.
        self.language_model.generation_kwargs.pop("max_completion_tokens", None)
        self._install_vllm_template_transport(self.language_model)
        self._install_vllm_template_transport(self.api_predictor.language_model)

    @staticmethod
    def _install_vllm_template_transport(language_model: Any) -> None:
        """Pass vLLM template kwargs through AppWorld's restrictive wrapper."""
        if getattr(language_model, "_life_harness_vllm_transport", False):
            return
        original_lm_call = language_model.lm_call

        def lm_call_with_template_kwargs(**arguments: Any) -> dict[str, Any]:
            extra_body = arguments.get("extra_body")
            if (
                not isinstance(extra_body, dict)
                or arguments.get("client_name") != "openai"
                or arguments.get("api_type") != "chat_completions"
            ):
                return original_lm_call(**arguments)

            request_arguments = dict(arguments)
            request_arguments.pop("client_name", None)
            request_arguments.pop("api_type", None)
            api_key = request_arguments.pop("api_key", None)
            base_url = request_arguments.pop("base_url", None)
            client_arguments: dict[str, Any] = {}
            if api_key is not None:
                client_arguments["api_key"] = api_key
            if base_url is not None:
                client_arguments["base_url"] = base_url
            response = OpenAI(**client_arguments).chat.completions.create(
                **request_arguments
            )
            if hasattr(response, "to_dict"):
                return cast(dict[str, Any], response.to_dict())
            return cast(dict[str, Any], response.model_dump())

        language_model.lm_call = lm_call_with_template_kwargs
        language_model._life_harness_vllm_transport = True

    def initialize(self, world: AppWorld) -> None:
        super().initialize(world)
        self.h4.reset()
        self.harness_events.initialize(world.output_logs_directory)
        self.h5_block = ""
        self._logged_in_apps = set()

        # H5's only hook: one task-conditioned, procedural hint at episode start.
        if not self.harness_h5 or self.baseline:
            return
        skills = self.h5.retrieve(world.task.instruction)
        block = self.h5.format(skills)
        if not block:
            return
        self.h5_block = block
        system_index = next(
            (
                index
                for index, message in enumerate(self.messages)
                if message.get("role") == "system"
            ),
            None,
        )
        if system_index is None:
            self.messages.insert(0, {"role": "system", "content": block})
        else:
            original = str(self.messages[system_index].get("content", "")).rstrip()
            self.messages[system_index]["content"] = original + "\n\n" + block
        self.harness_events.log(
            "H5",
            0,
            "episode_start_skill_retrieval",
            "injected one-time procedural guidance",
            skill_ids=[skill.id for skill in skills],
        )

    def first_execution_inputs_usage_and_status(
        self,
    ) -> tuple[list[ExecutionIO], Usage, Status]:
        original_build_messages = self.api_predictor.build_messages

        def build_conditioned_messages(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            messages = original_build_messages(*args, **kwargs)
            if not self.h5_block:
                return messages
            for message in reversed(messages):
                if message.get("role") != "user":
                    continue
                content = str(message.get("content", "")).rstrip()
                message["content"] = (
                    content
                    + "\n\n"
                    + self.h5_block
                    + "\nUse this procedural guidance only to identify potentially needed APIs. "
                    + "Return API names in the required predictor format."
                )
                break
            return messages

        self.api_predictor.build_messages = build_conditioned_messages
        try:
            execution_inputs, usage, status = super().first_execution_inputs_usage_and_status()
        finally:
            self.api_predictor.build_messages = original_build_messages
        if self.harness_h3 and not status.failed:
            # H3's only hook: clone and calibrate descriptions of tools already
            # selected by the unmodified upstream API predictor.
            self.functions, augmented = self.h3.apply(self.functions)
            self.harness_events.log(
                "H3",
                self.step_number,
                "episode_tool_initialization",
                "calibrated tool contracts once",
                augmented_tools=augmented,
            )
        if not status.failed and not self.baseline:
            self._ensure_infrastructure_apis()
        return list(execution_inputs), usage, status

    _QUESTION_START = re.compile(
        r"^(?:what|how|which|who|whose|when|where|why|is|are|was|were|"
        r"do|does|did|can|could|would|should|will|count|name|list|find)\b",
        re.I,
    )

    def _is_answer_required(self) -> bool:
        instruction = self.world.task.instruction.strip()
        if "?" in instruction:
            return True
        return bool(self._QUESTION_START.match(instruction))

    def _strip_answer_if_action_only(self, arguments: dict) -> dict:
        if (
            "answer" in arguments
            and arguments["answer"] is not None
            and not self._is_answer_required()
        ):
            arguments = dict(arguments)
            arguments.pop("answer", None)
            self.harness_events.log(
                "H2",
                self.step_number,
                "action_only_answer_stripped",
                "stripped `answer` from complete_task for an action-only instruction",
            )
        return arguments

    def _ensure_infrastructure_apis(self) -> None:
        """Ensure authentication and credential APIs are always available.

        The predictor sometimes omits login and supervisor credential APIs,
        which makes the task unsolvable.  This injects them deterministically
        without changing the model or task semantics.
        """
        all_tools = {
            tool["function"]["name"]: tool
            for tool in self.world.task.api_docs.function_calling()
            if isinstance(tool, dict)
            and isinstance(tool.get("function"), dict)
            and isinstance(tool["function"].get("name"), str)
        }
        existing_names = {
            tool["function"]["name"] for tool in self.functions
        }
        essential_names = {"supervisor__show_profile", "supervisor__show_account_passwords"}
        for name in list(all_tools):
            if name.endswith("__login") and name not in existing_names:
                app = name.split("__", 1)[0]
                if app in self.world.task.instruction.lower():
                    essential_names.add(name)
        for name in sorted(essential_names):
            if name in all_tools and name not in existing_names:
                self.functions.append(all_tools[name])
                existing_names.add(name)
                self.harness_events.log(
                    "H2",
                    self.step_number,
                    "infrastructure_api_injection",
                    f"ensured essential API `{name}` is available",
                    function_name=name,
                )

    def second_onwards_execution_inputs_usage_and_status(
        self, last_execution_outputs: list[ExecutionIO]
    ) -> tuple[list[ExecutionIO], Usage, Status]:
        full_last_execution_output = ""
        for output in last_execution_outputs:
            self.messages.append(
                {
                    "tool_call_id": output.metadata["id"],
                    "role": "tool",
                    "name": output.metadata["function_name"],
                    "content": output.content,
                }
            )
            full_last_execution_output += output.content + "\n"
        if not last_execution_outputs and self.step_number > 2:
            full_last_execution_output = (
                "No function calls available. Please call at least one function."
            )
            self.world.execute("")
            self.messages.append({"role": "user", "content": full_last_execution_output})
        if full_last_execution_output:
            self.logger.show_message(
                role="environment",
                content=full_last_execution_output,
                step_number=self.step_number - 1,
            )

        message = self.language_model.generate(
            messages=self.messages, tools=self.functions, cache_control_at=-1
        )
        error_message = message.pop("error", None)
        if error_message:
            return [], Usage(), Status(failed=True, message=error_message)
        usage = message.pop("standardized_usage")
        reasoning_content = message.get("reasoning_content", "")
        content = (message.get("content", "") or "").strip()
        if not self.language_model.tool_parser and content:
            reasoning_content = (
                reasoning_content + f"\n\n{'-+' * 30}-\n\n" + content
                if reasoning_content
                else content
            )
        raw_tool_calls = message.get("tool_calls", []) or []
        if not isinstance(raw_tool_calls, list):
            raw_tool_calls = []
        raw_tool_calls = raw_tool_calls[: self.world.max_api_calls_per_interaction]
        message["tool_calls"] = raw_tool_calls
        self.messages.append(message)
        if self.harness_h4 and not raw_tool_calls:
            removed_characters = self.h4.compact_no_action_message(message)
            if removed_characters:
                self.harness_events.log(
                    "H4",
                    self.step_number,
                    "oversized_no_action_reasoning",
                    "removed oversized private reasoning from a turn with no executable action",
                    removed_characters=removed_characters,
                )

        # H2 validates against the environment's complete executable contract,
        # not H3's predictor-selected documentation subset. The full contract
        # is used only for validation and is never exposed to the model here.
        tools_by_name = {
            tool["function"]["name"]: tool["function"]
            for tool in self.world.task.api_docs.function_calling()
            if isinstance(tool, dict)
            and isinstance(tool.get("function"), dict)
            and isinstance(tool["function"].get("name"), str)
        }
        execution_inputs: list[ExecutionIO] = []
        displayed_actions: list[str] = []
        realized_entries: list[tuple[dict[str, Any], GateDecision]] = []
        for raw_call in raw_tool_calls:
            function = raw_call.get("function", {})
            name = str(function.get("name", ""))
            call_id = str(raw_call.get("call_id", raw_call.get("id", "harness-call")))
            raw_arguments = function.get("arguments", "{}")
            if self.harness_h2:
                decision = self.h2.realize(call_id, name, raw_arguments, tools_by_name)
            else:
                try:
                    arguments = (
                        json.loads(raw_arguments)
                        if isinstance(raw_arguments, str)
                        else raw_arguments
                    )
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
                if not isinstance(arguments, dict):
                    arguments = {}
                decision = GateDecision(call=ToolCall(call_id, name, arguments))

            realized_entries.append((raw_call, decision))

        if self.harness_h2:
            batch_decisions = self.h2.apply_batch(
                [decision for _, decision in realized_entries],
                step=self.step_number,
            )
            realized_entries = [
                (raw_call, decision)
                for (raw_call, _), decision in zip(
                    realized_entries, batch_decisions, strict=True
                )
            ]

        for raw_call, decision in realized_entries:
            function = raw_call.get("function", {})
            realized = decision.call
            function["name"] = realized.name
            function["arguments"] = json.dumps(realized.arguments, ensure_ascii=False)
            metadata = {
                "id": realized.call_id,
                "function_name": realized.name,
                "arguments": deepcopy(realized.arguments),
                "h2_blocked": decision.blocked,
                "h2_feedback": decision.feedback,
            }
            if decision.repairs:
                self.harness_events.log(
                    "H2",
                    self.step_number,
                    "representation_repair",
                    "; ".join(decision.repairs),
                    function_name=realized.name,
                )
            if decision.blocked:
                self.harness_events.log(
                    "H2",
                    self.step_number,
                    "invalid_action_block",
                    decision.feedback,
                    function_name=realized.name,
                )
                execution_inputs.append(ExecutionIO(content="", metadata=metadata))
                displayed_actions.append(decision.feedback)
                continue
            if realized.name.count(self.app_api_separator) != 1:
                continue
            app_name, api_name = realized.name.split(self.app_api_separator, 1)
            if (
                not self.baseline
                and "access_token" in realized.arguments
                and app_name not in self._logged_in_apps
                and app_name != "supervisor"
            ):
                decision = GateDecision(
                    call=realized,
                    blocked=True,
                    feedback=(
                        f"[H2 action gate] Cannot use `{app_name}` APIs before "
                        f"logging in. Call `{app_name}__login` first with the "
                        f"supervisor email and password, then use the returned "
                        f"access token."
                    ),
                )
                realized_entries[-1] = (raw_call, decision)
                self.harness_events.log(
                    "H2",
                    self.step_number,
                    "login_required_block",
                    f"blocked {realized.name} before login",
                    function_name=realized.name,
                )
                continue

            if realized.name == "supervisor__complete_task" and not self.baseline:
                realized = ToolCall(
                    realized.call_id,
                    realized.name,
                    self._strip_answer_if_action_only(realized.arguments),
                )
                function["arguments"] = json.dumps(realized.arguments, ensure_ascii=False)
            code = (
                f"print({app_name}{self.app_api_separator}{api_name}"
                f"(**{repr(realized.arguments)}))"
            )
            execution_inputs.append(ExecutionIO(content=code, metadata=metadata))
            displayed_actions.append(code)

        self.logger.show_message(
            role="agent",
            content="\n".join(displayed_actions),
            reasoning_content=reasoning_content,
            step_number=self.step_number,
            syntax="python",
        )
        return execution_inputs, usage, Status(failed=False)

    def solve_task(self, task_id: str) -> None:
        self.usage_tracker.reset(task_id)
        with AppWorld(task_id=task_id) as world:
            execution_outputs: list[ExecutionIO] = []
            self.initialize(world)
            for _ in range(self.max_steps):
                self.step_number += 1
                execution_inputs, usage, status = self.next_execution_inputs_usage_and_status(
                    execution_outputs
                )
                if status.failed:
                    self.logger.show_message(role="termination", content=status.message)
                    break

                allowed = [
                    item for item in execution_inputs if not item.metadata.get("h2_blocked")
                ]
                results = world.batch_execute([item.content for item in allowed]) if allowed else []
                result_iterator = iter(results)
                execution_outputs = []
                executed_calls: list[ToolCall] = []
                executed_results: list[str] = []
                for item in execution_inputs:
                    if item.metadata.get("h2_blocked"):
                        result = str(item.metadata.get("h2_feedback", "[H2] Action blocked."))
                    else:
                        result = next(result_iterator)
                        call = ToolCall(
                            call_id=str(item.metadata["id"]),
                            name=str(item.metadata["function_name"]),
                            arguments=dict(item.metadata.get("arguments", {})),
                        )
                        executed_calls.append(call)
                        executed_results.append(result)
                    execution_outputs.append(
                        ExecutionIO(content=result, metadata=deepcopy(item.metadata))
                    )

                # H4's only hook: inspect the completed environment transition
                # and append bounded feedback for the next model turn.
                if self.harness_h4:
                    interventions = self.h4.observe(
                        step=self.step_number,
                        max_steps=self.max_steps,
                        calls=executed_calls,
                        outputs=executed_results,
                        no_tool=not execution_inputs and self.step_number > 1,
                    )
                    feedback = "\n\n".join(message for _, message in interventions)
                    for trigger, intervention in interventions:
                        self.harness_events.log(
                            "H4", self.step_number, trigger, intervention
                        )
                    if feedback and execution_outputs:
                        execution_outputs[-1].content += "\n\n" + feedback
                    elif feedback:
                        self.messages.append({"role": "user", "content": feedback})

                self.usage_tracker.add(task_id, usage)
                self.log_usage()
                for call, result in zip(executed_calls, executed_results):
                    if (
                        call.name.endswith("__login")
                        and "access_token" in result
                    ):
                        app = call.name.split("__", 1)[0]
                        self._logged_in_apps.add(app)
                if world.task_completed() or self.usage_tracker.exceeded(task_id):
                    break
        self.logger.complete_task()
