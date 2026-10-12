"""TauBench half-duplex adapter; user, environment and evaluator stay native."""
import json
import os
import time
import traceback
from pathlib import Path

from .session import NativeSession


def public_message(message):
    out = {"role": message.role, "content": message.content or ""}
    if getattr(message, "tool_calls", None):
        out["tool_calls"] = [{"id": call.id, "type": "function", "function": {
            "name": call.name, "arguments": json.dumps(call.arguments)}} for call in message.tool_calls]
    if message.role == "tool":
        out["tool_call_id"] = message.id
    return out


def agent_class():
    from pydantic import BaseModel
    from tau2.agent.base_agent import HalfDuplexAgent
    from tau2.agent.llm_agent import SYSTEM_PROMPT, AGENT_INSTRUCTION
    from tau2.data_model.message import AssistantMessage, ToolCall, MultiToolMessage

    class State(BaseModel):
        initialized: bool = True

    class NexAUAgent(HalfDuplexAgent):
        def __init__(self, tools, domain_policy, workspace, trace_dir, agent_config, capability_policy=None):
            super().__init__(tools, domain_policy)
            self.session = NativeSession(workspace, trace_dir,
                [tool.openai_schema for tool in tools], agent_config, capability_policy=capability_policy)
            self.session.inject({"role": "system", "content": SYSTEM_PROMPT.format(
                domain_policy=domain_policy, agent_instruction=AGENT_INSTRUCTION)})

        def get_init_state(self, message_history=None):
            for message in message_history or []:
                self.session.inject(public_message(message))
            return State()

        def generate_next_message(self, message, state):
            if isinstance(message, MultiToolMessage):
                for item in message.tool_messages:
                    self.session.inject(public_message(item))
            else:
                self.session.inject(public_message(message))
            raw = self.session.sync_action().messages[0]
            if raw.get("tool_calls"):
                calls = [ToolCall(id=call["id"], name=call["function"]["name"],
                    arguments=json.loads(call["function"]["arguments"]), requestor="assistant")
                    for call in raw["tool_calls"]]
                return AssistantMessage(role="assistant", content=None, tool_calls=calls), state
            return AssistantMessage(role="assistant", content=raw["content"]), state

        def stop(self, message=None, state=None):
            self.session.close()

    return NexAUAgent


def run_tau(request):
    from tau2.registry import registry
    from tau2.data_model.simulation import TextRunConfig
    from tau2.run import get_tasks, run_single_task
    from tau2.evaluator.evaluator import EvaluationType
    Agent = agent_class()
    agents = []

    def factory(tools, domain_policy, **kwargs):
        agent = Agent(tools, domain_policy, request["workspace"], request["trace_dir"],
                      request.get("agent_config", "code_agent.yaml"), request.get("capability_policy"))
        agents.append(agent)
        return agent

    registry.register_agent_factory(factory, "ahe_native")
    user_model = os.environ.get("AHE_TAU_USER_MODEL", "openai/deepseek-v4-flash")
    user_args = {"api_base": os.environ.get("AHE_TAU_USER_API_BASE") or os.environ.get("AHE_USER_API_BASE",
        "https://api.example.com/protocol/openai/v1"),
        "api_key": os.environ.get("AHE_USER_API_KEY", ""),
        "max_tokens": 100000 if user_model == "openai/deepseek-v4-flash" else 4096}
    if not user_args["api_key"]:
        raise RuntimeError("AHE_USER_API_KEY is required for TauBench's native user simulator")
    if user_model == "openai/qwen3.8-flash":
        user_args["extra_body"] = {"enable_thinking": True}
    config = TextRunConfig(domain=request["domain"], agent="ahe_native",
        llm_agent="openai/qwen3-4b", llm_user=user_model,
        llm_args_user=user_args, max_concurrency=1, num_trials=1,
        task_split_name=request["split"], harness_enabled=False, harness_h3=False,
        harness_h4=False, harness_h5=False)
    tasks = get_tasks(request["domain"], task_ids=[str(request["task_id"])],
                      task_split_name=request["split"])
    if len(tasks) != 1:
        raise RuntimeError("native Tau task selection did not return exactly one task")
    start = time.monotonic()
    try:
        # Retail's original comparison used natural-language assertions. Keep
        # that evaluator mode, using the same user/judge API route.
        if request["domain"] == "retail":
            import tau2.config as tau_config
            import tau2.evaluator.evaluator_nl_assertions as nl
            tau_config.DEFAULT_LLM_NL_ASSERTIONS = user_model
            nl.DEFAULT_LLM_NL_ASSERTIONS = user_model
            judge_args = {**user_args, "api_base": os.environ.get("AHE_TAU_JUDGE_API_BASE") or user_args["api_base"]}
            tau_config.DEFAULT_LLM_NL_ASSERTIONS_ARGS.update(judge_args)
        result = run_single_task(config, tasks[0], seed=42 + request["trial"],
            evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS if request["domain"] == "retail"
            else EvaluationType.ALL)
        success = float(result.reward_info.reward if result.reward_info else 0) >= 1
        row = {"success": int(success), "status": str(result.termination_reason),
               "infrastructure_error": False}
    except Exception as exc:
        traceback.print_exc()
        context_limit = "solver_context_limit_32768" in str(exc)
        row = {"success": 0, "status": "context_limit" if context_limit else type(exc).__name__,
               "infrastructure_error": not context_limit}
    finally:
        for agent in agents:
            agent.session.close()
    row.update(task_id=str(request["task_id"]), trial=request["trial"],
        messages=agents[0].session.history if agents else [{"role": "user", "content": "Initialization failed."}],
        wall_seconds=time.monotonic() - start)
    return row
