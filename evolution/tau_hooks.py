"""Thin native TauBench adapter for the shared, empty-by-default four hooks."""
import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from pydantic import PrivateAttr
from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import (AssistantMessage, MultiToolMessage,
                                    SystemMessage, ToolCall, ToolMessage, UserMessage)
from tau2.orchestrator.orchestrator import Orchestrator, Role
from tau2.utils.llm_utils import generate, to_litellm_messages

from evolution import harness_hooks as hooks
from evolution.environment_state import StateLog

CompleteSession = hooks.native_session_class()


def business_environment_state(environment):
    """Agent-side business DB only; never environment/task/user-simulator objects."""
    db = getattr(getattr(environment, "tools", None), "db", None)
    return None if db is None else {"db": db.model_dump(mode="json")}


def public_message(message):
    value = to_litellm_messages([message])[0]
    for call in value.get("tool_calls") or []:
        call.pop("name", None)
    return value


def native_message(value):
    role = value["role"]
    if role == "assistant":
        calls = [ToolCall(id=c["id"], name=c["function"]["name"],
                          arguments=json.loads(c["function"]["arguments"]))
                 for c in value.get("tool_calls") or []]
        return AssistantMessage(role=role, content=value.get("content"), tool_calls=calls or None)
    if role == "tool":
        return ToolMessage(role=role, id=value["tool_call_id"], content=value.get("content"))
    cls = {"system": SystemMessage, "user": UserMessage}[role]
    return cls(role=role, content=value.get("content"))


class SchemaTool:
    """Only model-facing descriptions differ; native environment dispatch is untouched."""
    def __init__(self, schema):
        self.openai_schema = copy.deepcopy(schema)
        self.name = schema["function"]["name"]


class Backend:
    def __init__(self, agent):
        self.agent = agent
        self.history = []
        self.controller = SimpleNamespace(env_output=SimpleNamespace(history=[]))
        self.calls = []
        self.environment_states = StateLog()

    def capture_environment_state(self):
        provider = self.agent._life_environment_state_provider
        self.environment_states.record(provider() if provider is not None else None)
        return self.environment_states.view

    def record_action_environment_state(self):
        self.calls[-1]["action_environment_state_index"] = self.environment_states.index

    def set_tools(self, tools):
        self.tools = [SchemaTool(schema) for schema in tools]

    def set_full_history(self, enabled):
        pass  # Native generate already receives full history.

    def inject(self, message):
        self.history.append(copy.deepcopy(message))

    def cover(self, messages):
        self.history = copy.deepcopy(messages)

    def sync_action(self):
        self.last_response = generate(
            model=self.agent.llm, tools=self.tools,
            messages=[native_message(m) for m in self.history],
            call_name="agent_response", **self.agent.llm_args)
        raw = public_message(self.last_response)
        # Public TRAIN export only: never record llm_args, secrets or simulator inputs.
        self.calls.append({"turn": len(self.calls) + 1, "remaining": self.agent._life_remaining,
                           "environment_state_index": self.environment_states.index,
                           "request": {"messages": copy.deepcopy(self.history),
                                       "tools": [copy.deepcopy(t.openai_schema) for t in self.tools]},
                           "response": {"choices": [{"message": copy.deepcopy(raw)}]}})
        return SimpleNamespace(messages=[raw])


class State(LLMAgentState):
    _life_session: object = PrivateAttr()
    _life_retry: bool = PrivateAttr(default=False)
    _life_cost: float = PrivateAttr(default=0.0)
    _life_usage: dict = PrivateAttr(default_factory=dict)


class SuppressedAction(Exception):
    """One solver turn was consumed, with no user/environment action dispatched."""


class LifeAgent(LLMAgent):
    def __init__(self, *args, harness_factory, **kwargs):
        super().__init__(*args, **kwargs)
        self._harness_factory = harness_factory
        self._life_native_tools = self.tools
        self._life_remaining = 1  # The native orchestrator binds the actual budget.
        self._life_turn_limit = 1
        self._life_environment_state_provider = None

    def get_init_state(self, message_history=None):
        original = super().get_init_state(message_history)
        state = State(**original.model_dump())
        backend = Backend(self)
        harness = self._harness_factory()
        harness.set_environment_state_provider(backend.capture_environment_state)
        backend.initial_environment_state_index = backend.environment_states.index
        state._life_session = CompleteSession(
            backend, harness, [t.openai_schema for t in self._life_native_tools],
            self._life_remaining, SimpleNamespace(**{f"h{i}_enabled": True for i in range(2, 6)}),
            {"policy": self.domain_policy})
        self.tools = backend.tools  # Native result snapshots must show actual model-facing schemas.
        for message in state.system_messages + state.messages:
            state._life_session.inject(public_message(message))
        return state

    def generate_next_message(self, message, state):
        session = state._life_session
        if not state._life_retry and message is not None:
            incoming = message.tool_messages if isinstance(message, MultiToolMessage) else [message]
            for item in incoming:
                session.inject(public_message(item))
        session.limit = session.turn + self._life_remaining
        result = session.sync_action()
        response = session.session.last_response
        state._life_cost += response.cost or 0.0
        for key, amount in (response.usage or {}).items():
            if isinstance(amount, (int, float)):
                state._life_usage[key] = state._life_usage.get(key, 0) + amount
        history = [native_message(m) for m in session.session.history]
        state.system_messages = [m for m in history if isinstance(m, SystemMessage)]
        state.messages = [m for m in history if not isinstance(m, SystemMessage)]
        state._life_retry = session.consume_h2_suppressed()
        if state._life_retry:
            raise SuppressedAction()
        repaired = native_message(result.messages[0])
        repaired = response.model_copy(update={"content": repaired.content,
            "tool_calls": repaired.tool_calls, "cost": state._life_cost,
            "usage": state._life_usage or response.usage})
        state._life_cost, state._life_usage = 0.0, {}
        return repaired, state

    def public_episode(self, state, *, index, success, status, infrastructure_error=False):
        session = state._life_session
        return {"index": index, "success": int(success), "status": str(status),
                "infrastructure_error": bool(infrastructure_error),
                "turns": session.turn, "turn_limit": self._life_turn_limit,
                "history": copy.deepcopy(session.session.history),
                "calls": copy.deepcopy(session.session.calls),
                "environment_states": copy.deepcopy(session.session.environment_states.records),
                "initial_environment_state_index": session.session.initial_environment_state_index,
                "harness_trace": [{"layer": layer, **event}
                                  for layer, events in session.trace.items() for event in events]}


class LifeOrchestrator(Orchestrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.agent, LifeAgent):
            self.agent._life_remaining = self.agent._life_turn_limit = self.max_steps
            self.agent._life_environment_state_provider = lambda: business_environment_state(self.environment)

    def step(self):
        if isinstance(self.agent, LifeAgent) and self.to_role == Role.AGENT:
            self.agent._life_remaining = self.max_steps - self.step_count
        try:
            return super().step()
        except SuppressedAction:
            if not isinstance(self.agent, LifeAgent) or self.to_role != Role.AGENT:
                raise
            # Leave native routing unchanged; next step resumes the same agent
            # without replaying the incoming message. Native termination applies.
            self.step_count += 1
            self.environment.sync_tools()

    def _finalize(self):
        result = super()._finalize()
        if isinstance(self.agent, LifeAgent) and self.agent_state is not None:
            episode = self.agent.public_episode(
                self.agent_state, index=hooks.object_digest([result.task_id, result.seed])[:24],
                success=0, status=result.termination_reason)
            result.info = {**(result.info or {}), "life_public_episode": episode}
        return result


def training_case_id(task_id):
    """Opaque, path-safe evidence identity; Telecom IDs encode private fault labels."""
    value = str(task_id)
    return "native_" + hashlib.sha256(value.encode()).hexdigest()


def export_training(results, domain, manifest_path, destination, interface_receipt=None):
    """Normalize a completed native TRAIN batch for Life prepare; no private task data."""
    root, records = hooks.manifest_records(manifest_path)
    destination = Path(destination)
    rows = []
    for simulation in results.simulations:
        row = copy.deepcopy((simulation.info or {}).get("life_public_episode"))
        if row is None:
            raise ValueError("native simulation is missing its Life public trace")
        row["index"] = training_case_id(getattr(simulation, "task_id", row["index"]))
        row["success"] = int(simulation.reward_info is not None and simulation.reward_info.reward == 1)
        row["infrastructure_error"] = simulation.reward_info is None or simulation.termination_reason in {"agent_error", "user_error"}
        if row["infrastructure_error"]:
            row["success"] = 0
        rows.append(row)
    if not rows or len({row["index"] for row in rows}) != len(rows):
        raise ValueError("empty or duplicate native TRAIN episode identities")
    if destination.exists():
        raise ValueError("use a new public TRAIN export directory; historical artifacts are never overwritten")
    destination.mkdir(parents=True)
    for row in rows:
        (destination / f"episode_{row['index']}.json").write_text(json.dumps(row, ensure_ascii=False) + "\n")
    outcomes = {row["index"]: row["success"] for row in rows}
    summary = {"split": "train", "domain": domain, "n": len(rows), "per_task": outcomes,
               "successes": sum(outcomes.values()), "infra_errors": sum(row["infrastructure_error"] for row in rows),
               "chain_hashes": {**{r["artifact_path"]: r["sha256"] for r in records},
                                "manifest": hooks.digest(manifest_path)}}
    if interface_receipt is not None:
        summary["life_interface"] = copy.deepcopy(interface_receipt)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def install(domain, manifest_path, options=None):
    """Opt-in worker-local installation using the public agent registry."""
    if domain not in {"airline", "retail", "telecom"}:
        raise ValueError("unsupported TauBench adapter")
    from tau2.registry import registry
    from tau2.runner import build

    options = copy.deepcopy(options)
    root, records = hooks.manifest_records(manifest_path)
    artifacts = [(root / row["artifact_path"], row["target_layer"]) for row in records]
    for path, target in artifacts:
        hooks.validate_source(path, target)

    def fresh():
        if any(hooks.digest(path) != row["sha256"] for (path, _), row in zip(artifacts, records)):
            raise ValueError("accepted artifact changed after worker initialization")
        return hooks.CompositeHarness(artifacts, domain, options)

    fresh()  # Fail before any solver or simulator request.
    receipt = {**hooks.interface_fingerprint(Orchestrator, options), "domain": domain,
               "adapter": "taubench", "adapter_sha256": hooks.digest(__file__),
               "native_agent_sha256": hooks.digest(inspect.getsourcefile(LLMAgent)),
               "native_session_sha256": hooks.digest(Path(__file__).resolve().parents[1]
                                                      / "AgentBench/src/server/harness/session.py"),
               "chain_manifest_sha256": hooks.digest(manifest_path)}
    name = "life_h2345_" + hooks.object_digest(receipt)[:16]
    if registry.get_agent_factory(name) is None:
        def factory(tools, domain_policy, **kwargs):
            return LifeAgent(tools=tools, domain_policy=domain_policy,
                             llm=kwargs.get("llm"), llm_args=kwargs.get("llm_args"),
                             harness_factory=fresh)
        registry.register_agent_factory(factory, name)
    if build.Orchestrator not in {Orchestrator, LifeOrchestrator}:
        raise ValueError("another orchestrator adapter is already installed")
    build.Orchestrator = LifeOrchestrator
    return name, receipt


def evaluation_cli(argv=None):
    """Reuse native evaluation CLI without editing sealed Meta/AHE source files."""
    import argparse
    import importlib.util
    import sys

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--life-manifest", required=True)
    flags, native_argv = parser.parse_known_args(argv)
    path = Path(__file__).resolve().parents[1] / "TauBench/scripts/eval_harness.py"
    spec = importlib.util.spec_from_file_location("life_native_tau_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    original_argv = sys.argv
    try:
        sys.argv = [str(path)] + native_argv
        args = cli.parse_args()
    finally:
        sys.argv = original_argv
    if (args.harness_plugin or args.enabled or args.retrieve_policy
            or args.h2 or args.h3 or args.h4 or args.h5):
        raise ValueError("--life-manifest selects all four empty/evolved hooks; do not mix legacy plugins/flags")
    name, receipt = install(args.domain, flags.life_manifest)
    receipt["native_cli_sha256"] = hooks.digest(path)
    native_run, native_summary = cli.run_tasks, cli.summarise

    def run(config, tasks, **kwargs):
        config = config.model_copy(update={"agent": name})
        results = native_run(config, tasks, **kwargs)
        if args.split == "train":
            export_training(results, args.domain, flags.life_manifest,
                            Path(kwargs["save_dir"]) / "life_train", receipt)
        return results

    def summarise(*args, **kwargs):
        summary = native_summary(*args, **kwargs)
        simulations = args[0].simulations
        infra = [s.reward_info is None or s.termination_reason in {"agent_error", "user_error"}
                 for s in simulations]
        successes = sum(not error and s.reward_info.reward == 1
                        for s, error in zip(simulations, infra))
        # Native legacy aggregates omit missing rewards. Canonical Life binary
        # accuracy retains every completed episode in the denominator.
        return {**summary, "life_interface": receipt, "n": len(simulations),
                "successes": successes, "infra_errors": sum(infra),
                "accuracy": successes / len(simulations) if simulations else 0.0}

    cli.parse_args = lambda: args
    cli.run_tasks, cli.summarise = run, summarise
    cli.main()


if __name__ == "__main__":
    evaluation_cli()
