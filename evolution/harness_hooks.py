"""Shared, policy-free H2--H5 contract for new AgentBench and TauBench runs.

Reuse the frozen composition/validator rather than maintaining a second set of
hooks. Benchmark adapters supply public context and perform native transport.
"""
import copy
import ast
import importlib.util
import re
from pathlib import Path

from evolution.agentbench_hooks_v31 import runtime as legacy
from evolution.agentbench_hooks_v31.contract import validate_source as legacy_validate_source
from evolution.environment_state import ReadOnlyDict, apply_changes, changes, read_only, snapshot

INTERFACE_VERSION = "life-h2345-unified-v2"
CONTEXT_FIELDS = {
    **legacy.CONTEXT_FIELDS,
    "os": {"description"},
    **{domain: {"instruction", "policy"} for domain in ("airline", "retail", "telecom")},
}
BUILTIN_DOMAINS = frozenset(CONTEXT_FIELDS)
DEFAULT_OPTIONS = {**legacy.DEFAULT_OPTIONS, "h2_preserve_raw_history": False,
                   "h4_message_prefix": "[H4] "}
digest = legacy.digest
object_digest = legacy.object_digest
manifest_records = legacy.manifest_records
TEMPLATES = Path(legacy.__file__).parent / "templates"
CONTRACT = Path(__file__).with_name("HARNESS_INTERFACE.md")


def register_domain(name, public_context_fields):
    """Register a trusted adapter's schema without changing existing domains."""
    identifier = re.compile(r"[a-z][a-z0-9_]{0,63}")
    if not isinstance(name, str) or not identifier.fullmatch(name):
        raise ValueError("adapter name must be a lowercase identifier")
    if not isinstance(public_context_fields, (list, tuple, set, frozenset)):
        raise ValueError("public_context_fields must be a collection of field names")
    if any(not isinstance(field, str) or not identifier.fullmatch(field) for field in public_context_fields):
        raise ValueError("public context field names must be lowercase identifiers")
    fields = set(public_context_fields)
    if name in CONTEXT_FIELDS and CONTEXT_FIELDS[name] != fields:
        raise ValueError("cannot replace an existing domain's public-context schema")
    CONTEXT_FIELDS[name] = fields


def validate_source(path, target):
    tree = legacy_validate_source(path, target)
    harness = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Harness")
    for node in harness.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {"h3_messages", "bind_environment_state"}:
            args = node.args
            if ((node.name == "h3_messages" and target != "h3") or isinstance(node, ast.AsyncFunctionDef)
                    or len(args.args) + len(args.posonlyargs) != 2 or args.vararg or args.kwarg
                    or args.kwonlyargs or args.defaults):
                raise ValueError(f"invalid optional hook signature/layer: {node.name} at line {node.lineno}; "
                                 f"expected two positional arguments including self, no defaults/varargs/keyword-only/async; "
                                 f"h3_messages belongs to h3; received layer={target!r}, args=({ast.unparse(args)})")
    return tree


class CompositeHarness(legacy.CompositeHarness):
    def __init__(self, artifacts, domain, options=None):
        if domain not in CONTEXT_FIELDS:
            raise ValueError(f"unknown public-context adapter: {domain!r}; allowed={sorted(CONTEXT_FIELDS)}")
        self.domain = domain
        self.options = {**DEFAULT_OPTIONS, **(options or {})}
        if set(self.options) != set(DEFAULT_OPTIONS):
            raise ValueError(f"unknown session option: unexpected={sorted(set(self.options) - set(DEFAULT_OPTIONS))}; "
                             f"allowed={sorted(DEFAULT_OPTIONS)}")
        for name, default in DEFAULT_OPTIONS.items():
            if type(self.options[name]) is not type(default):
                raise ValueError(f"invalid session option type: {name}; expected {type(default).__name__}, "
                                 f"received {type(self.options[name]).__name__}")
            setattr(self, name, self.options[name])
        self.shared_state = {}
        self._environment_state_provider = None
        self._environment_state_source = object()
        self._environment_state_views = {}
        self.layers = []
        for path, target in artifacts:
            instance = legacy.load_artifact(path, target).Harness()
            self.layers.append((target, instance))
            bind = getattr(instance, "bind_shared_state", None)
            if callable(bind):
                bind(self.shared_state)
        self.bind_environment_state(None)

    def bind_environment_state(self, state):
        """No live DB or write-back handle is ever handed to an artifact."""
        # Trusted frozen host views can be reused while DB contents are unchanged.
        # Plain provider dictionaries are validated/copied on every bind instead.
        if (state is None or isinstance(state, ReadOnlyDict)) and state is self._environment_state_source:
            return
        old = self._environment_state_source
        patch = changes(old, state) if ((old is None or isinstance(old, ReadOnlyDict))
                                       and (state is None or isinstance(state, ReadOnlyDict))) else None
        state = state if isinstance(state, ReadOnlyDict) else snapshot(state)
        for _, instance in self.layers:
            bind = getattr(instance, "bind_environment_state", None)
            if callable(bind):
                view = (apply_changes(self._environment_state_views[id(instance)], patch)
                        if patch is not None else read_only(state))
                self._environment_state_views[id(instance)] = view
                bind(view)
        self._environment_state_source = state

    def set_environment_state_provider(self, provider):
        if provider is not None and not callable(provider):
            raise ValueError(f"environment-state provider must be a host callback or None, "
                             f"not {type(provider).__name__}; artifacts receive read-only business snapshots")
        self._environment_state_provider = provider
        self.refresh_environment_state()

    def refresh_environment_state(self):
        if self._environment_state_provider is not None:
            self.bind_environment_state(self._environment_state_provider())

    def bind_task_context(self, context):
        if context is not None:
            if not isinstance(context, dict) or set(context) - CONTEXT_FIELDS[self.domain]:
                raise ValueError(f"only allowlisted public task-context fields may be bound: domain={self.domain}, "
                                 f"allowed={sorted(CONTEXT_FIELDS[self.domain])}, "
                                 f"received keys={list(context) if isinstance(context, dict) else type(context).__name__}")
        for _, instance in self.layers:
            bind = getattr(instance, "bind_task_context", None)
            if callable(bind):
                bind(copy.deepcopy(context))

    def h3_messages(self, messages):
        """Condition the model-visible view every turn; never mutate actual history."""
        current = copy.deepcopy(messages)
        for target, instance in self.layers:
            hook = getattr(instance, "h3_messages", None)
            if not callable(hook):
                continue
            if target != "h3":
                raise ValueError(f"h3_messages belongs to H3: received layer={target!r}")
            before = copy.deepcopy(current)
            current = hook(copy.deepcopy(current))
            if (not isinstance(current, list) or not all(isinstance(m, dict) for m in current)
                    or [legacy._skeleton(m, 'content') for m in current]
                    != [legacy._skeleton(m, 'content') for m in before]):
                raise ValueError(f"H3 model-view conditioning changes content only: preserve message count/order, roles, "
                                 f"IDs and tool-call metadata; expected count={len(before)}, "
                                 f"received type={type(current).__name__}, "
                                 f"count={len(current) if isinstance(current, list) else 'n/a'}")
        return current

    def h2(self, message, history):
        value = super().h2(message, history)
        control = value.get("_harness_control")
        if control is not None:
            if (not isinstance(control, dict) or set(control) != {"action", "messages"}
                    or control["action"] != "suppress_tool"
                    or not isinstance(control["messages"], list) or not control["messages"]):
                raise ValueError("H2 veto requires suppress_tool and nonempty corrective messages: "
                                 "_harness_control must have exactly action='suppress_tool' and messages=[...] "
                                 "with at least one public user/tool corrective message")
            calls = message.get("tool_calls") or []
            ids = {call.get("id") for call in calls}
            if len(ids) != len(calls) or any(not isinstance(i, str) or not i for i in ids):
                raise ValueError(f"suppressed calls require distinct nonempty IDs: received IDs={list(ids)!r}, "
                                 f"tool_call count={len(calls)}")
            answered = set()
            for item in control["messages"]:
                if (not isinstance(item, dict) or item.get("role") not in {"user", "tool"}
                        or not isinstance(item.get("content"), str) or not item["content"].strip()):
                    raise ValueError("invalid public H2 corrective message: expected dict with role='user' or 'tool' "
                                     "and nonblank string content; tool messages also need a proposed tool_call_id")
                if item["role"] == "tool":
                    call_id = item.get("tool_call_id")
                    if call_id not in ids or call_id in answered:
                        raise ValueError(f"H2 feedback must reference a distinct proposed tool-call ID: "
                                         f"received {call_id!r}, proposed IDs={sorted(ids)}, already answered={sorted(answered)}")
                    answered.add(call_id)
            # Close suppressed calls for native chat protocols. This is transport
            # feedback, not a fabricated environment result or recovery policy.
            control["messages"] = [
                {"role": "tool", "tool_call_id": call["id"],
                 "content": "[H2] Action suppressed before execution; no environment action occurred."}
                for call in message.get("tool_calls") or [] if call.get("id") not in answered
            ] + control["messages"]
        return value


def session_class(base):
    complete = legacy.session_class(base)

    class SharedFourHookSession(complete):
        def inject(self, item):
            self.harness.refresh_environment_state()
            return super().inject(item)

        def _prepare_action(self):
            self.harness.refresh_environment_state()
            original, public = super()._prepare_action()
            try:
                if self.switches.h3_enabled:
                    before = copy.deepcopy(self.session.history)
                    conditioned = self.harness.h3_messages(before)
                    if conditioned != before:
                        self.trace['h3'].append({'turn': self.turn, 'before': before, 'after': copy.deepcopy(conditioned)})
                    self.session.cover(conditioned)
            except BaseException:
                self.session.cover(original)
                raise
            return original, public

        def _finish_action(self, output, original, public):
            try:
                self.harness.refresh_environment_state()
                record = getattr(self.session, "record_action_environment_state", None)
                if callable(record):
                    record()
            except BaseException:
                self.session.cover(original)
                raise
            return super()._finish_action(output, original, public)

    return SharedFourHookSession


def native_session_class():
    path = Path(__file__).resolve().parents[1] / "AgentBench/src/server/harness/session.py"
    spec = importlib.util.spec_from_file_location("life_shared_native_session", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return session_class(module.FourHookSession)


def interface_fingerprint(base, options=None):
    receipt = legacy.interface_fingerprint(base, {**DEFAULT_OPTIONS, **(options or {})})
    receipt.update(interface_version=INTERFACE_VERSION,
                   common_sources={str(p): digest(p) for p in (Path(__file__), CONTRACT,
                                   Path(__file__).with_name("environment_state.py"))},
                   environment_state_contract="detached business-db JSON; refreshed before model/action; no write-back",
                   public_context_fields={d: sorted(fields) for d, fields in CONTEXT_FIELDS.items()})
    return receipt


def install_agentbench(task_module, domain, manifest_path, options=None, environment_state_provider=None):
    """Opt-in new-run adapter; do not alter any historical evaluator."""
    if domain not in (*legacy.CONTEXT_FIELDS, "os"):
        raise ValueError("not an AgentBench adapter")
    options = copy.deepcopy(options)
    root, records = manifest_records(manifest_path)
    artifacts = [(root / row["artifact_path"], row["target_layer"]) for row in records]
    for path, target in artifacts:
        validate_source(path, target)
    CompositeHarness(artifacts, domain, options)
    base = task_module.FourHookSession
    receipt = {**interface_fingerprint(base, options), "domain": domain,
               "environment_state_binding": environment_state_provider is not None,
               "chain_manifest_sha256": digest(manifest_path), "adapter": "agentbench"}

    def create(*_args, **_kwargs):
        if any(digest(path) != row["sha256"] for (path, _), row in zip(artifacts, records)):
            raise ValueError("accepted artifact changed after worker initialization")
        harness = CompositeHarness(artifacts, domain, options)
        harness.set_environment_state_provider(environment_state_provider)
        return harness

    symbol = {"alfworld": "Harness", "dbbench": "DBBenchHarness", "webshop": "WebShopHarness",
              "os": "OSInteractionHarness"}[domain]
    setattr(task_module, symbol, create)
    task_module.FourHookSession = session_class(base)
    return receipt
