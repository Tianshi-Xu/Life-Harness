"""Policy-free complete four-hook composition; never mutates old experiments."""
import copy
import hashlib
import inspect
import json
from pathlib import Path

from .contract import load_artifact

INTERFACE_VERSION = "life-agentbench-h2345-v3.1"
CONTEXT_FIELDS = {
    "dbbench": {"description", "type", "table", "evidence", "add_description"},
    "webshop": {"instruction"},
    "alfworld": {"instruction"},
}
DEFAULT_OPTIONS = {
    "h4_persistent": True,
    "h2_preserve_raw_history": True,
    "h4_message_prefix": "",
    "h5_message_prefix": "\n\nProcedural guidance:\n",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_digest(value):
    # Compatible with the historical bundle-v2 manifest encoder.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def manifest_records(path):
    value = json.loads(Path(path).read_text())
    supplied = value.pop("manifest_sha256", None)
    if value.get("schema_version") != 2 or supplied != object_digest(value):
        raise ValueError("invalid bundle-v2 manifest hash: schema_version must be 2 and "
                         "manifest_sha256 must match the manifest content; regenerate from the verified snapshot")
    root = Path(value["artifact_root"]).resolve()
    records = value["records"]
    if object_digest(records) != value["snapshot_sha256"]:
        raise ValueError(f"invalid snapshot hash: expected {object_digest(records)}, "
                         f"received {value['snapshot_sha256']!r}")
    seen = set()
    for row in records:
        artifact = (root / row["artifact_path"]).resolve()
        if (root not in artifact.parents or row["mechanism_id"] in seen
                or digest(artifact) != row["sha256"]):
            raise ValueError(f"invalid artifact path, identity or hash: mechanism={row['mechanism_id']!r}, "
                             f"artifact={row['artifact_path']!r}; require an in-root file, "
                             "unique mechanism ID and content matching its recorded sha256")
        seen.add(row["mechanism_id"])
    return root, records


def _hints(value):
    if not isinstance(value, list) or len(value) > 1 or not all(isinstance(x, str) for x in value):
        raise ValueError(f"H4/H5 return zero or one string; no word-count limit: expected [] or ['text']; "
                         f"received type={type(value).__name__}, count={len(value) if isinstance(value, list) else 'n/a'}")
    return value


def _skeleton(value, key):
    result = copy.deepcopy(value)
    if key == "content":
        result.pop("content", None)
    else:
        for tool in result:
            tool["function"].pop("description", None)
    return result


class CompositeHarness:
    def __init__(self, artifacts, domain, options=None):
        if domain not in CONTEXT_FIELDS:
            raise ValueError(f"unknown domain: {domain!r}; allowed={sorted(CONTEXT_FIELDS)}")
        self.domain = domain
        self.options = {**DEFAULT_OPTIONS, **(options or {})}
        if set(self.options) != set(DEFAULT_OPTIONS):
            raise ValueError(f"unknown session option: unexpected={sorted(set(self.options) - set(DEFAULT_OPTIONS))}; "
                             f"allowed={sorted(DEFAULT_OPTIONS)}")
        for name in ("h4_persistent", "h2_preserve_raw_history"):
            if not isinstance(self.options[name], bool):
                raise ValueError(f"session persistence flags must be boolean: {name} received {type(self.options[name]).__name__}")
        for name in ("h4_message_prefix", "h5_message_prefix"):
            if not isinstance(self.options[name], str):
                raise ValueError(f"message prefixes must be strings: {name} received {type(self.options[name]).__name__}")
        for name, value in self.options.items():
            setattr(self, name, value)
        self.shared_state = {}  # Fresh for every episode, not every turn.
        self.layers = []
        for path, target in artifacts:
            instance = load_artifact(path, target).Harness()
            self.layers.append((target, instance))
            bind = getattr(instance, "bind_shared_state", None)
            if callable(bind):
                bind(self.shared_state)  # Deliberately the SAME dictionary.

    def bind_task_context(self, context):
        if context is not None:
            if not isinstance(context, dict) or set(context) - CONTEXT_FIELDS[self.domain]:
                raise ValueError(f"only allowlisted public task-context fields may be bound: "
                                 f"domain={self.domain}, allowed={sorted(CONTEXT_FIELDS[self.domain])}, "
                                 f"received keys={list(context) if isinstance(context, dict) else type(context).__name__}")
        for _, instance in self.layers:
            bind = getattr(instance, "bind_task_context", None)
            if callable(bind):
                bind(copy.deepcopy(context))

    def h3(self, tools):
        current = copy.deepcopy(tools)
        for target, instance in self.layers:
            if target != "h3":
                continue
            before = copy.deepcopy(current)
            current = instance.h3(copy.deepcopy(current))
            if not isinstance(current, list) or _skeleton(before, "description") != _skeleton(current, "description"):
                raise ValueError(f"H3 changes tool descriptions only: preserve tool names, count/order and parameter schemas; "
                                 f"expected list count={len(before)}, received type={type(current).__name__}, "
                                 f"count={len(current) if isinstance(current, list) else 'n/a'}")
        return current

    def h3_message(self, message):
        current = copy.deepcopy(message)
        for target, instance in self.layers:
            hook = getattr(instance, "h3_message", None)
            if target == "h3" and callable(hook):
                before = copy.deepcopy(current)
                current = hook(copy.deepcopy(current))
                if not isinstance(current, dict) or _skeleton(before, "content") != _skeleton(current, "content"):
                    raise ValueError(f"H3 message conditioning changes content only: preserve role, IDs and tool calls; "
                                     f"received type={type(current).__name__}, "
                                     f"keys={list(current) if isinstance(current, dict) else 'n/a'}")
        return current

    def _guidance(self, target, *args):
        pieces = []
        for layer, instance in self.layers:
            if layer == target:
                pieces.extend(_hints(getattr(instance, target)(*copy.deepcopy(args))))
        return ["\n".join(pieces)] if pieces else []

    def h5(self, messages):
        return self._guidance("h5", messages)

    def h4(self, messages, remaining):
        return self._guidance("h4", messages, remaining)

    def h2(self, message, history):
        current = copy.deepcopy(message)
        for target, instance in self.layers:
            if target != "h2":
                continue
            current = instance.h2(copy.deepcopy(current), copy.deepcopy(history))
            if not isinstance(current, dict) or current.get("role") != "assistant":
                raise ValueError(f"H2 returns an assistant dictionary: expected dict with role='assistant'; "
                                 f"received type={type(current).__name__}, "
                                 f"role={current.get('role') if isinstance(current, dict) else 'n/a'}")
            if current.get("_harness_control"):
                break  # A later H2 must not erase an earlier veto.
        return current

    def h2_no_tool_message(self, message, remaining):
        hints = []
        for target, instance in self.layers:
            hook = getattr(instance, "h2_no_tool_message", None)
            if target == "h2" and callable(hook):
                value = hook(copy.deepcopy(message), remaining)
                if value is not None and not isinstance(value, str):
                    raise ValueError(f"H2 no-tool nudge must be a string or None: received {type(value).__name__}")
                if value:
                    hints.append(value)
        return "\n".join(hints) or None

    def drain_h2_messages(self):
        messages = []
        for target, instance in self.layers:
            hook = getattr(instance, "drain_h2_messages", None)
            if target == "h2" and callable(hook):
                values = hook()
                if not isinstance(values, list):
                    raise ValueError(f"H2 side messages must be a list: received {type(values).__name__}")
                for item in values:
                    if isinstance(item, str):
                        item = {"role": "user", "content": item}
                    if (not isinstance(item, dict) or item.get("role") != "user"
                            or not isinstance(item.get("content"), str)):
                        raise ValueError("H2 side messages must be public user messages: use a string or "
                                         "{'role':'user','content':'text'}, not tool/system/assistant messages")
                    messages.append(copy.deepcopy(item))
        return messages


def session_class(base):
    """Keep the Task bridge semantics; remove ONLY its obsolete word limit."""
    class CompleteFourHookSession(base):
        def _hints(self, layer, hints):
            hints = _hints(hints)
            if hints:
                self.trace[layer].append({"turn": self.turn, "hints": copy.deepcopy(hints)})
            return hints
    return CompleteFourHookSession


def interface_fingerprint(base, options=None):
    here = Path(__file__).resolve().parent
    return {"interface_version": INTERFACE_VERSION,
            "sources": {name: digest(here / name) for name in
                        ("__init__.py", "contract.py", "runtime.py", "INTERFACE.md",
                         "templates/h2.py", "templates/h3.py", "templates/h4.py", "templates/h5.py")},
            "base_session_sha256": digest(inspect.getsourcefile(base)),
            "session_options": {**DEFAULT_OPTIONS, **(options or {})}}


def install(task_module, domain, manifest_path, options=None):
    """Explicit worker-local integration, BEFORE Task construction.

    Return a receipt that MUST enter the new evaluator's cache/run fingerprint.
    No old evaluator, sealed source or process-global session is patched.
    """
    options = copy.deepcopy(options)
    root, records = manifest_records(manifest_path)
    artifacts = [(root / row["artifact_path"], row["target_layer"]) for row in records]
    CompositeHarness(artifacts, domain, options)  # Fail before any episode starts.
    base = task_module.FourHookSession
    receipt = {**interface_fingerprint(base, options),
               "chain_manifest_sha256": digest(manifest_path),
               "domain": domain}
    symbol = {"alfworld": "Harness", "dbbench": "DBBenchHarness", "webshop": "WebShopHarness"}[domain]
    def create(*_args, **_kwargs):
        for (path, _), record in zip(artifacts, records):
            if digest(path) != record["sha256"]:
                raise ValueError("accepted artifact changed after worker initialization")
        return CompositeHarness(artifacts, domain, options)

    setattr(task_module, symbol, create)
    task_module.FourHookSession = session_class(base)
    return receipt
