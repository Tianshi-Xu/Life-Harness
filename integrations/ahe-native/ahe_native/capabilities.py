"""Budget-restricted AHE port policy; never changes upstream optimizer sources.

This is a compatibility/cost contract, not a general Python security sandbox.
The existing private filesystem remains necessary. Candidate validation runs
before evaluation and before promotion; trusted runtime guards additionally
limit model and native-environment dispatch to the ordinary executor path.
"""
import ast
import hashlib
import json
import re
from pathlib import Path

import yaml

PROFILE = "h2345-budget-restricted-v1"
MAX_ANALYSIS_TASKS = 10
ROOT = Path(__file__).resolve().parents[1]
HOOK_MODULE = "nexau.archs.main_sub.execution.hooks"
HOOK_NAMES = {"Middleware", "HookResult", "BeforeAgentHookInput", "AfterAgentHookInput",
              "BeforeModelHookInput", "AfterModelHookInput", "BeforeToolHookInput", "AfterToolHookInput"}
PURE_NEXAU_IMPORTS = {HOOK_MODULE: HOOK_NAMES,
    "nexau.core.messages": {"Message", "Role", "TextBlock", "ToolUseBlock", "ToolResultBlock"},
    "nexau.archs.main_sub.execution.parse_structures": {"ToolCall", "ParsedResponse"}}
SAFE_IMPORTS = {
    "__future__": {"annotations"},
    "re": {"compile", "escape", "findall", "finditer", "fullmatch", "match", "search", "split", "sub", "subn",
           "I", "IGNORECASE", "S", "DOTALL", "M", "MULTILINE", "A", "ASCII", "X", "VERBOSE", "Pattern", "Match"},
    "json": {"dumps", "loads"}, "ast": {"literal_eval"}, "copy": {"copy", "deepcopy"},
    "datetime": {"date", "datetime", "time", "timedelta", "timezone", "tzinfo", "MINYEAR", "MAXYEAR"},
    "math": {"ceil", "floor", "sqrt", "log", "log10", "exp", "isclose", "isfinite", "inf", "nan", "pi", "e", "prod", "fsum"},
    "collections": {"Counter", "defaultdict", "deque"}, "difflib": {"SequenceMatcher", "get_close_matches"},
    "string": {"ascii_letters", "ascii_lowercase", "ascii_uppercase", "digits", "punctuation", "whitespace"},
    "typing": {"Any", "Callable", "Iterable", "Mapping", "Sequence", "Literal", "Optional", "Union", "TypedDict"},
    "dataclasses": {"dataclass", "field", "fields", "replace"},
    "functools": {"partial", "reduce", "cache", "lru_cache"},
    "itertools": {"chain", "combinations", "permutations", "product", "islice", "repeat", "zip_longest", "groupby", "accumulate"},
    "statistics": {"mean", "median", "fmean", "pstdev", "pvariance"}}
SAFE_MODULES = set(SAFE_IMPORTS)
FORBIDDEN = {"open", "eval", "exec", "compile", "__import__", "globals", "locals", "vars",
             "setattr", "delattr", "input", "breakpoint", "type", "BaseException"}
BLOCKED_ATTRIBUTES = {"executor", "_executor", "sandbox", "_sandbox", "sandbox_manager",
    "_sandbox_manager", "variables", "_variables", "parent_agent_state", "team_state",
    "openai_client", "llm_client", "llm_config", "client", "chat", "completions",
    "wrap_model_call", "wrap_tool_call", "stream_chunk", "register_tool", "register_tools",
    "add_tool", "add_tools", "remove_tool", "invoke", "run", "run_async", "execute",
    "execute_tool", "read_text", "read_bytes", "write_text", "write_bytes", "open",
    "load", "dump", "parse", "sys", "os", "builtins", "modules", "socket",
    "subprocess", "importlib", "inspect", "httpx", "urllib", "requests", "f_globals", "gi_frame", "tb_frame"}
PUBLIC_GETATTR = {"messages", "content", "role", "tool_name", "tool_input", "tool_output",
                  "agent_state", "agent_id", "current_iteration", "max_iterations", "parsed_response",
                  "get_context_value", "set_context_value", "tool_calls", "parameters", "tool_call_id",
                  # Public data fields of the permitted Message/block/response/hook models.
                  # This does not expose AgentState execution, config or global-storage methods.
                  "text", "value", "type", "id", "name", "input", "raw_input", "is_error",
                  "tool_use_id", "created_at", "metadata", "parallel_execution_id", "raw_content",
                  "source", "agent_response", "stop_reason", "model_response", "original_response",
                  "batch_agent_calls", "is_parallel_sub_agents", "is_parallel_tools", "sub_agent_calls"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restricted_policy(domain):
    baseline = ROOT / "baselines" / domain
    config = yaml.safe_load((baseline / "code_agent.yaml").read_text())
    frozen = {str(p.relative_to(baseline)): sha(p) for p in baseline.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts
              and str(p.relative_to(baseline)) not in {"code_agent.yaml", "systemprompt.md", "LongTermMEMORY.md"}
              and p.parent.name != "tool_descriptions"}
    descriptions = {str(p.relative_to(baseline)): yaml.safe_load(p.read_text())
                    for p in (baseline / "tool_descriptions").glob("*.yaml")}
    return {"profile": PROFILE, "domain": domain, "debugger_max_tasks": MAX_ANALYSIS_TASKS,
        "debugger_retry_attempts": 1, "reference_config": config,
        "frozen_files": frozen, "reference_descriptions": descriptions,
        "allowed": ["prompts", "tool_descriptions", "pure_state_and_planning_middleware",
                    "action_repair_or_replacement", "frozen_text_skills", "frozen_knowledge"],
        "disabled": ["new_tools", "tool_implementation_edits", "extra_llm_calls", "sub_agents",
                     "multiple_environment_calls_inside_one_tool", "cross_episode_memory_writes"]}


def check_policy(policy):
    if not isinstance(policy, dict) or policy.get("profile") != PROFILE:
        raise ValueError("unknown native capability policy")
    if policy.get("debugger_max_tasks") != MAX_ANALYSIS_TASKS:
        raise ValueError("restricted analysis cap must remain ten")


def validate_python(path, local_modules=()):
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    literal_modules = {alias.asname or alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                       for alias in node.names if alias.name == 'ast'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name not in SAFE_MODULES or alias.asname in FORBIDDEN for alias in node.names):
                raise ValueError("restricted middleware import is not allowed")
        if isinstance(node, ast.ImportFrom):
            if node.level or node.module not in SAFE_MODULES | set(PURE_NEXAU_IMPORTS) | set(local_modules):
                raise ValueError("restricted middleware import is not allowed")
            if any(alias.name == "*" or alias.name.startswith("__") or alias.name in FORBIDDEN or alias.asname in FORBIDDEN
                   or (node.module in SAFE_IMPORTS and alias.name not in SAFE_IMPORTS[node.module])
                   or (node.module in PURE_NEXAU_IMPORTS and alias.name not in PURE_NEXAU_IMPORTS[node.module]) for alias in node.names):
                raise ValueError("restricted middleware import member is not allowed")
        if isinstance(node, ast.Name) and (node.id in FORBIDDEN or node.id.startswith("__")):
            raise ValueError("restricted middleware introspection or I/O is not allowed")
        if isinstance(node, ast.Name) and node.id == "getattr":
            parent = parents.get(node)
            if not isinstance(parent, ast.Call) or parent.func is not node:
                raise ValueError("restricted getattr cannot be aliased")
        if isinstance(node, ast.Attribute) and (node.attr in BLOCKED_ATTRIBUTES or node.attr.startswith("__")):
            raise ValueError("restricted middleware execution capability is not allowed")
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in literal_modules and node.attr != 'literal_eval'):
            raise ValueError("restricted literal parser exposes literal_eval only")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in BLOCKED_ATTRIBUTES or (node.name.startswith("__") and node.name != "__init__"):
                raise ValueError("restricted middleware call wrappers are not allowed")
            if isinstance(node, ast.AsyncFunctionDef):
                raise ValueError("restricted middleware must be synchronous")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
            if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant) or node.args[1].value not in PUBLIC_GETATTR:
                raise ValueError("restricted getattr accepts public hook fields only")
    return tree


def schema_contract(value):
    """Ignore description annotations only at schema nodes, never parameter/default keys."""
    if not isinstance(value, dict):
        return value
    mappings = {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
    nodes = {"items", "additionalItems", "additionalProperties", "contains", "propertyNames",
             "if", "then", "else", "not", "unevaluatedItems", "unevaluatedProperties"}
    sequences = {"allOf", "anyOf", "oneOf", "prefixItems"}
    result = {}
    for key, item in value.items():
        if key == "description" and isinstance(item, str):
            continue
        if key in mappings and isinstance(item, dict):
            result[key] = {name: schema_contract(schema) for name, schema in item.items()}
        elif (key in sequences or key == "items") and isinstance(item, list):
            result[key] = [schema_contract(schema) for schema in item]
        elif key in nodes:
            result[key] = schema_contract(item)
        else:
            result[key] = item
    return result


def validate_workspace(workspace, policy, agent_config="code_agent.yaml"):
    check_policy(policy)
    root = Path(workspace)
    if agent_config != "code_agent.yaml" or any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("restricted workspace config or symlink is not allowed")
    for relative, expected in policy["frozen_files"].items():
        if not (root / relative).is_file() or sha(root / relative) != expected:
            raise ValueError(f"restricted immutable native file changed: {relative}")
    config = yaml.safe_load((root / agent_config).read_text())
    local_modules = {"middleware." + p.stem for p in (root / "middleware").glob("*.py")
                     if re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", p.stem)}
    fixed = {k: v for k, v in config.items() if k != "middlewares"}
    if fixed != policy["reference_config"]:
        raise ValueError("restricted tool registry, subagents, skills registry or model config changed")
    for entry in config.get("middlewares", []):
        if not isinstance(entry, dict) or set(entry) - {"import", "params"}:
            raise ValueError("restricted middleware registration is invalid")
        binding = entry.get("import", "")
        if not re.fullmatch(r"middleware\.[a-zA-Z][a-zA-Z0-9_]*:[a-zA-Z][a-zA-Z0-9_]*", binding):
            raise ValueError("restricted middleware must use a local pure Python module")
        module, symbol = binding.split(":")
        source = root / (module.replace(".", "/") + ".py")
        if not source.is_file():
            raise ValueError("registered restricted middleware is missing")
        tree = validate_python(source, local_modules)
        if not any(isinstance(n, ast.ClassDef) and n.name == symbol for n in tree.body):
            raise ValueError("registered restricted middleware class is missing")
        json.dumps(entry.get("params", {}), allow_nan=False)
    for relative, reference in policy["reference_descriptions"].items():
        current = yaml.safe_load((root / relative).read_text())
        if (not isinstance(current, dict) or not isinstance(current.get("description"), str)
                or {k: schema_contract(v) if k == "input_schema" else v
                    for k, v in current.items() if k != "description"} != {
                    k: schema_contract(v) if k == "input_schema" else v
                    for k, v in reference.items() if k != "description"}):
            raise ValueError("restricted tool descriptions must preserve names, schemas and dispatch settings")
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root)
        name = str(relative)
        if name in policy["frozen_files"] or name in policy["reference_descriptions"]:
            continue
        if name in {agent_config, "systemprompt.md", "LongTermMEMORY.md", ".gitignore", "README.md"}:
            continue
        if relative.parts[0] == "middleware" and path.suffix == ".py":
            validate_python(path, local_modules)
            continue
        if relative.parts[0] == "skills" and path.suffix == ".md":
            continue
        raise ValueError(f"restricted new component file is not allowed: {name}")
    if "{{ native_instructions }}" not in (root / "systemprompt.md").read_text():
        raise ValueError("restricted prompt must preserve native instructions")
    return {"profile": PROFILE, "candidate_valid": True,
            "immutable_native_files": len(policy["frozen_files"])}


def constraint_text(policy):
    check_policy(policy)
    return """\n\nHOST CAPABILITY CONTRACT: h2345-budget-restricted-v1 (overrides wider original permissions).
Keep native tools, bindings, names, schemas, model, tracing and budgets unchanged.
Do not modify tools/, ShortTermMEMORY.md, public_tool_schema.json or native_api.md.
Do not register new tools, skills loader tools or subagents. Do not call an LLM,
network, filesystem, sandbox, executor or RPC from candidate code. No wrap_model_call,
wrap_tool_call or stream_chunk overrides. No cross-episode persistent runtime writes.
Allowed: systemprompt.md, descriptions only in existing tool YAMLs, pure synchronous
middleware/*.py, and frozen skills/**/*.md / LongTermMEMORY.md text. In code_agent.yaml
ONLY the middlewares list may change. Use local middleware.<module>:<Class> imports.
Description text may change at tool, input schema, parameter and item schema nodes;
names, types, required fields, enum/default values and dispatch settings stay frozen.
Middleware may manage episode-local state, plan, repair/replace normal action arguments,
or condition public messages/feedback; it may not execute extra actions itself.
Public getattr access also includes AgentState.get_context_value/set_context_value
for episode-local state, ParsedResponse.tool_calls and ToolCall.parameters/tool_call_id.
Public message/block/response/hook data fields are readable, including TextBlock.text
and Role.value. Reading subagent-call data does not enable subagents or extra model calls.
Imports: ordinary pure stdlib modules; Middleware/HookResult/public hook inputs from
nexau.archs.main_sub.execution.hooks; Message/Role/text/tool blocks from
nexau.core.messages; ToolCall/ParsedResponse from nexau.archs.main_sub.execution.parse_structures;
and validated local middleware helper modules. No agents, LLM callers, dynamic imports
or execution/introspection. Existing public Message and ToolCall objects can be
modified/replaced for context conditioning and ordinary native action replacement.
Frozen skill and knowledge text are available as frozen_skills and frozen_memory Jinja
context variables; include them in the prompt if useful. Their content must evolve.
The host validates candidates before promotion and evaluation and rejects violations.
Detailed raw evidence is limited to ten selected tasks from the current iteration;
previous raw trajectories/details are absent, while previous decision summaries remain. Aggregate
evaluation still covers every training task. This is a restricted AHE port, not unrestricted
original AHE. No released Life/Meta content is supplied.\n"""


class CapabilityViolation(BaseException):
    """Fail closed even though NexAU logs and ignores ordinary middleware errors."""


class DispatchBudget:
    def __init__(self, tool_names):
        self.tool_names = set(tool_names)
        self.model_active = False
        self.model_credit = 0
        self.tool_active = None
        self.tool_credit = 0

    def consume_model(self):
        if not self.model_active or self.model_credit != 1:
            raise CapabilityViolation("extra or out-of-executor model call disabled")
        self.model_credit = 0

    def consume_tool(self, name):
        if self.tool_active != name or self.tool_credit != 1 or name not in self.tool_names:
            raise CapabilityViolation("extra or out-of-executor native action disabled")
        self.tool_credit = 0


def install_runtime_guard(agent, channel, policy):
    check_policy(policy)
    from nexau.archs.main_sub.execution.hooks import Middleware
    budget = DispatchBudget(item["name"] for item in policy["reference_config"]["tools"])
    if set(agent.tool_registry) != budget.tool_names or agent.config.sub_agents:
        raise CapabilityViolation("runtime tool registry or subagents differ from the frozen native baseline")
    original_create = agent.openai_client.chat.completions.create

    def create(*args, **kwargs):
        budget.consume_model()
        try:
            return original_create(*args, **kwargs)
        except Exception:
            # Preserve the fixed executor's normal transport retry path. A
            # completed generation consumes its credit; an unsuccessful API
            # attempt does not become a candidate-controlled auxiliary call.
            budget.model_credit = 1
            raise

    class Guard(Middleware):
        def wrap_model_call(self, params, call_next):
            if budget.model_active:
                raise CapabilityViolation("nested model call disabled")
            budget.model_active, budget.model_credit = True, 1
            try:
                return call_next(params)
            finally:
                budget.model_active, budget.model_credit = False, 0

        def wrap_tool_call(self, params, call_next):
            if budget.tool_active is not None or params.tool_name not in budget.tool_names:
                raise CapabilityViolation("nested or new tool disabled")
            budget.tool_active, budget.tool_credit = params.tool_name, 1
            try:
                return call_next(params)
            finally:
                budget.tool_active, budget.tool_credit = None, 0

    agent.openai_client.chat.completions.create = create
    channel.dispatch_budget = budget
    agent.executor.middleware_manager.middlewares.insert(0, Guard())
    return budget
