"""Deterministic contract and context-equivalence checks for all seven domains."""
from dataclasses import asdict
from pathlib import Path
import copy
import hashlib
import json
import re
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AB = ROOT / "AgentBench"
TAU = ROOT / "TauBench"
CURRENT_SOURCE = ROOT / "meta/experiments/current_agentbench_source_20260914"
sys.path.insert(0, str(AB))

from src.server.harness import (
    ALFWorldHarness,
    DBBenchHarness,
    OSInteractionHarness,
    WebShopHarness,
)
from src.server.harness.dbbench import DBBenchHarnessRuntime, DBBenchHarnessConfig
from src.server.harness.os_interaction import OSHarnessRuntime, OSHarnessConfig
from src.server.harness.webshop import WebShopHarnessRuntime, WebShopHarnessConfig


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def skeleton(tools):
    clean = copy.deepcopy(tools)
    for tool in clean:
        tool["function"].pop("description", None)
    return clean


agent_classes = {
    "alfworld": ALFWorldHarness,
    "dbbench": DBBenchHarness,
    "webshop": WebShopHarness,
    "os_interaction": OSInteractionHarness,
}
for name, cls in agent_classes.items():
    harness = cls()
    assert all(callable(getattr(harness, layer, None)) for layer in ("h2", "h3", "h4", "h5")), name


db_profiles = {}
for config_path in (AB / "configs/tasks/dbbench.yaml", AB / ".native/configs/dbbench.yaml"):
    match = re.search(r"(?ms)^dbbench-std:\n(?P<profile>.*?)(?=^\S|\Z)", config_path.read_text())
    assert match, config_path
    profile = match.group("profile")
    flags = {
        key: bool(re.search(rf"(?m)^\s+{key}:\s*true(?:\s|$)", profile))
        for key in ("enabled", "h2", "h3", "h4", "h5")
    }
    assert all(flags.values()), config_path
    db_profiles[str(config_path.relative_to(AB))] = flags


source_manifest = json.loads((CURRENT_SOURCE / "manifest.json").read_text())
assert source_manifest["current_task_snapshot"] is True
for relative_path, expected_hash in source_manifest["hashes"].items():
    assert digest(CURRENT_SOURCE / "task_snapshot" / relative_path) == expected_hash, relative_path
for domain in ("alfworld", "dbbench"):
    assert digest(CURRENT_SOURCE / domain / "baseline.py") == source_manifest["starting_candidate"][domain]


def hint_texts(runtime):
    return [item["text"] for item in runtime.cold_start_skill_hints()]


db_tools = [
    {"type": "function", "function": {"name": "execute_sql", "description": "execute", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "commit_final_answer", "description": "commit", "parameters": {"type": "object", "properties": {"answers": {"type": "array"}}, "required": ["answers"]}}},
]
db_rows = [json.loads(line) for line in (AB / "data/dbbench/standard.jsonl").read_text().splitlines()]
db_h5_cases = 0
for entry in db_rows:
    config = DBBenchHarnessConfig(enabled=True, h2_enabled=True, h3_enabled=True, h4_enabled=True, h5_enabled=True, h5_top_k=1)
    legacy = DBBenchHarnessRuntime(config)
    legacy.init_task(entry)
    current = DBBenchHarness(config)
    context = {key: copy.deepcopy(entry[key]) for key in ("description", "type", "table", "evidence", "add_description") if key in entry}
    current.bind_task_context(context)
    old_ctx = asdict(legacy.task_ctx)
    new_ctx = asdict(current.runtime.task_ctx)
    # expected_insert_cols was previously derived from reference SQL and is
    # intentionally removed from the legal candidate context.
    old_ctx.pop("expected_insert_cols", None)
    new_ctx.pop("expected_insert_cols", None)
    assert old_ctx == new_ctx
    assert legacy.schema_map == current.runtime.schema_map
    assert current.h5([]) == current._one_hint(*hint_texts(legacy))
    db_h5_cases += 1
assert skeleton(db_tools) == skeleton(DBBenchHarness().h3(db_tools))
try:
    DBBenchHarness().bind_task_context({"sql": {}, "description": "x", "type": ["SELECT"], "table": {}})
    raise AssertionError("oracle DB context was accepted")
except ValueError:
    pass


os_config = OSHarnessConfig(enabled=True, h2_enabled=True, h3_enabled=True, h4_enabled=True, h5_enabled=True, h5_top_k=1, h5_score_threshold=7.5)
os_rows = json.loads((AB / "data/os_interaction/train_0317/training.json").read_text())
os_h5_cases = 0
for entry in os_rows:
    legacy = OSHarnessRuntime(os_config)
    legacy.init_task(entry["description"])
    current = OSInteractionHarness(os_config)
    current.bind_task_context({"description": entry["description"]})
    assert current.runtime.task_ctx == legacy.task_ctx
    assert current.h5([]) == current._one_hint(*hint_texts(legacy))
    os_h5_cases += 1


web_config = WebShopHarnessConfig(enabled=True, h2_enabled=True, h3_enabled=True, h4_enabled=True, h5_enabled=True, h5_top_k=1, h5_score_threshold=3.0)
web_instructions = [
    "Buy a red medium cotton shirt under $30.",
    "Find black size 9 running shoes below $80.",
    "Buy a 32GB USB drive with a metal case.",
    "Find lavender shampoo, pack of 2, under $25.",
    "Buy an ivory queen cotton sheet set.",
    "Find organic almond snacks, 12 ounce package.",
]
for instruction in web_instructions:
    legacy = WebShopHarnessRuntime(web_config)
    legacy.init_task(instruction)
    current = WebShopHarness(web_config)
    current.bind_task_context({"instruction": instruction})
    assert current.runtime.requirements == legacy.requirements
    assert current.h5([]) == current._one_hint(*hint_texts(legacy))


tau_classes = {
    "airline": "H3H4HarnessedAirlineTools",
    "retail": "H3H4HarnessedRetailTools",
    "telecom": "H3H4HarnessedTelecomTools",
}
for domain, class_name in tau_classes.items():
    source = (TAU / f"src/tau2/harness/{domain}.py").read_text()
    assert f"class {class_name}" in source
loader = (TAU / "scripts/eval_harness.py").read_text()
assert "--harness-plugin" in loader and "register()" in loader


tau_report = json.loads((HERE / "tau_verification.json").read_text())
assert tau_report["selection_cases"] == 24
assert tau_report["noop_plugin_state_equivalence"] == "pass"

alf_replay = json.loads((HERE / "replay_alfworld.json").read_text())
db_replay = json.loads((HERE / "replay_dbbench.json").read_text())
for replay in (alf_replay, db_replay):
    assert not replay["outcome_mismatches"]
    assert not replay["status_mismatches"]
    assert not replay["turn_mismatches"]
    assert replay["successes"] == replay["expected_successes"]
    assert replay["model_input_differences"] == 0
    assert replay["new_model_calls"] == 0
assert not db_replay["replay_errors"]
assert db_replay["tool_input_differences"] == 0

http_smoke = {item["name"]: item for item in json.loads((HERE / "http_smoke.json").read_text())}
assert http_smoke["dbbench-std"]["indices"] == 300
assert http_smoke["dbbench-std"]["statuses"][-1] == "completed"
assert http_smoke["dbbench-std"]["scripted_tool_calls"] == 2
assert http_smoke["alfworld-std"]["indices"] == 109
assert http_smoke["alfworld-std"]["statuses"][-1] == "running"
assert http_smoke["alfworld-std"]["scripted_tool_calls"] == 1


report = {
    "domains": {
        "airline": {"format": "H2/H3/H4/H5 registries + register() plugin", "selection_cases": 8, "train_tasks": tau_report["declared_train_tasks"]["airline"]},
        "retail": {"format": "H2/H3/H4/H5 registries + register() plugin", "selection_cases": 8, "train_tasks": tau_report["declared_train_tasks"]["retail"]},
        "telecom": {"format": "H2/H3/H4/H5 registries + register() plugin", "selection_cases": 8, "train_tasks": tau_report["declared_train_tasks"]["telecom"]},
        "alfworld": {"format": "Harness.h2/h3/h4/h5", "frozen_replay": f"{alf_replay['successes']}/{alf_replay['n']}", "semantic_model_input_differences": alf_replay["model_input_differences"], "raw_h4_boundary_differences": alf_replay["raw_model_input_differences"]},
        "dbbench": {"format": "Harness.h2/h3/h4/h5 + bound public context", "context_cases": len(db_rows), "frozen_replay": f"{db_replay['successes']}/{db_replay['n']}", "semantic_model_input_differences": db_replay["model_input_differences"], "tool_input_differences": db_replay["tool_input_differences"]},
        "webshop": {"format": "Harness.h2/h3/h4/h5", "context_cases": len(web_instructions)},
        "os_interaction": {"format": "Harness.h2/h3/h4/h5", "context_cases": len(os_rows)},
    },
    "agentbench_hook_contract": "pass",
    "db_context_equivalence": f"pass ({len(db_rows)} standard tasks)",
    "db_h5_equivalence": f"pass ({db_h5_cases} standard tasks)",
    "db_oracle_rejection": "pass",
    "os_context_equivalence": f"pass ({len(os_rows)} train tasks)",
    "os_h5_equivalence": f"pass ({os_h5_cases} train tasks)",
    "webshop_context_equivalence": f"pass ({len(web_instructions)} representative task types)",
    "webshop_h5_equivalence": f"pass ({len(web_instructions)} representative task types)",
    "tau_plugin_contract": "pass",
    "native_http_smoke": {
        "dbbench-std": "pass (300 indices; execute + commit completed)",
        "alfworld-std": "pass (109 indices; environment action executed and session cancelled)",
    },
    "db_standard_profiles": db_profiles,
    "current_iteration_source": {
        "manifest": str((CURRENT_SOURCE / "manifest.json").relative_to(ROOT)),
        "snapshot_files_verified": len(source_manifest["hashes"]),
        "starting_candidates_verified": ["alfworld", "dbbench"],
    },
    "hashes": {
        str(path.relative_to(ROOT)): digest(path)
        for path in [
            AB / "src/server/harness/alfworld.py",
            AB / "src/server/harness/dbbench.py",
            AB / "src/server/harness/webshop.py",
            AB / "src/server/harness/os_interaction.py",
            AB / "src/server/harness/session.py",
            AB / "src/server/tasks/alfworld/task.py",
            AB / "src/server/tasks/dbbench/task.py",
            AB / "src/server/tasks/webshop/task.py",
            AB / "src/server/tasks/os_interaction/task.py",
            AB / "configs/tasks/dbbench.yaml",
            AB / ".native/configs/dbbench.yaml",
            CURRENT_SOURCE / "manifest.json",
            ROOT / "meta/agentbench_evaluate.py",
            ROOT / "meta/agentbench_preflight.py",
            ROOT / "meta/agentbench_loop.py",
        ]
    },
}
(HERE / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
