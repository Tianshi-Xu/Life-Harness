"""Dynamic no-model verification of the three TauBench domain adapters."""
from dataclasses import asdict, is_dataclass
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TAU = ROOT / "TauBench"
sys.path.insert(0, str(TAU / "src"))

from tau2.domains.airline.environment import get_environment as airline_environment
from tau2.domains.airline.environment import get_tasks as airline_tasks
from tau2.domains.retail.environment import get_environment as retail_environment
from tau2.domains.retail.environment import get_tasks as retail_tasks
from tau2.domains.telecom.environment import get_environment as telecom_environment
from tau2.domains.telecom.environment import get_tasks as telecom_tasks
from tau2.harness import airline, retail, telecom
from tau2.harness.skills import DOMAIN_SKILLS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def public(value):
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return repr(value)


def snapshot():
    classes = {
        "airline": airline.H3H4HarnessedAirlineTools,
        "retail": retail.H3H4HarnessedRetailTools,
        "telecom": telecom.H3H4HarnessedTelecomTools,
    }
    return {
        "rules": {
            domain: {
                name: [type(rule).__name__ for rule in rules]
                for name, rules in cls.harness_rules.items()
            }
            for domain, cls in classes.items()
        },
        "annotators": {
            domain: {
                name: [type(annotator).__name__ for annotator in annotators]
                for name, annotators in cls.harness_annotators.items()
            }
            for domain, cls in classes.items()
        },
        "skills": {
            domain: [public(skill) for skill in DOMAIN_SKILLS[domain]]
            for domain in classes
        },
    }


expected_classes = {
    "airline": {
        (False, False, False): "AirlineTools",
        (True, False, False): "HarnessedAirlineTools",
        (False, True, False): "H3AirlineTools",
        (False, False, True): "H4AirlineTools",
        (True, True, False): "H3HarnessedAirlineTools",
        (True, False, True): "H4HarnessedAirlineTools",
        # Airline's H4-only path subsumes H2 in the current source.
        (False, True, True): "H4AirlineTools",
        (True, True, True): "H3H4HarnessedAirlineTools",
    },
    "retail": {
        (False, False, False): "RetailTools",
        (True, False, False): "HarnessedRetailTools",
        (False, True, False): "H3RetailTools",
        (False, False, True): "H4RetailTools",
        (True, True, False): "H3HarnessedRetailTools",
        (True, False, True): "H4HarnessedRetailTools",
        (False, True, True): "H3H4RetailTools",
        (True, True, True): "H3H4HarnessedRetailTools",
    },
    "telecom": {
        (False, False, False): "TelecomTools",
        (True, False, False): "HarnessedTelecomTools",
        (False, True, False): "H3TelecomTools",
        (False, False, True): "H4TelecomTools",
        (True, True, False): "H3HarnessedTelecomTools",
        (True, False, True): "H4HarnessedTelecomTools",
        (False, True, True): "H3H4TelecomTools",
        (True, True, True): "H3H4HarnessedTelecomTools",
    },
}
factories = {
    "airline": airline_environment,
    "retail": retail_environment,
    "telecom": telecom_environment,
}
task_factories = {
    "airline": airline_tasks,
    "retail": retail_tasks,
    "telecom": telecom_tasks,
}

selections = {}
for domain, factory in factories.items():
    selections[domain] = {}
    for switches, expected in expected_classes[domain].items():
        h2, h3, h4 = switches
        env = factory(harness_enabled=h2, harness_h3=h3, harness_h4=h4)
        actual = type(env.tools).__name__
        assert actual == expected, (domain, switches, actual, expected)
        selections[domain]["".join(str(int(flag)) for flag in switches)] = actual

before = snapshot()
spec = importlib.util.spec_from_file_location("eval_harness", TAU / "scripts/eval_harness.py")
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)
loader._load_harness_plugin(str(HERE / "noop_tau_plugin.py"))
after = snapshot()
assert before == after

report = {
    "format": "H2/H3/H4 registries + H5 skill registry + register() plugin",
    "selection_cases": 24,
    "selected_classes": selections,
    "noop_plugin_state_equivalence": "pass",
    "h5_skill_counts": {domain: len(DOMAIN_SKILLS[domain]) for domain in factories},
    "declared_train_tasks": {
        domain: len(factory("train")) for domain, factory in task_factories.items()
    },
    "source_hashes": {
        str(path.relative_to(ROOT)): digest(path)
        for path in [
            TAU / "src/tau2/harness/base.py",
            TAU / "src/tau2/harness/airline.py",
            TAU / "src/tau2/harness/retail.py",
            TAU / "src/tau2/harness/telecom.py",
            TAU / "src/tau2/harness/skills.py",
            TAU / "scripts/eval_harness.py",
        ]
    },
}
(HERE / "tau_verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
