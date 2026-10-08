"""Hash-bound narrow runtime amendment: no H4/H5 word cap at the Task boundary."""
import json
import subprocess
from pathlib import Path

import agentbench_evaluate
import driver
import protocol

SOURCES = ("runtime_support.py", "corrected_evaluate.py", "corrected_runner.py",
           "corrected_final.py", "corrected_workflow.py", "repair_runtime.py")
ORIGINAL_RUN = subprocess.run
ORIGINAL_VALIDATE = agentbench_evaluate.validate_cached_summary


def bind_amendment():
    value = {"reason": "frozen Task session still enforced a 120-word cap contrary to the declared no-word-limit protocol",
             "scope": "only FourHookSession._hints word-count validation; preserve one optional string, trace, messages and every other hook behavior",
             "source_sha256": {name: protocol.digest(protocol.HERE / name) for name in SOURCES},
             "frozen_core_sha256": {name: protocol.digest(protocol.HERE / name) for name in
                 ("protocol.py", "protocol.json", "driver.py", "agentbench_chain.py", "agentbench_evaluate.py")},
             "unaffected_evidence_sha256": protocol.digest(protocol.HERE / "unaffected_evidence.json"),
             "invalid_evaluation_archive": "runtime_cap_archive/dbbench/candidate_09",
             "rerun": "DBBench round 9, identical cached proposal and frozen 50-task batch; no new proposal"}
    protocol.bind_fingerprint(protocol.HERE / "runtime_amendment.json", value)
    return value


def verify_evaluation(folder, summary):
    ORIGINAL_VALIDATE(folder, summary)
    folder = Path(folder).resolve()
    if protocol.HERE not in folder.parents:
        return
    relative = str(folder.relative_to(protocol.HERE))
    unaffected = json.loads((protocol.HERE / "unaffected_evidence.json").read_text())
    if relative in unaffected:
        for filename, expected in unaffected[relative].items():
            if protocol.digest(folder / filename) != expected:
                raise RuntimeError("unaffected pre-amendment evaluation changed")
        return
    marker = json.loads((folder / "runtime_amendment.json").read_text())
    expected = {"config_sha256": summary["config_sha256"],
                "runtime_amendment_sha256": protocol.digest(protocol.HERE / "runtime_amendment.json")}
    if marker != expected:
        raise RuntimeError("evaluation runtime amendment mismatch")


def routed_run(command, *args, **kwargs):
    if isinstance(command, (list, tuple)) and len(command) > 2:
        expected = str(protocol.HERE / "agentbench_evaluate.py")
        if expected in command:
            command = list(command)
            command[command.index(expected)] = str(protocol.HERE / "corrected_evaluate.py")
    return ORIGINAL_RUN(command, *args, **kwargs)


def install():
    bind_amendment()
    subprocess.run = routed_run
    agentbench_evaluate.validate_cached_summary = verify_evaluation
