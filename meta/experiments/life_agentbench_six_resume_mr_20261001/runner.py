"""Prepare inherited state, smoke current train hooks, or run one additional round."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

import driver
import inheritance
import protocol
import status


def prepare():
    protocol.validate_protocol()
    for domain in protocol.DOMAINS:
        folder = protocol.HERE / domain
        folder.mkdir(parents=True, exist_ok=True)
        protocol.bind_fingerprint(folder / "run_config.json", driver.run_fingerprint(domain))
        path = folder / "state.json"
        if not path.exists():
            protocol.atomic_json(path, inheritance.initialize(domain, folder))
        state = json.loads(path.read_text())
        driver.verify_active(folder, state)
        driver.verify_training_history(folder, state)
        print(json.dumps({"domain": domain, "inherited_rounds": 5,
                          "round": state["round"], "batch": state["batch"],
                          "snapshot": state["active_snapshot_sha256"]}), flush=True)


def smoke():
    prepare()
    for domain in protocol.DOMAINS:
        folder = protocol.HERE / domain
        state = json.loads((folder / "state.json").read_text())
        manifest = driver.write_active_chain(folder, state, "smoke_inherited")
        python = {"alfworld": "alfworld-venv", "dbbench": "core", "webshop": "webshop-venv"}[domain]
        command = [str(protocol.ROOT / "AgentBench/.native" / python / "bin/python"), "-u",
                   str(protocol.HERE / "agentbench_evaluate.py"), domain, "train", "smoke_inherited",
                   str(manifest), "--batch", str(state["batch"]), "--task-limit", "1"]
        env = dict(os.environ)
        if domain == "webshop":
            java = protocol.ROOT / "AgentBench/.native/sysroot/usr/lib/jvm/java-11-openjdk-amd64"
            env.update(JAVA_HOME=str(java), JVM_PATH=str(java / "lib/server/libjvm.so"),
                       PATH=str(java / "bin") + ":" + env.get("PATH", ""), _JAVA_OPTIONS="-Xmx4g",
                       PYTHONPATH=str(protocol.ROOT / "AgentBench/.native/webshop-src") + ":" + str(protocol.ROOT / "AgentBench"))
        log = folder / "smoke_inherited.log"
        with log.open("a") as stream:
            subprocess.run(command, cwd=protocol.ROOT, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True, timeout=1200)
        summary = json.loads((folder / "evals/smoke_inherited/summary.json").read_text())
        if summary["infra_errors"]:
            raise RuntimeError(f"{domain} smoke has an infrastructure error; inspect {log}")
        print(json.dumps({"smoke": domain, "n": summary["n"], "successes": summary["successes"],
                          "infra_errors": summary["infra_errors"]}), flush=True)


def train():
    prepare()
    for domain in protocol.DOMAINS:
        folder = protocol.HERE / domain
        with (folder / ".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock.write(str(os.getpid()) + "\n")
            lock.flush()
            print(json.dumps({"event": "round_six_start", "domain": domain}), flush=True)
            state = driver.run_domain(domain)
            print(json.dumps({"event": "round_six_complete", "domain": domain,
                              "decision": state["rounds"][-1]}), flush=True)
    path, seal = status.seal_training()
    print(json.dumps({"event": "six_round_training_sealed", "path": str(path),
                      "seal": seal, "final_test_started": False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "smoke", "train"))
    args = parser.parse_args()
    with (protocol.HERE / ".runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.write(str(os.getpid()) + "\n")
        lock.flush()
        {"prepare": prepare, "smoke": smoke, "train": train}[args.command]()


if __name__ == "__main__":
    main()
