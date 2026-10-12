"""Current Life WebShop experiment; reuse native tasks and immutable bundle staging."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from evolution import harness_hooks as hooks
from evolution.life_evolve import profile
from evolution.life_evolve.context import ContextLimit, OUTPUT_LIMIT, verify_usage
from evolution.life_evolve.tau_run import COMPOSITION, composition_module
from evolution.tau_comparison.common import ROOT, TAU, AHE_NATIVE, child_env, digest, load_module, read, save

HISTORICAL = ROOT / "evolution/reproduction/reference"
_worker = None
_interface = None


def native_module():
    sys.path.insert(0, str(COMPOSITION))
    return load_module("life_current_webshop_native", COMPOSITION / "agentbench_evaluate.py")


def native_environment():
    # Same Java/search and numerical-thread settings as the proven native evaluator.
    java = ROOT / "AgentBench/.native/sysroot/usr/lib/jvm/java-11-openjdk-amd64"
    env = child_env(PYTHONPATH=str(ROOT))
    env.update(JAVA_HOME=str(java), JVM_PATH=str(java / "lib/server/libjvm.so"),
        PATH=str(java / "bin") + ":" + env.get("PATH", ""), _JAVA_OPTIONS="-Xmx4g",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    return env


def initialize_worker(split, manifest, domain="webshop"):
    global _worker, _interface
    _worker = native_module()

    def inject(domain, task_module, path):
        global _interface
        _interface = hooks.install_agentbench(task_module, domain, path)

    _worker.inject_task_module = inject
    _worker.initialize(domain, split, manifest)


def episode_worker(index):
    return {**_worker.episode(index), "interface": _interface}


def optimizer_usage(summary, experiment, number=None, domain="webshop"):
    groups = [g for g in summary["groups"].values() if g["experiment"] == experiment
              and (domain is None or g["domain"] == domain) and g["role"] == "life_optimizer"
              and (number is None or g["round"] == number)]
    fields = ("attempts", "known_usage_calls", "unknown_usage_calls", "pending_calls",
              "input_tokens", "output_tokens", "total_tokens", "rate_limit_rejections_without_usage")
    result = {k: sum(g.get(k, 0) for g in groups) for k in fields}
    result["accounting_complete"] = result["unknown_usage_calls"] == result["pending_calls"] == 0
    result["inference_accounting_complete"] = (result["pending_calls"] == 0
        and result["unknown_usage_calls"] == result["rate_limit_rejections_without_usage"])
    return result


def initialize(root, domain="webshop"):
    if root.exists():
        raise FileExistsError("new run required; no blind replay of paid sessions")
    original = read(HISTORICAL / "manifest.json")
    spec = original["domains"][domain]
    from evolution.reproduction.life_runtime.protocol import task_ids
    batches = [task_ids(spec, n, "train") for n in range(6)]
    tests = task_ids(spec, 0, "test")
    if (not tests or len(set(tests)) != len(tests)
            or any(len(b) != 50 or len(set(b)) != 50 or set(b) & set(tests) for b in batches)):
        raise ValueError("frozen unique TRAIN/TEST schedule mismatch")
    frozen = {str(ROOT / name): sha for name, sha in profile.source_hashes().items()}
    for name, sha in original["snapshot_hashes"].items():
        frozen[str(HISTORICAL / "task_snapshot" / name)] = sha
    for name, sha in original["data_hashes"].items():
        frozen[str(ROOT / "AgentBench" / name)] = sha
    files = [Path(__file__), HISTORICAL / "manifest.json", ROOT / "evolution/tau_comparison/common.py"]
    files += list(COMPOSITION.glob("*.py"))
    files += [AHE_NATIVE / "ahe_native" / name for name in
              ("api_broker.py", "metered_broker.py", "token_accounting.py", "connection.py")]
    frozen.update({str(p.resolve()): digest(p) for p in files})
    protocol = {"name": root.name, "domain": domain, "rounds": 6,
        "batches": batches, "test_ids": tests,
        "initialization": "empty H2-H5; no imported historical strategies",
        "solver": original["agent"], "solver_total_context": 32768, "solver_workers": 50,
        "optimizer": {"model": "deepseek-v4-flash", "max_output": OUTPUT_LIMIT,
            "interaction_limit": None, "token_limit": profile.MAX_TOTAL_TOKENS, "max_repairs": 3},
        "global_mr_concurrency": 8,
        "selection": "strict improvement on same complete 50-task TRAIN batch",
        "test": "fixed TRAIN-selected incumbent; full TEST after six rounds and seal"}
    root.mkdir(parents=True)
    save(root / "protocol.json", protocol)
    frozen[str(root / "protocol.json")] = digest(root / "protocol.json")
    save(root / "frozen_sources.json", frozen)
    save(root / domain / "state.json", {"domain": domain, "round": 0, "rounds": [],
        "active_snapshot": [], "active_snapshot_sha256": hooks.object_digest([]),
        "accepted_bundle_ids": [], "operation_log": []})
    save(root / "status.json", {"phase": "prepared", "completed_rounds": 0})


def evaluation_config(root, split, name, manifest, ids):
    protocol = read(root / "protocol.json")
    _, records = hooks.manifest_records(manifest)
    if split == "test":
        state = read(root / protocol.get("domain", "webshop") / "state.json")
        seal = read(root / "training_seal.json")
        if (name != "final_test" or ids != protocol["test_ids"] or state["round"] != 6
                or len(state["rounds"]) != 6 or digest(root / protocol.get("domain", "webshop") / "state.json") != seal["state_sha256"]
                or records != seal["snapshot"]):
            raise ValueError("TEST requires the sealed six-round TRAIN-selected snapshot")
    elif split != "train" or not ids or not set(ids) <= set(sum(protocol["batches"], [])):
        raise ValueError("invalid TRAIN task list")
    body = {"evaluator_schema_version": 1, "domain": protocol.get("domain", "webshop"), "split": split, "phase": name,
        "ids": ids, "snapshot_sha256": hooks.object_digest(records), "manifest_sha256": digest(manifest),
        "protocol_sha256": digest(root / "protocol.json"), "source_sha256": digest(__file__),
        "interface_version": hooks.INTERFACE_VERSION, "concurrency": protocol["solver_workers"],
        "max_context_tokens": 32768, "max_output_tokens": 4096}
    return {**body, "config_sha256": hooks.object_digest(body)}


def native_evaluate(root, split, name, manifest, ids):
    config = evaluation_config(root, split, name, manifest, ids)
    native = native_module()
    directory = root / config["domain"] / "evals" / name
    if directory.exists():
        raise RuntimeError("incomplete evaluation must be audited, not silently rerun")
    save(directory / "config.json", config)
    with ProcessPoolExecutor(max_workers=config["concurrency"], mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker, initargs=(split, str(manifest), config["domain"])) as pool:
        futures = {pool.submit(episode_worker, index): index for index in ids}
        for future in as_completed(futures):
            row = future.result()
            if row["index"] != futures[future] or row["interface"]["interface_version"] != hooks.INTERFACE_VERSION:
                raise ValueError("native worker index/interface mismatch")
            row.update(episode_schema_version=1, config_sha256=config["config_sha256"])
            save(directory / f"episode_{row['index']}.json", native._jsonable(row))
            print(json.dumps({k: row[k] for k in ("index", "success", "status")}), flush=True)
    rows = [read(directory / f"episode_{i}.json") for i in ids]
    summary = {**config, "n": len(rows), "successes": sum(r["success"] for r in rows),
        "accuracy": sum(r["success"] for r in rows) / len(rows),
        "infra_errors": sum(r["infrastructure_error"] for r in rows),
        "tokens": sum(r["tokens"] for r in rows), "turns": sum(r["turns"] for r in rows),
        "per_task": {str(r["index"]): r["success"] for r in rows}}
    save(directory / "summary.json", summary)
    native.validate_cached_summary(directory, summary)


class Run:
    def __init__(self, root):
        self.root = root
        self.domain = read(root / "protocol.json").get("domain", "webshop")
        self.work = root / self.domain
        self.protocol, self.frozen = read(root / "protocol.json"), read(root / "frozen_sources.json")
        self.state, self.compose = read(self.work / "state.json"), composition_module()
        self.broker = None

    def verify(self):
        if any(not Path(p).is_file() or digest(p) != sha for p, sha in self.frozen.items()):
            raise RuntimeError("frozen source/data changed")
        self.compose.verify_active(self.work, self.state)

    def progress(self, phase, **extra):
        value = {"phase": phase, "pid": os.getpid(), "utc_unix": time.time(),
                 "completed_rounds": self.state["round"], **extra}
        save(self.root / "status.json", value)
        print(json.dumps(value), flush=True)

    def usage(self, number=None):
        summary = self.broker.ledger.summary()
        save(self.root / "OPTIMIZER_TOKEN_USAGE.json", optimizer_usage(summary, self.root.name, domain=self.domain))
        return optimizer_usage(summary, self.root.name, number, domain=self.domain)

    def metrics(self):
        while not self.stopped.wait(30):
            save(self.root / "flash_metrics.json", {"gate": self.gate.snapshot(),
                "usage": self.broker.ledger.summary()["overall"], "utc_unix": time.time()})

    def evaluate(self, split, name, manifest, ids):
        self.verify()
        directory = self.work / "evals" / name
        config = evaluation_config(self.root, split, name, manifest, ids)
        if (directory / "summary.json").exists():
            summary = read(directory / "summary.json")
            if any(summary.get(k) != v for k, v in config.items()):
                raise ValueError("cached evaluation identity changed")
            native_module().validate_cached_summary(directory, summary)
            return summary, directory
        if directory.exists():
            raise RuntimeError("incomplete evaluation; no automatic replay")
        self.progress("evaluating", evaluation=name, split=split, task_count=len(ids))
        logs = self.work / "logs"
        logs.mkdir(exist_ok=True)
        with (logs / f"{name}.log").open("a") as log:
            result = subprocess.run([str(ROOT / {"alfworld": "AgentBench/.native/alfworld-venv/bin/python", "dbbench": "AgentBench/.native/core/bin/python", "webshop": "AgentBench/.native/webshop-venv/bin/python"}[self.domain]), "-u", "-m",
                "evolution.life_agentbench_run", "--name", self.root.name, "--domain", self.domain, "--evaluate", split, "--phase", name,
                "--manifest", str(manifest), "--ids", json.dumps(ids)], cwd=ROOT,
                env=native_environment(), stdout=log, stderr=subprocess.STDOUT, timeout=21600)
        if result.returncode:
            raise RuntimeError("native evaluation failed: " + str(logs / f"{name}.log"))
        summary = read(directory / "summary.json")
        native_module().validate_cached_summary(directory, summary)
        if summary["n"] != len(ids) or set(summary["per_task"]) != {str(i) for i in ids}:
            raise ValueError("incomplete fixed task denominator")
        return summary, directory

    def completion(self, number):
        from ahe_native.token_accounting import scoped_url
        endpoint = scoped_url(self.broker.url, experiment=self.root.name, domain=self.domain,
                              round=number, role="life_optimizer")

        def call(messages, tools):
            self.progress("evolving", round=number)
            request = urllib.request.Request(endpoint + "/chat/completions", method="POST",
                data=json.dumps({"model": "deepseek-v4-flash", "messages": messages, "tools": tools,
                    "tool_choice": "auto", "max_tokens": OUTPUT_LIMIT, "stream": False}).encode(),
                headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"})
            try:
                with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=2100) as response:
                    value = json.load(response)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")
                if exc.code == 400 and any(t in body.lower() for t in ("context", "maximum sequence", "too long")):
                    raise ContextLimit("provider context exhausted") from exc
                raise RuntimeError("Flash HTTP failure " + str(exc.code)) from exc
            verify_usage(value)
            if not value.get("choices") or value.get("error"):
                raise RuntimeError("Flash response missing assistant choice")
            return value
        return call

    def round(self, number):
        ids = self.protocol["batches"][number - 1]
        manifest = self.compose.write_active_chain(self.work, self.state, f"incumbent_{number:02d}")
        incumbent, evidence = self.evaluate("train", f"incumbent_{number:02d}", manifest, ids)
        workspace, output = self.root / f"optimizer_inputs/round_{number:02d}", self.work / f"proposals/round_{number:02d}"
        profile.prepare(self.work, evidence, workspace, seed=number)
        bundle_path = profile.propose(workspace, output, completion=self.completion(number),
            checkpoint_dir=self.work / f"optimizer_sessions/round_{number:02d}")
        validation, usage = read(output / "validation.json"), self.usage(number)
        if (not usage["inference_accounting_complete"] or not usage["total_tokens"]
                or usage["total_tokens"] != validation["usage"].get("total_tokens", 0)):
            raise RuntimeError("optimizer ledger/checkpoint mismatch")
        bundle, _, staged = self.compose.stage_bundle(self.work, number, bundle_path, self.state)
        candidate, _ = self.evaluate("train", f"candidate_{number:02d}", staged, ids)
        accepted = profile.strict_improvement(candidate, incumbent, expected_task_ids=[str(i) for i in ids])
        parent = self.state["active_snapshot_sha256"]
        operations = [{k: v for k, v in op.items() if k != "artifact_path"} for op in bundle["operations"]]
        if accepted:
            active = self.compose.commit_bundle(self.work, bundle, self.state)
            self.state["active_snapshot"], self.state["active_snapshot_sha256"] = active, hooks.object_digest(active)
            self.state["accepted_bundle_ids"].append(bundle["bundle_id"])
            self.state["operation_log"].append({"bundle_id": bundle["bundle_id"], "operations": operations,
                "parent_snapshot_sha256": parent, "committed_snapshot_sha256": hooks.object_digest(active)})
        row = {"round": number, "train_batch": number - 1, "incumbent_score": incumbent["accuracy"],
            "candidate_score": candidate["accuracy"], "retained_score": (candidate if accepted else incumbent)["accuracy"],
            "incumbent_successes": incumbent["successes"], "candidate_successes": candidate["successes"],
            "incumbent_infra_errors": incumbent["infra_errors"], "candidate_infra_errors": candidate["infra_errors"],
            "accepted": accepted, "outcome": "accepted" if accepted else "rejected_not_strict_improvement",
            "bundle_id": bundle["bundle_id"], "bundle_hypothesis": bundle["hypothesis"], "operations": operations,
            "incumbent_eval": str(evidence), "candidate_eval": str(self.work / f"evals/candidate_{number:02d}"),
            "active_snapshot_sha256": self.state["active_snapshot_sha256"], "optimizer_usage": usage,
            "proposer_requests": validation["proposer_requests"], "repairs_granted": validation["repairs_granted"],
            "meta_reference": self.protocol["meta_reference"][number - 1], "ahe_reference": self.protocol["ahe_reference"][number - 1]}
        self.state["rounds"].append(row)
        self.state["round"] = number
        save(self.work / "state.json", self.state)
        self.verify()
        save(self.root / "TRAINING_RESULTS.json", {"domain": self.domain, "rounds": self.state["rounds"]})
        self.progress("round_complete", **{k: row[k] for k in ("round", "candidate_score", "retained_score", "accepted")})

    def run(self):
        sys.path.insert(0, str(AHE_NATIVE))
        from ahe_native.api_broker import RequestGate
        from ahe_native.connection import credentials
        from ahe_native.metered_broker import Broker
        with (self.root / ".runner.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if read(self.root / "status.json")["phase"] != "prepared" or self.state["round"]:
                raise RuntimeError("started experiment requires audited continuation")
            self.verify()
            key, endpoint = credentials()
            self.gate = RequestGate(8, min_interval=1.25)
            self.broker = Broker(key, endpoint, self.gate, rate_limit_retries=3,
                ledger_path=self.root / "ledgers/mr_requests.jsonl", ledger_summary_path=self.root / "ledgers/MR_TOKEN_USAGE.json")
            del key
            self.stopped = threading.Event()
            threading.Thread(target=self.metrics, daemon=True).start()
            try:
                empty = self.compose.write_active_chain(self.work, self.state, "setup_empty")
                smoke, _ = self.evaluate("train", "setup_smoke", empty, self.protocol["batches"][0][:1])
                if smoke["infra_errors"]:
                    raise RuntimeError("native integration smoke failed; no Evolve calls")
                for number in range(1, 7):
                    self.round(number)
                save(self.root / "training_seal.json", {"state_sha256": digest(self.work / "state.json"),
                    "optimizer_usage": self.usage(), "snapshot": self.state["active_snapshot"]})
                self.broker.blocked_roles.add("life_optimizer")
                final = self.compose.write_active_chain(self.work, self.state, "final")
                result, _ = self.evaluate("test", "final_test", final, self.protocol["test_ids"])
                save(self.root / "FINAL_RESULTS.json", {"life": result,
                    "note": "fixed TRAIN-selected incumbent; not TEST-selected"})
                self.progress("complete", test_accuracy=result["accuracy"])
            except Exception as exc:
                self.usage()
                save(self.root / "failure.json", {"type": type(exc).__name__, "detail": str(exc)[:6000]})
                self.progress("stopped_on_error", error_type=type(exc).__name__)
                raise
            finally:
                self.stopped.set()
                self.broker.close()
                save(self.root / "flash_metrics.json", {"gate": self.gate.snapshot(), "closed": True,
                    "usage": self.broker.ledger.summary()["overall"], "utc_unix": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--domain", choices=("alfworld", "dbbench", "webshop"), default="webshop")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--launch", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--evaluate", choices=("train", "test"))
    parser.add_argument("--phase")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--ids")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9_]+", args.name):
        parser.error("safe experiment name required")
    root = ROOT / "evolution/experiments" / args.name
    if args.evaluate:
        runner = Run(root)
        runner.verify()
        native_evaluate(root, args.evaluate, args.phase, args.manifest, json.loads(args.ids))
    elif args.run:
        Run(root).run()
    else:
        initialize(root, args.domain)
        if args.launch:
            with (root / "runner.log").open("a") as log:
                process = subprocess.Popen([str(TAU / ".venv/bin/python"), "-u", "-m", "evolution.life_agentbench_run",
                    "--name", args.name, "--domain", args.domain, "--run"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    env=child_env(PYTHONPATH=str(ROOT)), start_new_session=True)
            save(root / "launch.json", {"pid": process.pid, "started": time.time(), "domain": domain, "rounds": 6})
            print(json.dumps({"pid": process.pid, "path": str(root)}))
        else:
            print(root)


if __name__ == "__main__":
    main()
