"""AHE evaluation transport invoking native interpreters and exporting real traces."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path

from .records import export_job, write_json

HERE = Path(__file__).resolve().parents[1]
LIFE = HERE.parents[1]
MANIFEST = LIFE / "evolution/reproduction/reference/manifest.json"
MANIFEST_SHA256 = "1d60aa4e2d905a393d7dea0f81da02d9d8ec58a48fc7b8070532b574ceccd96a"


def interpreter(domain):
    native = LIFE / "AgentBench/.native"
    if domain in {"dbbench", "os"}:
        return native / "core/bin/python"
    if domain == "webshop":
        return native / "webshop-venv/bin/python"
    if domain == "alfworld":
        return native / "alfworld-venv/bin/python"
    return LIFE / "TauBench/.venv/bin/python"


def episode_timeout(domain):
    # Telecom's globally paced simulator can legitimately exceed 30 minutes.
    # Match the native Tau evaluator; do not change solver steps or token limits.
    return 21600 if domain in {"airline", "retail", "telecom"} else 1800


def native_env(domain=None, *, scope=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(HERE), str(LIFE)))
    env["NO_PROXY"] = env["no_proxy"] = "localhost,127.0.0.1"
    env["TAU_USER_LLM_MIN_INTERVAL_SECONDS"] = "3.75"
    if domain in {"airline", "retail", "telecom"} and os.environ.get("AHE_TAU_USER_BASE_" + domain.upper()):
        from .token_accounting import scoped_url
        if scope:
            for role, name in (("user_simulator", "AHE_TAU_USER_API_BASE"), ("judge", "AHE_TAU_JUDGE_API_BASE")):
                env[name] = scoped_url(os.environ["AHE_TAU_USER_BASE_" + domain.upper()], role=role, **scope)
        env["AHE_TAU_USER_MODEL"] = os.environ["AHE_TAU_USER_MODEL_" + domain.upper()]
        for name in list(env):
            if any(word in name.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")):
                env.pop(name)
        env["AHE_USER_API_KEY"] = "EMPTY"
        env["LLM_API_KEY"] = "EMPTY"
        env["OPENAI_API_KEY"] = "EMPTY"
        # MR pacing/admission is global in the host broker. Aliyun is not gated.
        env["TAU_USER_LLM_MIN_INTERVAL_SECONDS"] = "0"
    if domain == "webshop":
        java = LIFE / "AgentBench/.native/sysroot/usr/lib/jvm/java-11-openjdk-amd64"
        env.update(JAVA_HOME=str(java), JVM_PATH=str(java / "lib/server/libjvm.so"),
                   PATH=str(java / "bin") + ":" + env.get("PATH", ""), _JAVA_OPTIONS="-Xmx4g")
    return env


def describe(domain, destination):
    subprocess.run([str(interpreter(domain)), "-m", "ahe_native.native_worker",
        "--describe", domain, "--out", str(Path(destination).resolve())], env=native_env(domain),
        cwd=HERE, check=True, timeout=180)
    return json.loads(Path(destination).read_text())


def validate_request(request):
    domain = request["domain"]
    if domain == "os":
        if request.get("split") != "train" or not request.get("training_schedule_file"):
            raise ValueError("OS backend requires source-pinned native TRAIN schedule; TEST forbidden")
        from .training import scheduled_batch
        ids = [str(x) for x in request["task_ids"]]
        if ids != [str(x) for x in scheduled_batch(request)]:
            raise ValueError("OS tasks differ from frozen training schedule")
        if type(request["concurrency"]) is not int or not 1 <= request["concurrency"] <= 50:
            raise ValueError("invalid OS native concurrency")
        workspace = Path(request["workspace"]).resolve()
        if not (workspace / request["agent_config"]).is_file():
            raise ValueError("evolving NexAU OS workspace missing")
        from .capabilities import restricted_policy, validate_workspace
        if request.get("capability_policy") != restricted_policy("os"):
            raise ValueError("source-bound restricted OS capability policy required")
        validate_workspace(workspace, request["capability_policy"], request["agent_config"])
        return ids
    if hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise ValueError("frozen benchmark manifest changed; audit before running")
    historical = json.loads(MANIFEST.read_text())
    snapshot = MANIFEST.parent / "task_snapshot"
    for relative, expected in historical["snapshot_hashes"].items():
        if hashlib.sha256((snapshot / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen native task source changed: {relative}")
    spec = historical["domains"][domain]
    if request["split"] != "train":
        raise ValueError("optimization backend must never run test tasks")
    if domain == "dbbench":
        relative = "data/dbbench/db_out_new.jsonl"
        if hashlib.sha256((LIFE / "AgentBench" / relative).read_bytes()).hexdigest() != historical["data_hashes"][relative]:
            raise ValueError("frozen DB training data changed")
    ids = [str(x) for x in request["task_ids"]]
    if request.get("training_schedule_file"):
        from .training import scheduled_batch
        expected = [str(x) for x in scheduled_batch(request)]
        if ids != expected:
            raise ValueError("native tasks differ from the pinned iteration training batch")
        pool = set(expected)
    else:
        pool = {str(x) for batch in spec["batches"] for x in batch.get("indices", batch.get("task_ids", []))}
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= pool:
        raise ValueError("native task IDs must belong to the frozen training pool")
    if type(request["k"]) is not int or request["k"] < 1:
        raise ValueError("invalid repeat count")
    if type(request["concurrency"]) is not int or request["concurrency"] < 1:
        raise ValueError("invalid concurrency")
    workspace = Path(request["workspace"]).resolve()
    if not (workspace / request["agent_config"]).is_file():
        raise ValueError("evolving NexAU workspace config missing")
    if request.get("capability_policy"):
        from .capabilities import restricted_policy, validate_workspace
        if request["capability_policy"] != restricted_policy(domain):
            raise ValueError("restricted capability policy differs from host-pinned baseline")
        validate_workspace(workspace, request["capability_policy"], request["agent_config"])
    return ids


def execute(request):
    ids = validate_request(request)
    cached = {}
    reuse = request.get("reuse_completed_episodes")
    if reuse:
        if workspace_hash(request["workspace"]) != reuse["workspace_sha256"]:
            raise ValueError("cached episodes were measured with a different workspace")
        for item in reuse["episodes"]:
            folder = Path(item["folder"])
            for name, expected in item["files"].items():
                if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
                    raise ValueError("cached native episode changed after recovery audit")
            previous = json.loads((folder / "request.json").read_text())
            record = json.loads((folder / "record.json").read_text())
            if any(previous.get(k) != request.get(k) for k in
                   ("domain", "split", "agent_config", "capability_policy")):
                raise ValueError("cached native episode protocol differs")
            key = (previous["task_id"], previous["trial"])
            if (key in cached or key[0] not in ids or key[1] not in range(request["k"])
                    or key != (record["task_id"], record["trial"])):
                raise ValueError("cached native episode identity differs")
            cached[key] = (record, folder / "agent")
    output = Path(request["output_root"])
    stamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S") + "_" + uuid.uuid4().hex[:8]
    pending = output / ".native_pending" / stamp
    pending.mkdir(parents=True)

    def run(task_id, trial):
        if (task_id, trial) in cached:
            return cached[(task_id, trial)]
        key = hashlib.sha256(f"{task_id}:{trial}".encode()).hexdigest()[:12]
        folder = pending / key
        folder.mkdir()
        value = {"domain": request["domain"], "split": "train", "task_id": task_id,
                 "trial": trial, "workspace": request["workspace"],
                 "agent_config": request["agent_config"], "trace_dir": str(folder / "agent")}
        if request.get("capability_policy"):
            value["capability_policy"] = request["capability_policy"]
        write_json(folder / "request.json", value)
        with (folder / "environment.log").open("w") as log:
            proc = subprocess.run([str(interpreter(request["domain"])), "-m",
                "ahe_native.native_worker", "--request", str(folder / "request.json"),
                "--out", str(folder / "record.json")], cwd=HERE, env=native_env(request["domain"], scope={
                    "experiment": Path(request["workspace"]).parent.name,
                    "domain": request["domain"], "round": request.get("iteration", 1)}),
                stdout=log, stderr=subprocess.STDOUT, timeout=episode_timeout(request["domain"]))
        if proc.returncode or not (folder / "record.json").exists():
            raise RuntimeError(f"native worker failed for {task_id}:{trial}; inspect {folder}/environment.log")
        record = json.loads((folder / "record.json").read_text())
        print(json.dumps({"domain": request["domain"], "task_id": task_id,
                          "trial": trial, "success": record["success"],
                          "infra": record["infrastructure_error"]}), flush=True)
        return record, folder / "agent"

    with concurrent.futures.ThreadPoolExecutor(max_workers=request["concurrency"]) as pool:
        futures = [pool.submit(run, task_id, trial) for task_id in ids for trial in range(request["k"])]
        values = [future.result() for future in futures]
    payload = {"schema_version": 1, "domain": request["domain"], "split": "train",
               "records": [record for record, _ in values]}
    runtime_artifacts = {(record["task_id"], record["trial"]): folder for record, folder in values}
    job = export_job(payload, output / stamp, runtime_artifacts=runtime_artifacts)
    write_json(job / "workspace_receipt.json", {"workspace_sha256": workspace_hash(request["workspace"]),
                "benchmark_manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                "nexau_runtime": True, "solver": "qwen3-4b", "released_life_layers": False,
                "capability_profile": request.get("capability_policy", {}).get("profile", "historical-unrestricted")})
    if reuse:
        write_json(job / "episode_reuse_receipt.json", {"completed_episodes_reused": len(cached),
            "source": reuse, "new_episodes": len(values) - len(cached)})
    return job


def workspace_hash(workspace):
    root = Path(workspace)
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in root.rglob("*") if p.is_file() and ".git" not in p.parts
             and "__pycache__" not in p.parts}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    print(execute(json.loads(args.request.read_text())), flush=True)


if __name__ == "__main__":
    main()
