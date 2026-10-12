import os
"""Host-only native split identities and immutable six-round OS schedules."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import random
import re

from .native import AB, ROOT, close_controller, task


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def native_sources():
    paths = [AB / "configs/tasks/os.yaml", AB / "extra/docker-compose.yml"]
    for folder in (AB / "src/server/tasks/os_interaction", AB / "src/server/harness",
                   AB / "data/os_interaction"):
        paths += [p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    return {str(p.resolve()): digest(p) for p in sorted(set(paths))}


def verify_sources(hashes):
    if not hashes or any(not Path(p).is_file() or digest(p) != expected for p, expected in hashes.items()):
        raise ValueError("native OS source/data changed")


async def split_spec():
    train, _ = task("train")
    test, _ = task("test")
    try:
        train_ids, test_ids = sorted(train.problem_configs), sorted(test.problem_configs)
        descriptions = {entry["config"].description for entry in train.problem_configs.values()}
        if descriptions & {entry["config"].description for entry in test.problem_configs.values()}:
            raise ValueError("native OS TRAIN/TEST descriptions overlap")
        return {"train_pool_size": len(train_ids), "train_ids": train_ids,
                "test_ids": test_ids, "round_limit": train.round_limit,
                "identity": "split plus deterministic native integer index; train/test integers are separate namespaces"}
    finally:
        await close_controller(train)
        await close_controller(test)


def prepare(name, seed=20261008):
    if not re.fullmatch(r"[a-z][a-z0-9_]+", name):
        raise ValueError("safe new experiment name required")
    root = ROOT / "evolution/experiments" / name
    if root.exists():
        raise FileExistsError("never overwrite an experiment or replay a paid proposal")
    spec = asyncio.run(split_spec())
    pool = list(spec["train_ids"])
    random.Random(f"{seed}:os").shuffle(pool)
    if len(pool) < 300:
        raise ValueError("six disjoint batches of 50 TRAIN tasks required")
    batches = [{"iteration": i + 1, "id": f"b{i:02d}", "task_ids": pool[i*50:(i+1)*50]} for i in range(6)]
    hashes = native_sources()
    schedule = {"schema_version": 1, "split": "train", "iterations": 6,
                "tasks_per_iteration": 50, "seed": seed,
                "batch_policy": "frozen_os_native_batches", "native_source_hashes": hashes,
                "domains": {"os": {"train_pool_size": spec["train_pool_size"], "batches": batches}}}
    protocol = {"name": name, "domain": "os", "methods": ["meta", "ahe", "life"], "rounds": 6,
                "native_spec": spec, "batches": batches, "seed": seed,
                "solver": {"model": "qwen3-4b", "model_path": os.environ.get("LIFE_SOLVER_TOKENIZER", "Qwen/Qwen3-4B"),
                           "max_context": 32768, "max_output": 4096, "enable_thinking": False,
                           "temperature": 0.0, "workers": 50},
                "optimizer": {"requested_model": "deepseek-v4-flash", "upstream_model": "deepseek-v4.1-flash",
                              "same_route_confirmed_by_user": True,
                              "base_url": "https://maas.qianwenaiapi.com/compatible-mode/v1",
                              "global_concurrency": 8, "min_interval_seconds": 1.25,
                              "cumulative_token_limit": None, "life_max_output": 100000,
                              "life_interaction_limit": None, "life_max_repairs": 3},
                "initialization": "policy-free Meta/AHE identity and empty Life H2-H5",
                "selection": {"meta": "strict improvement on same complete batch; advance batch on rejection",
                              "ahe": "unchanged official best-ever/evaluate-before-evolve logic",
                              "life": "strict improvement on same complete batch; rotating batches"},
                "ahe_capability_profile": "h2345-budget-restricted-v1",
                "ahe_detailed_cases_per_round": 10,
                "accounting": "API-measured optimizer input plus output only, including failed/repaired attempts; unknown is never zero",
                "test": "once per method after all eighteen TRAIN optimizations and source-bound training seal; no TEST evidence",
                "prior_test_exposure": False,
                "native_source_hashes": hashes}
    root.mkdir(parents=True)
    save(root / "training_schedule.json", schedule)
    protocol["training_schedule_sha256"] = digest(root / "training_schedule.json")
    save(root / "protocol.json", protocol)
    save(root / "status.json", {"phase": "prepared", "completed_rounds": {m: 0 for m in protocol["methods"]}})
    return root


def scheduled_batch(request, schedule):
    if (request.get("domain") != "os" or request.get("split") != "train" or request.get("k") != 1
            or schedule.get("schema_version") != 1 or schedule.get("split") != "train"
            or schedule.get("batch_policy") != "frozen_os_native_batches"
            or schedule.get("iterations") != 6 or schedule.get("tasks_per_iteration") != 50
            or set(schedule.get("domains", {})) != {"os"}):
        raise ValueError("fixed six-round native OS TRAIN protocol required")
    verify_sources(schedule["native_source_hashes"])
    spec = schedule["domains"]["os"]
    seen = set()
    if len(spec["batches"]) != 6:
        raise ValueError("six native OS batches required")
    for number, batch in enumerate(spec["batches"], 1):
        ids = batch["task_ids"]
        if (batch["iteration"] != number or len(ids) != 50 or len(set(ids)) != 50
                or any(type(i) is not int or not 0 <= i < spec["train_pool_size"] for i in ids)
                or seen.intersection(ids)):
            raise ValueError("invalid or overlapping native OS TRAIN IDs")
        seen.update(ids)
    number = request["iteration"]
    if type(number) is not int or not 1 <= number <= 6:
        raise ValueError("OS optimization round outside frozen budget")
    return list(spec["batches"][number-1]["task_ids"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    print(prepare(args.name))
