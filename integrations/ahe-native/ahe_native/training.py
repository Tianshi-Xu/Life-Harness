"""Frozen, rotating public train batches; never contains test tasks or answers."""
import hashlib
import json
import random
from pathlib import Path

from .backend import MANIFEST, MANIFEST_SHA256

AGENTBENCH = ("alfworld", "dbbench", "webshop")


def train_ids(domain, historical):
    spec = historical["domains"][domain]
    if domain == "webshop":
        start, end = spec["train_range"]
        if end - start != spec["train_pool"]:
            raise ValueError("inconsistent frozen WebShop train range")
        return list(range(start, end))
    if domain in ("alfworld", "dbbench"):
        return list(range(spec["train_pool"]))
    raise ValueError("formal rotating launcher accepts AgentBench only")


def make_schedule(iterations=9, size=50, seed=20261003):
    if type(iterations) is not int or type(size) is not int or iterations < 1 or size < 1:
        raise ValueError("positive iteration and batch sizes required")
    if hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise ValueError("historical benchmark manifest changed")
    historical = json.loads(MANIFEST.read_text())
    result = {"schema_version": 1, "historical_manifest_sha256": MANIFEST_SHA256,
        "split": "train", "iterations": iterations, "tasks_per_iteration": size,
        "seed": seed, "batch_policy": "pre-frozen disjoint shuffled training batches",
        "domains": {}}
    for domain in AGENTBENCH:
        pool = train_ids(domain, historical)
        if iterations * size > len(pool):
            raise ValueError("not enough unique training tasks for this fixed budget")
        random.Random(f"{seed}:{domain}").shuffle(pool)
        batches = [{"iteration": i + 1, "id": f"b{i:02d}",
                    "task_ids": pool[i * size:(i + 1) * size]}
                   for i in range(iterations)]
        result["domains"][domain] = {"train_pool_size": len(pool), "batches": batches}
    return result


def scheduled_batch(request):
    path = Path(request["training_schedule_file"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != request["training_schedule_sha256"]:
        raise ValueError("frozen training schedule changed")
    schedule = json.loads(raw)
    if schedule.get("split") != "train" or schedule.get("schema_version") != 1:
        raise ValueError("training schedule cannot contain test tasks")
    if request.get("domain") == "os":
        from evolution.os_comparison.protocol import scheduled_batch as os_batch
        return os_batch(request, schedule)
    if schedule["historical_manifest_sha256"] != MANIFEST_SHA256:
        raise ValueError("training schedule historical manifest mismatch")
    historical = json.loads(MANIFEST.read_text())
    domain = request["domain"]
    if schedule.get("batch_policy") == "frozen_tau_native_batches":
        from .tau_training import scheduled_tau_batch
        return scheduled_tau_batch(request, schedule, historical)
    pool = set(train_ids(domain, historical))
    batches = schedule["domains"][domain]["batches"]
    if len(batches) != schedule["iterations"]:
        raise ValueError("training schedule length mismatch")
    seen = set()
    for iteration, batch in enumerate(batches, 1):
        ids = batch["task_ids"]
        if batch["iteration"] != iteration or len(ids) != schedule["tasks_per_iteration"]:
            raise ValueError("training batch size or iteration mismatch")
        if any(type(task) is not int for task in ids) or not set(ids) <= pool:
            raise ValueError("task outside native training split")
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("rotating training tasks must not repeat")
        seen.update(ids)
    iteration = request["iteration"]
    if type(iteration) is not int or not 1 <= iteration <= len(batches):
        raise ValueError("training iteration outside frozen budget")
    return list(batches[iteration - 1]["task_ids"])
