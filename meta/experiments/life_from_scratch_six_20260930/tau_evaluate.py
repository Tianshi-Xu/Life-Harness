"""Evaluate one frozen Tau batch with an ordered Life register() chain."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from protocol import HERE, ROOT, TAU_DOMAINS, atomic_json, digest, task_ids, validate_protocol

BENCHMARK = ROOT / "meta" / "benchmark.py"
USER_MODEL = "openai/qwen3.8-flash"
SOLVER_MODEL = "openai/qwen3-4b"


def plugin_hashes(loader: Path) -> dict[str, str]:
    result = {str(loader): digest(loader)}
    manifest_path = loader.with_name("manifest.json")
    if manifest_path.exists():
        result[str(manifest_path)] = digest(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        for item in manifest.get("layers", []):
            layer = Path(item)
            if not layer.is_absolute():
                layer = manifest_path.parent / layer
            result[str(layer.resolve())] = digest(layer.resolve())
    return result


def add_trial_rewards(score: dict) -> dict:
    grouped: dict[str, list[float]] = defaultdict(list)
    results = Path(score["save_dir"]) / "results.json"
    for row in json.loads(results.read_text()).get("simulations", []):
        reward = float((row.get("reward_info") or {}).get("reward", 0))
        grouped[str(row["task_id"])].append(reward)
    score["per_task_trials"] = dict(grouped)
    return score


def evaluate(domain: str, split: str, phase: str, loader: Path, batch: int = 0,
             task_limit: int | None = None) -> dict:
    manifest = validate_protocol()
    spec = manifest["domains"][domain]
    ids = task_ids(spec, batch, split)
    if task_limit is not None:
        ids = ids[:task_limit]
    trials = spec["train_trials"] if split == "train" else 1
    api_base = os.environ.get("USER_API_BASE", "")
    if not api_base.startswith(("http://", "https://")):
        raise RuntimeError("USER_API_BASE must provide the current Tau user endpoint")
    loader = Path(loader).resolve()
    id_dir = HERE / domain / "task_ids"
    id_dir.mkdir(parents=True, exist_ok=True)
    id_file = id_dir / (
        f"smoke_{phase}.txt" if task_limit is not None
        else f"batch_{batch:02d}.txt" if split == "train" else "test.txt"
    )
    frozen_ids = "\n".join(map(str, ids)) + "\n"
    if id_file.exists() and id_file.read_text() != frozen_ids:
        raise RuntimeError("Frozen task IDs changed")
    id_file.write_text(frozen_ids)
    out_dir = HERE / domain / "evals" / phase
    out = out_dir / ("val.json" if split == "train" else "test.json")
    config = {
        "domain": domain, "split": split, "phase": phase,
        "batch": batch if split == "train" else None, "task_ids": ids,
        "trials": trials, "concurrency": 8, "solver": SOLVER_MODEL,
        "user": USER_MODEL, "plugin_hashes": plugin_hashes(loader),
        "protocol_hash": digest(HERE / "protocol.json"), "task_limit": task_limit,
    }
    config_path = out_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("Resume configuration mismatch")
    if not config_path.exists():
        atomic_json(config_path, config)
    if out.exists():
        return add_trial_rewards(json.loads(out.read_text()))
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(BENCHMARK), "--domain", domain,
        "--split", "search" if split == "train" else "test",
        "--candidate-name", phase, "--candidate-file", str(loader),
        "--flags", "h2,h3,h4,h5", "--run-tag", f"{HERE.name}_{domain}",
        "--out", str(out), "--trials", str(trials), "--task-id-file", str(id_file),
        "--concurrency", "8", "--agent-llm", SOLVER_MODEL,
        "--user-llm", USER_MODEL, "--user-api-base", api_base,
        "--user-enable-thinking", "--eval-timeout", "21600",
    ]
    if domain == "retail":
        command.append("--nl")
    env = dict(os.environ)
    env["UV_CACHE_DIR"] = "/tmp/uv-cache"
    env["PATH"] = "/mnt/data/xts/tools/uv/bin:" + env.get("PATH", "")
    with (out_dir / "driver.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=21600)
    if result.returncode or not out.exists():
        raise RuntimeError(f"Tau evaluation failed; see {out_dir / 'driver.log'}")
    score = add_trial_rewards(json.loads(out.read_text()))
    score.update(config)
    atomic_json(out, score)
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", choices=TAU_DOMAINS)
    parser.add_argument("split", choices=("train", "test"))
    parser.add_argument("phase")
    parser.add_argument("loader", type=Path)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--task-limit", type=int)
    args = parser.parse_args()
    score = evaluate(args.domain, args.split, args.phase, args.loader, args.batch,
                     args.task_limit)
    print(json.dumps(score, ensure_ascii=False))


if __name__ == "__main__":
    main()
