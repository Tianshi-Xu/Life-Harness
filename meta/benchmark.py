"""TauBench adapter for the meta-harness outer loop.

Wraps ``TauBench/scripts/eval_harness.py`` as a subprocess, then converts the
produced ``harness_summary.json`` into the meta-harness score contract:
a JSON file with an ``accuracy`` field (tau-bench average_reward) plus
supporting metrics. Mirrors the role of benchmark.py in meta-harness'
text_classification reference example.

Split mapping: ``search`` -> tau2 ``train`` split, ``test`` -> ``test`` split.
Test results are written to a caller-chosen path so evolution never touches
them.

Set META_MOCK_EVAL=1 to skip the real evaluation and emit a deterministic
synthetic score (plumbing smoke tests only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

META_DIR = Path(__file__).resolve().parent
TAUBENCH_DIR = META_DIR.parent / "TauBench"

ALL_FLAGS = ("h2", "h3", "h4", "h5")


def build_eval_cmd(args: argparse.Namespace, save_to: str) -> list[str]:
    cmd = [
        "uv", "run", "python", "scripts/eval_harness.py",
        "--domain", args.domain,
        "--split", "train" if args.split == "search" else "test",
        "--save-to", save_to,
        "--trials", str(args.trials),
        "--concurrency", str(args.concurrency),
    ]
    if not args.no_harness:
        cmd.append("--enabled")
        for flag in args.flags.split(","):
            flag = flag.strip()
            if flag:
                if flag not in ALL_FLAGS:
                    raise ValueError(f"unknown harness flag: {flag}")
                cmd.append(f"--{flag}")
    if args.candidate_file:
        cmd += ["--harness-plugin", str(Path(args.candidate_file).resolve())]
    if args.num_tasks:
        cmd += ["--num-tasks", str(args.num_tasks)]
    if args.task_id_file:
        cmd += ["--task-id-file", args.task_id_file]
    if args.agent_llm:
        cmd += ["--agent-llm", args.agent_llm]
    if args.user_llm:
        cmd += ["--user-llm", args.user_llm]
    if args.user_api_base:
        cmd += ["--user-api-base", args.user_api_base]
    if args.user_disable_thinking:
        cmd.append("--user-disable-thinking")
    if args.max_steps:
        cmd += ["--max-steps", str(args.max_steps)]
    if args.nl:
        cmd.append("--nl")
    if args.h5_top_k is not None:
        cmd += ["--h5-top-k", str(args.h5_top_k)]
    return cmd


def run_eval(cmd: list[str], timeout: int, stdout_log: Path,
             taubench_dir: Path) -> int:
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    with stdout_log.open("w") as fh:
        proc = subprocess.run(
            cmd, cwd=str(taubench_dir), stdout=fh, stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    return proc.returncode


def extract_score(save_to: str, taubench_dir: Path) -> dict:
    summary_path = taubench_dir / "data" / "simulations" / save_to / "harness_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    return {
        "accuracy": summary["average_reward"],
        "average_reward": summary["average_reward"],
        "pass@k": summary.get("pass@k"),
        "pass^k": summary.get("pass^k"),
        "k": summary.get("k"),
        "per_task": summary.get("per_task", {}),
        "total_simulations": summary.get("total_simulations"),
        "total_tokens": summary.get("total_tokens", {}).get("total"),
        "harness_plugin": summary.get("harness_plugin"),
        "save_dir": str(summary_path.parent),
    }


def mock_score(candidate_file: str | None, domain: str, name: str = "",
               task_id_file: str | None = None,
               num_tasks: int | None = None) -> dict:
    seed = f"{domain}:{name}:{candidate_file or 'none'}"
    if candidate_file and Path(candidate_file).is_file():
        seed += Path(candidate_file).read_text()
    base = int(hashlib.sha1(seed.encode()).hexdigest()[:8], 16)
    ids: list[str] = []
    if task_id_file and Path(task_id_file).is_file():
        ids = [l.strip() for l in Path(task_id_file).read_text().splitlines()
               if l.strip() and not l.startswith("#")]
    elif num_tasks:
        ids = [str(i) for i in range(num_tasks)]
    else:
        ids = [str(i) for i in range(25)]
    per_task = {}
    for tid in ids:
        h = int(hashlib.sha1(f"{seed}:{tid}".encode()).hexdigest()[:8], 16)
        per_task[tid] = float(h % 100 < 60)  # deterministic 60% pass
    value = round(sum(per_task.values()) / len(per_task), 4) if per_task \
        else round(base % 100 / 100.0, 4)
    return {
        "accuracy": value,
        "average_reward": value,
        "pass@k": None, "pass^k": None, "k": 0,
        "per_task": per_task, "total_simulations": len(ids) or 0,
        "total_tokens": 0,
        "harness_plugin": candidate_file,
        "save_dir": None,
        "mock": True,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", required=True,
                   choices=["airline", "retail", "telecom"])
    p.add_argument("--split", required=True, choices=["search", "test"])
    p.add_argument("--candidate-name", required=True)
    p.add_argument("--candidate-file", default=None,
                   help="Path to the candidate plugin .py (omit for anchors)")
    p.add_argument("--flags", default="h2,h3,h4,h5",
                   help="Comma-separated harness layers to enable")
    p.add_argument("--no-harness", action="store_true",
                   help="Run the frozen baseline (no --enabled, no plugin)")
    p.add_argument("--run-tag", required=True,
                   help="Run name; determines the deterministic save_to dir")
    p.add_argument("--out", required=True, help="Path to write the score JSON")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--num-tasks", type=int, default=None)
    p.add_argument("--task-id-file", default=None,
                   help="evaluate only the task IDs listed in this file")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--agent-llm", default=None)
    p.add_argument("--user-llm", default=None)
    p.add_argument("--user-api-base", default=None)
    p.add_argument("--user-disable-thinking", action="store_true")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--eval-timeout", type=int, default=6 * 3600)
    p.add_argument("--nl", action="store_true",
                   help="enable the NL assertion judge (required for retail)")
    p.add_argument("--h5-top-k", type=int, default=None,
                   help="H5 skills injected (paper setting: 1; eval default: 3)")
    p.add_argument("--taubench-dir", default=None,
                   help="TauBench repo to run in (default: the main checkout; "
                        "life_loop passes its git worktree)")
    args = p.parse_args()
    taubench_dir = Path(args.taubench_dir).resolve() if args.taubench_dir else TAUBENCH_DIR

    save_to = (
        f"meta_harness/{args.run_tag}/{args.domain}/{args.candidate_name}"
        + ("_test" if args.split == "test" else "")
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if os.environ.get("META_MOCK_EVAL") == "1":
        score = mock_score(args.candidate_file, args.domain, args.candidate_name,
                           task_id_file=args.task_id_file,
                           num_tasks=args.num_tasks)
        score["cmd"] = []
    else:
        cmd = build_eval_cmd(args, save_to)
        save_dir = taubench_dir / "data" / "simulations" / save_to
        if save_dir.exists() and not (save_dir / "harness_summary.json").is_file():
            shutil.rmtree(save_dir)  # stale partial run: avoid interactive resume prompt
        # A crash can occur after the evaluator wrote its summary but before
        # val.json was committed. Life's request fingerprint has already
        # validated the inputs; recover that completed evaluation directly.
        rc = 0 if (save_dir / "harness_summary.json").is_file() else run_eval(
            cmd, args.eval_timeout, stdout_log=out_path.parent / "eval_stdout.log",
            taubench_dir=taubench_dir)
        if rc != 0:
            print(f"eval failed (rc={rc}), see {out_path.parent / 'eval_stdout.log'}",
                  file=sys.stderr)
            return rc
        score = extract_score(save_to, taubench_dir)
        results_path = taubench_dir / "data" / "simulations" / save_to / "results.json"
        if results_path.is_file():
            sims = json.loads(results_path.read_text()).get("simulations", [])
            n_infra = sum(1 for s in sims
                          if s.get("termination_reason") == "infrastructure_error")
            if sims and n_infra / len(sims) > 0.2:
                print(f"eval had {n_infra}/{len(sims)} infrastructure errors "
                      f"(>20%) — refusing to score; check model serving "
                      f"(see {out_path.parent / 'eval_stdout.log'})",
                      file=sys.stderr)
                return 1
        if not score["total_simulations"]:
            print(f"eval produced 0 valid simulations (infrastructure "
                  f"failure?), see {out_path.parent / 'eval_stdout.log'}",
                  file=sys.stderr)
            return 1
        score["cmd"] = cmd

    score.update({
        "domain": args.domain,
        "split": args.split,
        "candidate": args.candidate_name,
        "flags": None if args.no_harness else args.flags,
    })
    out_path.write_text(json.dumps(score, indent=2))
    print(f"[benchmark] {args.domain}/{args.candidate_name} "
          f"split={args.split} accuracy={score['accuracy']} -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
