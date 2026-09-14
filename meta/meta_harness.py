#!/usr/bin/env python3
"""Meta-harness outer loop for Life-Harness / TauBench.

Adapted from meta-harness (reference_examples/text_classification/
meta_harness.py). The base model and the environment are frozen; the search
object is harness code — single-file Python plugins evaluated through
``TauBench/scripts/eval_harness.py --harness-plugin``.

Loop per iteration:
  1. proposer.py runs one headless qoder session with the skill in
     skills/tau2-harness/SKILL.md; it writes candidate plugins under
     <run>/candidates/ and a <run>/pending_eval.json manifest.
  2. Each candidate is scanned for forbidden references, import-smoke-checked
     inside the TauBench uv environment, then evaluated on the search split
     of every selected domain via benchmark.py.
  3. Scores are appended to evolution_summary.jsonl and frontier_val.json is
     updated. Evolution only ever sees the search split; ``--test`` finalizes
     the run on the held-out test split and freezes it.

Anchors evaluated before iteration 1 (unless --skip-baselines):
  baseline  — no harness (the paper's frozen baseline)
  h2345     — the paper's full H2+H3+H4+H5 harness, no plugin
"""

from __future__ import annotations

import argparse
import json
import py_compile
import shutil
import subprocess
import sys
import time
from pathlib import Path

import proposer

META_DIR = Path(__file__).resolve().parent
LIFE_HARNESS_DIR = META_DIR.parent
TAUBENCH_DIR = LIFE_HARNESS_DIR / "TauBench"
RUNS_DIR = META_DIR / "runs"
SKILL_PATH = META_DIR / "skills" / "tau2-harness" / "SKILL.md"

ANCHORS = {
    "baseline": {"no_harness": True, "flags": None},
    "h2345": {"no_harness": False, "flags": "h2,h3,h4,h5"},
}

# Substrings that must never appear in candidate code: they indicate the
# plugin is reading task data, split definitions, prior simulation logs, the
# network, or the filesystem at runtime (all leak / side-effect vectors).
FORBIDDEN_REFERENCES = (
    "split_tasks",
    "tasks.json",
    "data/tau2",
    "tau2/data",
    "simulations",
    "open(",
    "subprocess",
    "os.environ",
    "requests",
    "urllib",
    "socket",
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def load_json(path: Path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text())


def check_not_finalized(run_dir: Path) -> None:
    finalized = load_json(run_dir / "finalized.json")
    if finalized and finalized.get("status") == "complete":
        sys.exit(
            f"run {run_dir.name} is finalized (frozen). "
            "Use --test to re-run missing test evals, or start a new --run-name."
        )


# ---------------------------------------------------------------- eval


def run_benchmark(run_dir: Path, domain: str, name: str, split: str,
                  args: argparse.Namespace, candidate_file: Path | None,
                  no_harness: bool, flags: str | None) -> dict | None:
    """Evaluate one candidate on one domain; return the score dict or None.

    Idempotent: an existing score file is reused, which makes interrupted
    iterations resumable.
    """
    if split == "search":
        out = run_dir / "evals" / domain / name / "val.json"
    else:
        out = run_dir / "test" / domain / name / "test.json"
    if out.is_file():
        return load_json(out)

    cmd = [
        sys.executable, str(META_DIR / "benchmark.py"),
        "--domain", domain,
        "--split", split,
        "--candidate-name", name,
        "--run-tag", run_dir.name,
        "--out", str(out),
        "--trials", str(args.trials),
        "--concurrency", str(args.concurrency),
        "--taubench-dir", str(TAUBENCH_DIR),
    ]
    if candidate_file:
        cmd += ["--candidate-file", str(candidate_file)]
    if no_harness:
        cmd += ["--no-harness"]
    elif flags is not None:
        cmd += ["--flags", flags]
    for opt in ("num_tasks", "agent_llm", "user_llm", "user_api_base",
                "max_steps", "eval_timeout", "h5_top_k"):
        value = getattr(args, opt)
        if value is not None:
            cmd += [f"--{opt.replace('_', '-')}", str(value)]
    if getattr(args, "user_disable_thinking", False):
        cmd.append("--user-disable-thinking")
    if domain in (getattr(args, "nl_domains", None) or []):
        cmd.append("--nl")

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0 or not out.is_file():
        print(f"  [eval-error] {domain}/{name} split={split}: "
              f"{proc.stdout.strip()} {proc.stderr.strip()}")
        return None
    score = load_json(out)
    score["timing_s"] = round(elapsed, 1)
    out.write_text(json.dumps(score, indent=2))
    return score


def active_anchors(args: argparse.Namespace) -> dict:
    if getattr(args, "from_scratch", False):
        return {"baseline": ANCHORS["baseline"]}
    return ANCHORS


def ensure_anchors(run_dir: Path, args: argparse.Namespace) -> None:
    summary_path = run_dir / "evolution_summary.jsonl"
    done = {r["system"] for r in read_jsonl(summary_path)
            if r["iteration"] == 0 and r.get("outcome") == "ok"}
    for name, spec in active_anchors(args).items():
        if name in done:
            continue
        per_domain, ok = {}, True
        for domain in args.domains:
            score = run_benchmark(run_dir, domain, name, "search", args,
                                  candidate_file=None,
                                  no_harness=spec["no_harness"],
                                  flags=spec["flags"])
            if score is None:
                ok = False
                break
            per_domain[domain] = score["accuracy"]
        row = {
            "iteration": 0,
            "system": name,
            "scores": per_domain,
            "mean_accuracy": round(sum(per_domain.values()) / len(per_domain), 4),
            "outcome": "ok" if ok else "eval_error",
            "hypothesis": "anchor",
        }
        append_jsonl(summary_path, row)
        print(f"[anchor] {name}: {per_domain}")
        if not ok:
            sys.exit("anchor evaluation failed — check model/user-simulator services")


# ---------------------------------------------------------- propose phase


def render_prompt(iteration: int, run_dir: Path, args: argparse.Namespace) -> str:
    skill = SKILL_PATH.read_text()
    frontier = load_json(run_dir / "frontier_val.json", default={})
    return f"""Follow these skill instructions:

## Skill: tau2-harness
{skill}

---

## Task: propose {args.candidates_per_iter} candidate harness(es), iteration {iteration}

State and resources:
- Run directory: {run_dir}
- Score history: {run_dir}/evolution_summary.jsonl
- Current frontier: {run_dir}/frontier_val.json (content below)
- Candidates directory (write your plugins here): {run_dir}/candidates/
- TauBench repo (read-only for you): {TAUBENCH_DIR}
- Per-candidate search-split scores incl. per-task rewards and simulation
  dirs for failure analysis: {run_dir}/evals/<domain>/<candidate>/val.json

Current frontier:
```json
{json.dumps(frontier, indent=2)}
```

Selected domains for this run: {", ".join(args.domains)}
Mode: {"FROM-SCRATCH — candidates must contain ALL harness logic in the plugin; no H2-H5 layers are enabled by default (use \"flags\": [])" if args.from_scratch else "LAYERED — candidates run on top of the full H2-H5 harness by default"}

Write exactly one file at {run_dir}/pending_eval.json with this schema:
```json
{{
  "iteration": {iteration},
  "candidates": [
    {{
      "name": "<unique lower_snake_case>",
      "file": "candidates/<name>.py",
      "hypothesis": "<what failure mode this fixes and why>",
      "flags": ["h2", "h3", "h4", "h5"]
    }}
  ]
}}
```
"flags" lists the built-in H-layers to enable alongside your plugin
(subset of h2,h3,h4,h5). Names must not repeat any "system" already present
in evolution_summary.jsonl. The run directory above is the only place you
may write files; treat the TauBench repo as read-only reference material.
"""


MOCK_CANDIDATE = '''"""No-op candidate emitted by --mock-proposer for plumbing smoke tests."""


def register():
    pass
'''


def propose(iteration: int, run_dir: Path, args: argparse.Namespace) -> list[dict]:
    pending_path = run_dir / "pending_eval.json"
    saved_manifest = run_dir / "prompts" / f"iter_{iteration:02d}_pending_eval.json"
    if saved_manifest.exists():
        return load_json(saved_manifest).get("candidates", [])[:args.candidates_per_iter]
    if pending_path.exists():
        pending_path.unlink()

    if args.mock_proposer:
        name = f"mock_candidate_i{iteration}"
        (run_dir / "candidates" / f"{name}.py").write_text(MOCK_CANDIDATE)
        return [{
            "name": name,
            "file": f"candidates/{name}.py",
            "hypothesis": "mock proposer plumbing test",
            "flags": ["h2", "h3", "h4", "h5"],
        }]

    prompt = render_prompt(iteration, run_dir, args)
    experience = run_dir / "search_experience.md"
    if experience.is_file():
        prompt += "\nRun-specific search instructions:\n" + experience.read_text()
    prompts_dir = run_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    (prompts_dir / f"iter_{iteration:02d}.md").write_text(prompt)

    result = proposer.run(
        prompt=prompt,
        cwd=run_dir,
        add_dirs=[TAUBENCH_DIR, SKILL_PATH.parent],
        model=args.proposer_model,
        timeout=args.propose_timeout,
        log_path=run_dir / "qoder_logs" / f"iter_{iteration:02d}.json",
    )
    if result.timed_out:
        print(f"  [proposer] timed out after {args.propose_timeout}s")

    if not pending_path.is_file() and not result.timed_out:
        # One nudge retry: the proposer may have done the analysis but missed
        # the manifest contract.
        nudge = (f"You did not create {pending_path}. Create it now with the "
                 f"schema from the task instructions, referencing the candidate "
                 f"files you already wrote under {run_dir}/candidates/.")
        proposer.run(prompt=nudge, cwd=run_dir,
                     add_dirs=[TAUBENCH_DIR, SKILL_PATH.parent],
                     model=args.proposer_model, timeout=args.propose_timeout,
                     log_path=run_dir / "qoder_logs" / f"iter_{iteration:02d}_nudge.json")

    if not pending_path.is_file():
        print("  [proposer] no pending_eval.json produced; skipping iteration")
        return []

    pending = load_json(pending_path)
    shutil.move(str(pending_path),
                str(prompts_dir / f"iter_{iteration:02d}_pending_eval.json"))
    candidates = pending.get("candidates", [])
    return candidates[: args.candidates_per_iter]


# --------------------------------------------------------- validate phase


def smoke_check(candidate_file: Path) -> tuple[bool, str]:
    """py_compile + import inside the TauBench uv environment."""
    try:
        py_compile.compile(str(candidate_file), doraise=True)
    except py_compile.PyCompileError as e:
        return False, f"py_compile: {e}"
    code = (
        "import importlib.util;"
        f"spec=importlib.util.spec_from_file_location('cand', r'{candidate_file}');"
        "m=importlib.util.module_from_spec(spec);"
        "spec.loader.exec_module(m);"
        "r=getattr(m,'register',None);"
        "r() if callable(r) else None;"
        "print('import/register ok')"
    )
    try:
        proc = subprocess.run(
            ["uv", "run", "python", "-c", code],
            cwd=str(TAUBENCH_DIR), capture_output=True, text=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        return False, "import timed out"
    if proc.returncode != 0:
        return False, proc.stderr.strip()[-500:]
    return True, ""


def validate_candidates(candidates: list[dict], iteration: int,
                        run_dir: Path) -> list[tuple[dict, str]]:
    """Return per-candidate (entry, reject_reason); empty reason = accepted."""
    summary_path = run_dir / "evolution_summary.jsonl"
    used_names = {r["system"] for r in read_jsonl(summary_path)}
    results = []
    for entry in candidates:
        name = entry.get("name", "")
        file_rel = entry.get("file", "")
        candidate_file = run_dir / file_rel
        if not name or not name.replace("_", "").isalnum() or name.lower() != name:
            results.append((entry, "bad name"))
        elif name in used_names or name in ANCHORS:
            results.append((entry, "name already used"))
        elif not candidate_file.is_file():
            results.append((entry, f"missing file {file_rel}"))
        else:
            text = candidate_file.read_text()
            hit = next((s for s in FORBIDDEN_REFERENCES if s in text), None)
            if hit:
                results.append((entry, f"forbidden reference: {hit!r}"))
                continue
            ok, err = smoke_check(candidate_file)
            results.append((entry, "" if ok else f"smoke: {err}"))
    return results


# ------------------------------------------------------------- main loop


def update_frontier(run_dir: Path, args: argparse.Namespace) -> dict:
    rows = [r for r in read_jsonl(run_dir / "evolution_summary.jsonl")
            if r.get("outcome") == "ok"]
    frontier: dict = {}
    for domain in args.domains:
        best = None
        for r in rows:
            if domain in r.get("scores", {}):
                acc = r["scores"][domain]
                if best is None or acc > best[1]:
                    best = (r["system"], acc)
        if best:
            frontier[domain] = {"best_system": best[0], "accuracy": best[1]}
    full = [r for r in rows
            if set(args.domains) <= set(r.get("scores", {}))]
    if full:
        best_overall = max(full, key=lambda r: r["mean_accuracy"])
        frontier["_overall"] = {"best_system": best_overall["system"],
                                "mean_accuracy": best_overall["mean_accuracy"]}
    (run_dir / "frontier_val.json").write_text(json.dumps(frontier, indent=2))
    return frontier


def evolve(args: argparse.Namespace) -> None:
    run_dir = RUNS_DIR / args.run_name
    if args.fresh and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "candidates").mkdir(exist_ok=True)
    check_not_finalized(run_dir)

    if not args.skip_baselines:
        print("== anchors (iteration 0) ==")
        ensure_anchors(run_dir, args)
    update_frontier(run_dir, args)

    summary_path = run_dir / "evolution_summary.jsonl"
    history = read_jsonl(summary_path)
    start = max((r["iteration"] for r in history), default=0) + 1

    for iteration in range(start, args.iterations + 1):
        print(f"\n== iteration {iteration}/{args.iterations} ==")
        candidates = propose(iteration, run_dir, args)
        if not candidates:
            continue
        for entry, reason in validate_candidates(candidates, iteration, run_dir):
            name = entry.get("name", "?")
            if reason:
                print(f"  [rejected] {name}: {reason}")
                append_jsonl(summary_path, {
                    "iteration": iteration, "system": name,
                    "hypothesis": entry.get("hypothesis", ""),
                    "outcome": "rejected", "reject_reason": reason,
                })
                continue
            flags = entry.get("flags")
            default_flags = "" if args.from_scratch else "h2,h3,h4,h5"
            flags_str = ",".join(flags) if flags else default_flags
            per_domain = {}
            for domain in args.domains:
                print(f"  [eval] {domain}/{name} ...")
                score = run_benchmark(
                    run_dir, domain, name, "search", args,
                    candidate_file=run_dir / entry["file"],
                    no_harness=False, flags=flags_str)
                if score is None:
                    per_domain = {}
                    break
                per_domain[domain] = score["accuracy"]
            if per_domain:
                row = {
                    "iteration": iteration, "system": name,
                    "hypothesis": entry.get("hypothesis", ""),
                    "flags": flags_str, "scores": per_domain,
                    "mean_accuracy": round(sum(per_domain.values()) / len(per_domain), 4),
                    "outcome": "ok",
                }
            else:
                row = {"iteration": iteration, "system": name,
                       "hypothesis": entry.get("hypothesis", ""),
                       "outcome": "eval_error"}
            append_jsonl(summary_path, row)
            print(f"  [score] {name}: {row.get('scores', 'eval_error')}")
        frontier = update_frontier(run_dir, args)
        print(f"  [frontier] {json.dumps(frontier)}")

    print(f"\nevolution done. Frontier: {RUNS_DIR / args.run_name}/frontier_val.json")
    print(f"finalize with: python {META_DIR}/meta_harness.py --test --run-name {args.run_name}")


def finalize(args: argparse.Namespace) -> None:
    run_dir = RUNS_DIR / args.run_name
    if not run_dir.is_dir():
        sys.exit(f"no such run: {run_dir}")
    finalized_path = run_dir / "finalized.json"
    finalized = load_json(finalized_path, default={})
    if finalized.get("status") != "in_progress":
        finalized = {"status": "in_progress", "systems": []}
        finalized_path.write_text(json.dumps(finalized, indent=2))

    frontier = load_json(run_dir / "frontier_val.json", default={})
    anchors = active_anchors(args)
    systems = set(anchors)
    for domain in args.domains:
        if domain in frontier:
            systems.add(frontier[domain]["best_system"])
    if "_overall" in frontier:
        systems.add(frontier["_overall"]["best_system"])

    flags_by_system = {}
    for r in read_jsonl(run_dir / "evolution_summary.jsonl"):
        if r.get("flags") is not None:
            flags_by_system[r["system"]] = r["flags"]
    default_test_flags = "" if args.from_scratch else "h2,h3,h4,h5"

    for domain in args.domains:
        for name in sorted(systems):
            candidate_file = run_dir / "candidates" / f"{name}.py"
            if name in ANCHORS:
                spec = ANCHORS[name]
                score = run_benchmark(run_dir, domain, name, "test", args,
                                      candidate_file=None,
                                      no_harness=spec["no_harness"],
                                      flags=spec["flags"])
            else:
                if not candidate_file.is_file():
                    print(f"  [skip] {name}: candidate file missing")
                    continue
                score = run_benchmark(run_dir, domain, name, "test", args,
                                      candidate_file=candidate_file,
                                      no_harness=False,
                                      flags=flags_by_system.get(name,
                                                                default_test_flags))
            if score:
                print(f"  [test] {domain}/{name}: {score['accuracy']}")
                if name not in finalized["systems"]:
                    finalized["systems"].append(name)

    finalized["status"] = "complete"
    finalized_path.write_text(json.dumps(finalized, indent=2))
    print(f"finalized. test results under {run_dir}/test/")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iterations", type=int, default=10,
                   help="number of proposer iterations (anchors are iteration 0)")
    p.add_argument("--run-name", default="tau2_pilot")
    p.add_argument("--fresh", action="store_true", help="wipe the run dir first")
    p.add_argument("--skip-baselines", action="store_true")
    p.add_argument("--from-scratch", action="store_true",
                   help="purist meta-harness arm: only the no-harness baseline "
                        "anchor, candidates carry all logic in the plugin")
    p.add_argument("--test", action="store_true",
                   help="finalize: evaluate anchors + frontier on the held-out "
                        "test split, then freeze the run")
    p.add_argument("--domains", nargs="+", default=None,
                   choices=["airline", "retail", "telecom"],
                   help="evolve: default all three; finalize: default to the "
                        "domains present in the run's evolution history")
    p.add_argument("--candidates-per-iter", type=int, default=2)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--num-tasks", type=int, default=None,
                   help="cap search-split tasks per domain (cost control)")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--agent-llm", default=None)
    p.add_argument("--user-llm", default=None)
    p.add_argument("--user-api-base", default=None)
    p.add_argument("--user-disable-thinking", action="store_true")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--eval-timeout", type=int, default=6 * 3600)
    p.add_argument("--nl-domains", nargs="*", default=["retail"],
                   help="domains that need the NL assertion judge "
                        "(paper: retail only)")
    p.add_argument("--h5-top-k", type=int, default=1,
                   help="H5 skills injected (paper setting: 1)")
    p.add_argument("--proposer-model", default="DeepSeek-Flash",
                   help="qoder model for the proposer (default: DeepSeek-Flash)")
    p.add_argument("--propose-timeout", type=int, default=5400)
    p.add_argument("--mock-proposer", action="store_true",
                   help="skip the qoder proposer and emit a canned no-op "
                        "candidate each iteration (plumbing smoke tests)")
    args = p.parse_args()

    if args.test and args.fresh:
        p.error("--test and --fresh are mutually exclusive")
    if args.test:
        if args.domains is None:
            history = read_jsonl(RUNS_DIR / args.run_name / "evolution_summary.jsonl")
            seen = {d for r in history for d in r.get("scores", {})}
            args.domains = sorted(seen) or ["airline", "retail", "telecom"]
        finalize(args)
    else:
        if args.domains is None:
            args.domains = ["airline", "retail", "telecom"]
        evolve(args)


if __name__ == "__main__":
    main()
