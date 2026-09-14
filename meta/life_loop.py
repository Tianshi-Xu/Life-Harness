#!/usr/bin/env python3
"""Automated Life-Harness evolution loop (the paper's own iteration method).

Candidates are self-contained plugin files (meta-harness engineering: a file
imported via --harness-plugin before environment construction), but every
intervention mounts onto the predefined H2-H5 lifecycle hooks (H2 rules /
H3 tool hints / H4 annotators / H5 skills) — the paper's structured search
space. Acceptance appends the plugin to an ordered chain
(run_dir/chain/manifest.json); rejection discards it. No source files are
ever edited, so rollback is free and every state is reproducible.

Evaluation discipline:
- search split = tau2 train split only; --test finalizes on the held-out
  test split (baseline / initial / evolved-chain), then freezes the run
- tiered evaluation: a cheap screen (failing + sentry tasks) fast-rejects;
  acceptance requires a full-split confirmation (--accept-on full, default)
  or is provisional with periodic batch confirmation (--accept-on screen)
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import json
import py_compile
import shutil
import subprocess
import sys
import time
from pathlib import Path

import proposer
from meta_harness import FORBIDDEN_REFERENCES

META_DIR = Path(__file__).resolve().parent
LIFE_HARNESS_DIR = META_DIR.parent
RUNS_DIR = META_DIR / "runs_life"
WORKTREES_DIR = META_DIR / "worktrees"
SKILL_DOC = META_DIR / "skills" / "tau2-harness" / "SKILL.md"
ZERO_LAYER = META_DIR / "zero_layer.py"

CHAIN_LOADER_TEMPLATE = '''"""Auto-generated plugin chain loader (life_loop). Do not edit.

Loads the layers listed in manifest.json (next to this file) in order and
invokes their register() hooks. Layers: {layers_comment}
"""

import importlib.util
import json
from pathlib import Path

_MANIFEST = Path(__file__).with_name("manifest.json")


def register() -> None:
    layers = json.loads(_MANIFEST.read_text())["layers"]
    for idx, path in enumerate(layers):
        spec = importlib.util.spec_from_file_location(f"life_layer_{{idx}}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        hook = getattr(module, "register", None)
        if callable(hook):
            hook()
'''

EVOLUTION_PROMPT = """You are a coding agent responsible for improving a runtime harness for a
deterministic LLM-agent environment. Your goal is to improve task performance by
adapting the runtime interface between the frozen model and the environment,
without changing model weights, benchmark tasks, or environment evaluation logic.

Inputs:
- current harness implementation: {harness_dir}
- trajectory directory from the previous iteration, including summary metrics: {trajectory_dir}
- harness design guide: {design_guide}

Inspect the previous iteration's trajectories and identify recurring failure
patterns. For each pattern, determine the earliest lifecycle point where it can
be reliably detected or prevented: before interaction, during task conditioning,
before environment execution, or after execution.

Focus on mechanically identifiable deterministic failures such as invalid action
formats, wrong tool conventions, missing required fields, repeated no-op actions,
loops, premature submissions, budget exhaustion, or recurring procedural
mistakes.

Directly implement targeted, minimal updates in the appropriate harness layer.
Do not only return an analysis report. Do not use hidden oracle information,
test labels, task modifications, environment transition changes, or
evaluation-criteria changes.

After editing, run or recommend the narrowest regression checks available.
Inspect cases where the harness may over-trigger, block a valid action, inject
misleading guidance, or reduce performance on previously successful
trajectories.

When finished, summarize:
1. dominant failure patterns found;
2. harness layer responsible for each update;
3. implemented code changes;
4. why each update is safe under the deterministic environment contract;
5. remaining failure modes to monitor next.
"""

LOOP_ADDENDUM = """
---
Loop-specific constraints (automated run — read carefully):
- Your output is ONE new self-contained plugin file at:
  {candidate_path}
  Do NOT modify any existing file — source edits have NO effect, the
  evaluation loads ONLY your plugin via --harness-plugin (imported once
  before environment construction; a callable register() is invoked after
  import).
- The plugin mount-point contract (H2 rules / H3 tool hints / H4 annotators /
  H5 skills, with real class and registry names) is documented in:
  {skill_doc}
  Read it first. Your intervention must land in code paths active under
  --enabled --h2 --h3 --h4 --h5, domain {domains}.
- The harness source at {harness_dir} is READ-ONLY reference material.
- Your plugin is evaluated ON TOP of the accepted chain — previous accepted
  plugins run first, in order, and stay active:
{chain_brief}
- You have no shell access: read files and write your plugin; recommend (not
  run) regression checks.
- ONE focused hypothesis per iteration — a small, mechanistically clear
  change. Do NOT bundle multiple independent fixes; later iterations get
  their own turns. (Bundling caused a full revert in a previous run.)
- NEVER read anything under data/ (no tasks.json, split_tasks.json, db.json,
  policy files). The held-out test split lives there and must stay invisible.
  Your only legal evidence is the trajectory directory above (search split).
- Plugin code is scanned and rejected if it contains any of: open(,
  subprocess, os.environ, requests, urllib, socket, or references to task
  data / split files / simulation logs.
- Before finishing, write {run_dir}/pending_proposal.json declaring your
  change (this drives cheap targeted screening):
  {{"name": "<lower_snake_case>", "hypothesis": "<failure mode + fix mechanism>",
    "target_domains": ["airline"|"retail"|"telecom"], "target_layer": "h2"|"h3"|"h4"|"h5"|"mixed"}}
- This is iteration {iteration}. Current search-split score: {current_score}.
  Previous iterations' outcomes: {history_brief}
"""

MOCK_CANDIDATE = '''"""Mock candidate plugin (plumbing smoke test), iteration {iteration}."""

_NOTE = "mock candidate iter {iteration} — content varies so mock evals differ"


def register() -> None:
    pass
'''


# ------------------------------------------------------------------- utils


def run_cmd(cmd: list[str], cwd: Path | None = None, check: bool = True,
            timeout: int = 1800) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None,
                          capture_output=True, text=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise RuntimeError(f"cmd failed: {cmd}\n{proc.stdout}\n{proc.stderr}")
    return proc


def load_json(path: Path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text())


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2))
    temporary.replace(path)


EVAL_OPTIONS = ("domains", "trials", "num_tasks", "concurrency", "agent_llm",
                "user_llm", "user_api_base", "user_disable_thinking",
                "max_steps", "nl_domains", "h5_top_k", "from_scratch")


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eval_config(args: argparse.Namespace) -> dict:
    config = {name: getattr(args, name, None) for name in EVAL_OPTIONS}
    # Store endpoints, never credentials. A different serving model must use
    # a new run even when its public model name stays the same.
    config["endpoints"] = {name: os.environ.get(name) for name in
                           ("AGENT_API_BASE", "USER_API_BASE")}
    config["mock_eval"] = os.environ.get("META_MOCK_EVAL") == "1"
    return config


def bind_record(path: Path, record: dict) -> None:
    if path.exists():
        if load_json(path) != record:
            raise RuntimeError(f"Configuration/code mismatch at {path}; use a new run name.")
    else:
        save_json(path, record)


def source_digest(wt: Path) -> str:
    digest = hashlib.sha256()
    for root in (wt / "TauBench" / "src", wt / "TauBench" / "scripts"):
        for path in sorted(root.rglob("*.py")):
            digest.update(str(path.relative_to(wt)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def plugin_digest(plugin: Path | None) -> dict:
    if plugin is None:
        return {}
    result = {str(plugin): file_digest(plugin)}
    manifest = plugin.with_name("manifest.json")
    if manifest.exists():
        result[str(manifest)] = file_digest(manifest)
        for layer in load_json(manifest)["layers"]:
            result[str(layer)] = file_digest(Path(layer))
    return result


def restore_task_snapshot(state: dict, snapshot: dict) -> None:
    state["task_scores"] = copy.deepcopy(snapshot)


# --------------------------------------------------------------- worktree


def setup_worktree(run_name: str, fresh: bool) -> Path:
    """Create (or reuse) an isolated checkout of the repo for this run.

    The harness source is never edited anymore (candidates are plugins); the
    worktree exists so concurrent runs get their own venv and simulation
    output directories.
    """
    wt = WORKTREES_DIR / run_name
    branch = f"life-loop/{run_name}"
    if wt.exists():
        if fresh:
            run_cmd(["git", "-C", str(LIFE_HARNESS_DIR), "worktree", "remove",
                     "--force", str(wt)])
            run_cmd(["git", "-C", str(LIFE_HARNESS_DIR), "branch", "-D", branch],
                    check=False)
        else:
            return wt
    run_cmd(["git", "-C", str(LIFE_HARNESS_DIR), "worktree", "add",
             str(wt), "-b", branch, "HEAD"])
    # eval_harness.py carries the (uncommitted) --harness-plugin feature;
    # worktrees check out committed HEAD, so sync the main checkout's copy.
    shutil.copy(LIFE_HARNESS_DIR / "TauBench" / "scripts" / "eval_harness.py",
                wt / "TauBench" / "scripts" / "eval_harness.py")
    print("[setup] uv sync in worktree (warm cache, should be fast)...")
    run_cmd(["uv", "sync"], cwd=wt / "TauBench", timeout=3600)
    return wt


def clean_worktree(wt: Path) -> list[str]:
    """Revert anything the proposer touched in the worktree (has no effect
    on evaluation, but keeps the checkout pristine). Returns file list."""
    out = run_cmd(["git", "status", "--porcelain"], cwd=wt).stdout
    # eval_harness.py is deliberately synced from the main checkout
    # (--harness-plugin feature); everything else the proposer touched goes.
    # Porcelain paths are repo-root-relative (e.g. "TauBench/scripts/...").
    synced = "TauBench/scripts/eval_harness.py"
    entries = [(l[:2], l[3:]) for l in out.splitlines() if l.strip()]
    touched = [p for _, p in entries if p != synced]
    tracked = [p for st, p in entries if st != "??" and p != synced]
    if tracked:
        run_cmd(["git", "checkout", "--"] + tracked, cwd=wt, check=False)
    return touched


# ------------------------------------------------------------------ chain


def write_loader(loader_dir: Path, layer_paths: list[Path]) -> Path:
    """Write loader.py + manifest.json for a plugin chain. Returns loader path."""
    loader_dir.mkdir(parents=True, exist_ok=True)
    comment = ", ".join(p.name for p in layer_paths) or "(empty)"
    (loader_dir / "loader.py").write_text(
        CHAIN_LOADER_TEMPLATE.replace("{layers_comment}", comment))
    save_json(loader_dir / "manifest.json",
              {"layers": [str(p) for p in layer_paths]})
    return loader_dir / "loader.py"


def chain_dir_of(run_dir: Path) -> Path:
    return run_dir / "chain"


def chain_layer_paths(run_dir: Path, state: dict) -> list[Path]:
    return [run_dir / "chain" / e["file"] for e in state["accepted"]]


def current_chain_loader(run_dir: Path, state: dict) -> Path | None:
    """Loader for the accepted chain, or None if the chain is empty."""
    paths = chain_layer_paths(run_dir, state)
    if not paths:
        return None
    return write_loader(chain_dir_of(run_dir), paths)


def seed_zero_layer(run_dir: Path, state: dict) -> None:
    """--from-scratch: layer 0 clears all released H2-H5 content."""
    chain = chain_dir_of(run_dir)
    chain.mkdir(parents=True, exist_ok=True)
    dest = chain / "layer_000_zero.py"
    if not dest.exists():
        shutil.copy(ZERO_LAYER, dest)
    if not any(e["file"] == dest.name for e in state["accepted"]):
        state["accepted"].insert(0, {
            "iteration": 0, "name": "zero_state", "file": dest.name,
            "target_layer": "mixed",
            "hypothesis": "clear all released H2-H5 content (from-scratch arm)"})
        state["confirmed_accepted"] = json.loads(json.dumps(state["accepted"]))


# -------------------------------------------------------------------- eval


def evaluate(wt: Path, run_dir: Path, tag: str, split: str,
             args: argparse.Namespace, plugin: Path | None = None,
             no_harness: bool = False,
             task_ids: dict[str, list[str]] | None = None
             ) -> tuple[dict, dict, dict] | None:
    """Eval a plugin (or the bare harness) on all domains.

    Returns (scores, save_dirs, per_tasks); scores[domain] is None for domains
    skipped by a task_ids screen.
    """
    scores, save_dirs, per_tasks = {}, {}, {}
    for domain in args.domains:
        if task_ids is not None and not task_ids.get(domain):
            scores[domain] = None
            continue
        out = run_dir / "evals" / domain / tag / ("val.json" if split == "search"
                                                  else "test.json")
        request_path = out.with_name("request.json")
        if out.exists() and not request_path.exists():
            raise RuntimeError(f"Unverified legacy cache at {out}; use a new run name.")
        bind_record(request_path, {
            "config": eval_config(args), "domain": domain, "split": split,
            "task_ids": task_ids.get(domain) if task_ids is not None else None,
            "no_harness": no_harness, "plugin": plugin_digest(plugin),
            "source": source_digest(wt),
            "benchmark": file_digest(META_DIR / "benchmark.py"),
        })
        if out.is_file():
            score = load_json(out)
        else:
            cmd = [
                sys.executable, str(META_DIR / "benchmark.py"),
                "--domain", domain, "--split", split,
                "--candidate-name", tag, "--run-tag", f"life_{run_dir.name}",
                "--out", str(out),
                "--trials", str(args.trials),
                "--concurrency", str(args.concurrency),
                "--taubench-dir", str(wt / "TauBench"),
            ]
            if no_harness:
                cmd.append("--no-harness")
            else:
                cmd += ["--flags", "h2,h3,h4,h5"]
            if plugin is not None:
                cmd += ["--candidate-file", str(plugin)]
            if task_ids is not None:
                ids = task_ids.get(domain)
                if not ids:
                    scores[domain] = None  # domain not screened this round
                    continue
                id_file = run_dir / "evals" / domain / tag / "task_ids.txt"
                id_file.parent.mkdir(parents=True, exist_ok=True)
                id_file.write_text("\n".join(ids) + "\n")
                cmd += ["--task-id-file", str(id_file)]
            for opt in ("num_tasks", "agent_llm", "user_llm", "user_api_base",
                        "max_steps", "eval_timeout", "h5_top_k"):
                value = getattr(args, opt)
                if value is not None:
                    cmd += [f"--{opt.replace('_', '-')}", str(value)]
            if getattr(args, "user_disable_thinking", False):
                cmd.append("--user-disable-thinking")
            if domain in (getattr(args, "nl_domains", None) or []):
                cmd.append("--nl")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0 or not out.is_file():
                print(f"  [eval-error] {domain}/{tag}: "
                      f"{proc.stdout.strip()} {proc.stderr.strip()}")
                return None
            score = load_json(out)
        scores[domain] = score["accuracy"]
        per_tasks[domain] = score.get("per_task") or {}
        if score.get("save_dir"):
            save_dirs[domain] = score["save_dir"]
    return scores, save_dirs, per_tasks


def mean_score(scores: dict) -> float | None:
    vals = [v for v in scores.values() if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


# --------------------------------------------------------------- validation


def smoke_check(wt: Path, plugin: Path) -> tuple[bool, str]:
    """py_compile + import + register() inside the worktree venv."""
    try:
        py_compile.compile(str(plugin), doraise=True)
    except py_compile.PyCompileError as e:
        return False, f"py_compile: {e}"
    code = (
        "import importlib.util;"
        f"spec=importlib.util.spec_from_file_location('staged', r'{plugin}');"
        "m=importlib.util.module_from_spec(spec);"
        "spec.loader.exec_module(m);"
        "r=getattr(m,'register',None);"
        "r() if callable(r) else None;"
        "print('smoke ok')"
    )
    try:
        proc = subprocess.run(["uv", "run", "python", "-c", code],
                              cwd=str(wt / "TauBench"), capture_output=True,
                              text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return False, "import timed out"
    if proc.returncode != 0:
        return False, proc.stderr.strip()[-500:]
    return True, ""


def validate_candidate(wt: Path, candidate: Path, staged: Path) -> str:
    """Return a reject reason, or '' if the candidate is evaluable.

    Scans the candidate text for forbidden references, then smoke-checks the
    staged chain loader (accepted chain + candidate) so composition errors
    are caught before spending eval budget.
    """
    if not candidate.is_file():
        return f"missing file {candidate}"
    text = candidate.read_text()
    hit = next((s for s in FORBIDDEN_REFERENCES if s in text), None)
    if hit:
        return f"forbidden reference: {hit!r}"
    ok, err = smoke_check(wt, staged)
    return "" if ok else f"smoke: {err}"


# ------------------------------------------------------- tiered evaluation


def domains_of_plugin(candidate: Path, declared: list[str],
                      all_domains: list[str]) -> list[str]:
    """Infer affected domains from the plugin text + the declared manifest.

    A plugin importing a domain module affects that domain; plugins touching
    only shared machinery (or nothing identifiable) are screened everywhere.
    """
    text = candidate.read_text() if candidate.is_file() else ""
    found = {d for d in all_domains
             if f"harness.{d}" in text or f"harness import {d}" in text}
    domains = set(declared) | found
    return sorted(domains) or list(all_domains)


def select_screen_tasks(task_scores: dict[str, float],
                        sentry_n: int) -> list[str]:
    """Failed tasks (re-measured) + a deterministic sentry sample of passing
    tasks (regression guard)."""
    failing = sorted(t for t, r in task_scores.items() if r < 1.0)
    passing = sorted(t for t, r in task_scores.items() if r >= 1.0)
    step = max(1, len(passing) // max(sentry_n, 1))
    sentry = passing[::step][:sentry_n]
    return failing + sentry


def merge_task_scores(cache: dict, domain: str, per_task: dict) -> None:
    cache.setdefault(domain, {}).update(
        {str(t): float(r) for t, r in per_task.items()})


def estimate_mean(cache: dict, domains: list[str]) -> float | None:
    vals = []
    for d in domains:
        ts = cache.get(d, {})
        if not ts:
            return None
        vals.append(sum(ts.values()) / len(ts))
    return round(sum(vals) / len(vals), 4)


# -------------------------------------------------------------------- loop


def render_prompt(iteration: int, wt: Path, run_dir: Path,
                  trajectory_dirs: dict, args: argparse.Namespace,
                  state: dict) -> str:
    history_brief = "; ".join(
        f"iter{h['iteration']}:{h['decision']}({h.get('mean_after')})"
        for h in state["history"][-5:]
    ) or "none yet"
    chain_brief = []
    for e in state["accepted"]:
        chain_brief.append(
            f"  - {e['file']} [{e.get('target_layer', '?')}] "
            f"{(e.get('hypothesis') or '')[:150]}")
    chain_brief = "\n".join(chain_brief) or "  (empty — you are the first)"
    tried_lines = []
    for h in state["history"]:
        p = h.get("proposal") or {}
        if p.get("name"):
            tried_lines.append(
                f"- iter{h['iteration']} [{h['decision']}] "
                f"{p.get('name')} ({p.get('target_layer') or '?'}): "
                f"{(p.get('hypothesis') or '').strip()[:200]} — candidate: "
                f"{run_dir}/candidates/iter_{h['iteration']:02d}.py")
    tried = ("\n\nPreviously attempted changes (rejected approaches must NOT be "
             "retried unchanged or with cosmetic variations — if your planned "
             "plugin is functionally the same as a rejected one, pick a "
             "DIFFERENT hypothesis; read the candidate files if unsure):\n"
             + "\n".join(tried_lines)) if tried_lines else ""
    traj_lines = "\n".join(f"  {d}: {p}" for d, p in trajectory_dirs.items())
    prompt = EVOLUTION_PROMPT.format(
        harness_dir=str(wt / "TauBench" / "src" / "tau2" / "harness"),
        trajectory_dir="\n" + traj_lines if traj_lines else "(iteration 0 state; see summary metrics in the evals dir)",
        design_guide=str(wt / "TauBench" / "Harness.md"),
    )
    prompt += LOOP_ADDENDUM.format(
        iteration=iteration,
        domains=", ".join(args.domains),
        harness_dir=str(wt / "TauBench" / "src" / "tau2" / "harness"),
        candidate_path=run_dir / "candidates" / f"iter_{iteration:02d}.py",
        skill_doc=SKILL_DOC,
        run_dir=run_dir,
        chain_brief=chain_brief,
        current_score=state.get("current_mean"),
        history_brief=history_brief,
    )
    prompt += tried
    experience = run_dir / "search_experience.md"
    if experience.is_file():
        prompt += "\nPrior search-only learning notes:\n" + experience.read_text()
    if state.get("from_scratch"):
        prompt += "\nZERO CONTENT MODE: released H2-H5 registries are empty; create entries, do not assume existing hints or rules.\n"
    if not trajectory_dirs:
        prompt += (
            f"\nNo trajectories yet (this is the initial harness state). Initial "
            f"search-split scores per domain are in "
            f"{run_dir}/evals/<domain>/iter_000/val.json — start from "
            f"the harness code and known failure modes of a frozen small model.\n"
        )
        if state.get("from_scratch"):
            prompt += (
                "\nIMPORTANT: a zero layer is active as layer 0 — it has CLEARED "
                "all released H2-H5 content (empty harness_rules / "
                "harness_annotators / H3 hints / skill bank). The hook machinery "
                "still works; there is simply no existing content to tune. You "
                "are building the harness from zero: read base.py for the "
                "HarnessRule/HarnessAnnotator protocols and register real "
                "content via your plugin.\n"
            )
        else:
            prompt += (
                "\nThe released h2345 harness content is active. Your plugin may "
                "APPEND to the registries or OVERRIDE existing entries (e.g. "
                "replace a hint's text) — prefer appending unless an existing "
                "entry is the problem.\n"
            )
    return prompt


def full_confirm(wt: Path, run_dir: Path, state: dict,
                 args: argparse.Namespace, iteration: int) -> None:
    """Full-split confirmation eval of the current chain; on regression,
    batch-rolls back to the last confirmed chain."""
    print(f"  [full] confirmation eval after {state['since_full']} "
          f"provisional accepts")
    loader = current_chain_loader(run_dir, state)
    result = evaluate(wt, run_dir, f"iter_{iteration:03d}_full", "search", args,
                      plugin=loader)
    if result is None:
        raise RuntimeError("Full confirmation failed; resume to retry before further evolution.")
    _, save_dirs, per_tasks = result
    for d, pt in per_tasks.items():
        merge_task_scores(state["task_scores"], d, pt)
    true_mean = estimate_mean(state["task_scores"], args.domains)
    print(f"  [full] true mean={true_mean} (confirmed {state['confirmed_mean']})")
    if true_mean is not None and true_mean >= state["confirmed_mean"]:
        state["confirmed_mean"] = true_mean
        state["current_mean"] = true_mean
        state["confirmed_accepted"] = json.loads(json.dumps(state["accepted"]))
        state["confirmed_task_scores"] = json.loads(
            json.dumps(state["task_scores"]))
        state["last_save_dirs"] = save_dirs
        print("  [full] confirmed new best")
    else:
        dropped = [e["name"] for e in state["accepted"]
                   if e not in state["confirmed_accepted"]]
        print(f"  [full] REGRESSION vs confirmed best — rolling back chain, "
              f"dropping: {dropped}")
        state["accepted"] = json.loads(
            json.dumps(state["confirmed_accepted"]))
        current_chain_loader(run_dir, state)  # regenerate loader
        state["task_scores"] = json.loads(
            json.dumps(state["confirmed_task_scores"]))
        state["current_mean"] = state["confirmed_mean"]
    state["since_full"] = 0


def evolve(args: argparse.Namespace) -> None:
    run_dir = RUNS_DIR / args.run_name
    if args.fresh and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "candidates").mkdir(exist_ok=True)
    (run_dir / "chain").mkdir(exist_ok=True)
    if (run_dir / "finalized.json").exists():
        raise RuntimeError("This run has entered held-out finalization and is frozen.")
    if (run_dir / "state.json").exists() and not (run_dir / "config.json").exists():
        raise RuntimeError("Legacy run has no configuration fingerprint; use a new run name.")
    bind_record(run_dir / "config.json", eval_config(args))
    wt = setup_worktree(args.run_name, args.fresh)
    bind_record(run_dir / "source.json", {"source": source_digest(wt),
                "loop": file_digest(Path(__file__)),
                "benchmark": file_digest(META_DIR / "benchmark.py")})
    state_path = run_dir / "state.json"
    state = load_json(state_path, default={
        "accepted": [],             # ordered chain entries (incl. zero layer)
        "confirmed_accepted": [],   # snapshot at last full confirmation
        "current_mean": None,       # latest estimate (screen-updated)
        "confirmed_mean": None,     # last full-eval ground truth
        "task_scores": {},          # per-domain per-task reward cache
        "confirmed_task_scores": {},  # snapshot at last confirmed best
        "iteration": 0,
        "since_full": 0,
        "history": [],
        "from_scratch": getattr(args, "from_scratch", False),
    })
    if state["from_scratch"] and not state["accepted"]:
        seed_zero_layer(run_dir, state)
        save_json(state_path, state)

    # Iteration 0: full eval of the initial state to seed the task cache.
    if state["current_mean"] is None:
        print("== iteration 0: initial state (full eval) ==")
        loader = current_chain_loader(run_dir, state)
        result = evaluate(wt, run_dir, "iter_000", "search", args, plugin=loader)
        if result is None:
            sys.exit("initial evaluation failed — check services")
        scores, save_dirs, per_tasks = result
        for d, pt in per_tasks.items():
            merge_task_scores(state["task_scores"], d, pt)
        m = estimate_mean(state["task_scores"], args.domains)
        state.update(current_mean=m, confirmed_mean=m,
                     confirmed_task_scores=json.loads(
                         json.dumps(state["task_scores"])),
                     last_save_dirs=save_dirs)
        save_json(state_path, state)
        print(f"  initial scores: {scores} mean={m}")

    if state["since_full"] > 0:
        full_confirm(wt, run_dir, state, args, state["iteration"])
        save_json(state_path, state)

    for iteration in range(state["iteration"] + 1, args.iterations + 1):
        print(f"\n== iteration {iteration}/{args.iterations} ==")
        candidate = run_dir / "candidates" / f"iter_{iteration:02d}.py"
        proposal_path = run_dir / "prompts" / f"iter_{iteration:02d}_proposal.json"
        if candidate.exists() and proposal_path.exists():
            print("  [resume] reusing the saved candidate and proposal")
            proposal = load_json(proposal_path)
        else:
            prompt = render_prompt(iteration, wt, run_dir,
                                   state.get("last_save_dirs", {}), args, state)
            (run_dir / "prompts").mkdir(exist_ok=True)
            (run_dir / "prompts" / f"iter_{iteration:02d}.md").write_text(prompt)

            pending_proposal = run_dir / "pending_proposal.json"
            if pending_proposal.exists():
                pending_proposal.unlink()
            if args.mock_proposer:
                candidate.write_text(MOCK_CANDIDATE.format(iteration=iteration))
                pending_proposal.write_text(json.dumps({
                    "name": f"mock_iter_{iteration}", "hypothesis": "plumbing test",
                    "target_domains": list(args.domains), "target_layer": "h4"}))
            else:
                proposer.run(
                    prompt=prompt, cwd=run_dir,
                    model=args.proposer_model, timeout=args.propose_timeout,
                    log_path=run_dir / "qoder_logs" / f"iter_{iteration:02d}.json",
                    add_dirs=[wt / "TauBench", SKILL_DOC.parent],
                )
                if not candidate.is_file():
                    proposer.run(
                        prompt=(f"You finished without writing {candidate}. "
                                "Write it now, following the plugin contract from "
                                "the task instructions."),
                        cwd=run_dir, model=args.proposer_model,
                        timeout=1200,
                        log_path=run_dir / "qoder_logs" / f"iter_{iteration:02d}_nudge.json",
                        add_dirs=[wt / "TauBench", SKILL_DOC.parent],
                    )
            touched = clean_worktree(wt)
            if touched:
                print(f"  [guard] worktree source edits reverted (no effect): "
                      f"{touched[:5]}")
            proposal = load_json(pending_proposal, default={}) or {}
            save_json(proposal_path, proposal)
            if pending_proposal.exists():
                pending_proposal.unlink()

        staged = write_loader(run_dir / "staged",
                              chain_layer_paths(run_dir, state) + [candidate])
        reason = validate_candidate(wt, candidate, staged)
        if reason:
            print(f"  [invalid] {reason}; skipping eval")
            state["history"].append({
                "iteration": iteration, "decision": "invalid",
                "proposal": {k: proposal.get(k) for k in
                             ("name", "hypothesis", "target_layer")},
                "reason": reason, "mean_after": state["current_mean"]})
            state["iteration"] = iteration
            save_json(state_path, state)
            continue

        # Candidate measurements must never leak into the incumbent cache.
        task_snapshot = copy.deepcopy(state["task_scores"])
        # ---- screen: affected domains x (failing + sentry) tasks
        declared = [d for d in proposal.get("target_domains", [])
                    if d in args.domains]
        screen_domains = domains_of_plugin(candidate, declared, args.domains)
        screen_ids = {}
        for d in screen_domains:
            ids = select_screen_tasks(state["task_scores"].get(d, {}),
                                      args.screen_sentry)
            if ids:
                screen_ids[d] = ids
        screen_is_full = not screen_ids
        if screen_is_full:
            print("  [screen] no cached tasks to screen on; full eval instead")
            result = evaluate(wt, run_dir, f"iter_{iteration:03d}", "search",
                              args, plugin=staged)
        else:
            n_tasks = sum(len(v) for v in screen_ids.values())
            print(f"  [screen] domains={screen_domains} tasks={n_tasks} "
                  f"(proposal: {proposal.get('name', '?')})")
            result = evaluate(wt, run_dir, f"iter_{iteration:03d}_screen",
                              "search", args, plugin=staged,
                              task_ids=screen_ids)
        if result is None:
            print("  [eval-error] discarding candidate")
            state["history"].append({"iteration": iteration,
                                     "decision": "eval_error_reject",
                                     "mean_after": None})
            state["iteration"] = iteration
            save_json(state_path, state)
            continue
        _, save_dirs, per_tasks = result
        for d, pt in per_tasks.items():
            merge_task_scores(state["task_scores"], d, pt)
        est = estimate_mean(state["task_scores"], args.domains)
        print(f"  estimate mean={est} (current {state['current_mean']})")

        def reject(kind: str) -> str:
            restore_task_snapshot(state, task_snapshot)
            return f"reject({kind})"

        if est is not None and est >= state["current_mean"]:
            if args.accept_on == "screen" and not screen_is_full:
                name = proposal.get("name") or f"iter_{iteration}"
                layer_file = f"layer_{iteration:03d}_{name}.py"
                shutil.copy(candidate, run_dir / "chain" / layer_file)
                state["accepted"].append({
                    "iteration": iteration, "name": name, "file": layer_file,
                    "target_layer": proposal.get("target_layer"),
                    "hypothesis": proposal.get("hypothesis")})
                current_chain_loader(run_dir, state)
                decision = "accept(screen)"
                state["current_mean"] = est
                state["last_save_dirs"] = save_dirs or state.get("last_save_dirs", {})
                state["since_full"] += 1
            else:
                # Screen passed the fast-reject gate (or was already a full
                # eval); acceptance requires full-split confirmation — screens
                # re-measure the very failures the edit targeted (optimistic).
                if screen_is_full:
                    true_mean = est
                else:
                    print("  [screen] passed — full confirmation eval")
                    result = evaluate(wt, run_dir, f"iter_{iteration:03d}",
                                      "search", args, plugin=staged)
                    if result is None:
                        restore_task_snapshot(state, task_snapshot)
                        print("  [eval-error] discarding candidate")
                        state["history"].append({
                            "iteration": iteration,
                            "decision": "eval_error_reject",
                            "mean_after": None})
                        state["iteration"] = iteration
                        save_json(state_path, state)
                        continue
                    _, save_dirs, per_tasks = result
                    for d, pt in per_tasks.items():
                        merge_task_scores(state["task_scores"], d, pt)
                    true_mean = estimate_mean(state["task_scores"], args.domains)
                if true_mean is not None and true_mean >= state["confirmed_mean"]:
                    name = proposal.get("name") or f"iter_{iteration}"
                    layer_file = f"layer_{iteration:03d}_{name}.py"
                    shutil.copy(candidate, run_dir / "chain" / layer_file)
                    state["accepted"].append({
                        "iteration": iteration, "name": name, "file": layer_file,
                        "target_layer": proposal.get("target_layer"),
                        "hypothesis": proposal.get("hypothesis")})
                    current_chain_loader(run_dir, state)
                    decision = "accept(full)"
                    state.update(
                        current_mean=true_mean, confirmed_mean=true_mean,
                        since_full=0,
                        confirmed_accepted=json.loads(
                            json.dumps(state["accepted"])),
                        confirmed_task_scores=json.loads(
                            json.dumps(state["task_scores"])),
                        last_save_dirs=save_dirs)
                else:
                    decision = reject("full")
        else:
            decision = reject("screen")
        state["history"].append({
            "iteration": iteration, "decision": decision,
            "proposal": {k: proposal.get(k) for k in
                         ("name", "hypothesis", "target_layer")},
            "screen_domains": screen_domains, "mean_estimate": est,
            "mean_after": state["current_mean"],
        })
        state["iteration"] = iteration
        save_json(state_path, state)
        print(f"  [{decision}] estimate mean: {state['current_mean']}")

        # ---- periodic full confirmation with batch rollback
        if state["since_full"] >= args.full_every:
            full_confirm(wt, run_dir, state, args, iteration)
            save_json(state_path, state)

    # Final confirmation: never leave provisional accepts unconfirmed.
    if state["since_full"] > 0:
        print("\n== final confirmation ==")
        full_confirm(wt, run_dir, state, args, state["iteration"])
        save_json(state_path, state)

    print(f"\nevolution done. accepted chain: "
          f"{[e['name'] for e in state['accepted']]}")
    print(f"finalize with: python {META_DIR}/life_loop.py --test "
          f"--run-name {args.run_name}")


def finalize(args: argparse.Namespace) -> None:
    run_dir = RUNS_DIR / args.run_name
    state = load_json(run_dir / "state.json")
    if not state:
        sys.exit(f"no such run: {run_dir}")
    wt = WORKTREES_DIR / args.run_name
    from_scratch = state.get("from_scratch", False)
    bind_record(run_dir / "config.json", eval_config(args))
    if state.get("since_full", 0):
        raise RuntimeError("Confirm provisional layers before finalization.")
    save_json(run_dir / "finalized.json", {"status": "in_progress"})

    # baseline: harness disabled
    result = evaluate(wt, run_dir, "final_baseline", "test", args,
                      no_harness=True)
    if result is None:
        raise RuntimeError("Baseline test failed; finalization remains incomplete.")
    if result:
        print(f"  [test] baseline: {result[0]} mean={mean_score(result[0])}")

    # initial state: released h2345 (no plugin) or the zero layer alone
    if from_scratch:
        initial_loader = write_loader(
            run_dir / "final_initial",
            [run_dir / "chain" / state["accepted"][0]["file"]]
            if state["accepted"] else [])
    else:
        initial_loader = None
    result = evaluate(wt, run_dir, "final_initial", "test", args,
                      plugin=initial_loader)
    if result is None:
        raise RuntimeError("Initial test failed; finalization remains incomplete.")
    if result:
        print(f"  [test] initial: {result[0]} mean={mean_score(result[0])}")

    evolved = [e for e in state["accepted"]
               if e["name"] != "zero_state"]
    if not evolved:
        print("  [test] no accepted iteration beyond the initial state; skip")
    else:
        loader = current_chain_loader(run_dir, state)
        result = evaluate(wt, run_dir, "final_evolved", "test", args,
                          plugin=loader)
        if result is None:
            raise RuntimeError("Evolved test failed; finalization remains incomplete.")
        if result:
            print(f"  [test] evolved: {result[0]} mean={mean_score(result[0])}")
    save_json(run_dir / "finalized.json",
              {"status": "complete", "completed_at": time.time()})
    print(f"finalized. test results under {run_dir}/evals/*/final_*/")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--run-name", default="life_pilot")
    p.add_argument("--fresh", action="store_true",
                   help="recreate the worktree from HEAD and restart")
    p.add_argument("--test", action="store_true",
                   help="evaluate baseline/initial/evolved on held-out test split")
    p.add_argument("--domains", nargs="+", default=["airline"],
                   choices=["airline", "retail", "telecom"])
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--num-tasks", type=int, default=None)
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
    p.add_argument("--proposer-model", default="DeepSeek-Flash")
    p.add_argument("--propose-timeout", type=int, default=5400)
    p.add_argument("--mock-proposer", action="store_true",
                   help="write a mock plugin instead of calling qoder "
                        "(plumbing smoke tests)")
    p.add_argument("--from-scratch", action="store_true",
                   help="start from the zero harness state (a layer-0 plugin "
                        "clears all released H2-H5 content) instead of the "
                        "released h2345 content — the fair-comparison arm")
    p.add_argument("--screen-sentry", type=int, default=8,
                   help="passing tasks sampled per screened domain as a "
                        "regression sentry (screen = failing + sentry)")
    p.add_argument("--full-every", type=int, default=3,
                   help="run a full confirmation eval after this many "
                        "accepted screen edits (batch rollback on regression)")
    p.add_argument("--accept-on", choices=["full", "screen"], default="full",
                   help="full: confirm every screen-pass on the full split "
                        "before accepting (default). screen: provisionally "
                        "accept on the screen estimate and confirm in batches "
                        "(cheaper, batch rollback on regression).")
    args = p.parse_args()

    if args.test and args.fresh:
        p.error("--test and --fresh are mutually exclusive")
    if args.test:
        finalize(args)
    else:
        evolve(args)


if __name__ == "__main__":
    main()
