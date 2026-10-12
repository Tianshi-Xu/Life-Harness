import os
"""Native OS execution for identity/Meta/Life; solver accounting stays separate."""
import argparse
import ast
import asyncio
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import json
import multiprocessing
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

from evolution import harness_hooks as hooks
from .native import ROOT, close_controller, task
from .protocol import digest, save, verify_sources

TASK = INTERFACE = TOKENIZER = PLUGIN = SOLVER_BASE = None


def trace_events(native):
    """Normalize native per-layer audit lists to the public evidence event format."""
    if isinstance(native, dict):
        return [{**copy.deepcopy(event), 'layer': layer} for layer in ('h2','h3','h4','h5')
                for event in native.get(layer, []) if isinstance(event, dict)]
    return copy.deepcopy(native) if isinstance(native, list) else []


def validate_meta(path):
    tree = ast.parse(Path(path).read_text())
    safe = {"re", "json", "copy", "math", "collections", "difflib", "string", "typing", "random",
            "ast", "dataclasses", "functools", "itertools", "statistics", "datetime", "__future__"}
    forbidden = {"open", "eval", "exec", "compile", "__import__", "globals", "locals", "vars",
                 "input", "breakpoint"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name not in safe for a in node.names):
            raise ValueError("Meta candidate import outside pure stdlib")
        if isinstance(node, ast.ImportFrom) and (node.level or node.module not in safe or any(
                a.name == "*" or a.name.startswith("__") for a in node.names)):
            raise ValueError("Meta candidate import outside pure stdlib")
        if isinstance(node, ast.Name) and node.id in forbidden:
            raise ValueError("Meta candidate runtime introspection/I/O forbidden")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("__") or node.attr in {"_os", "_sys"}):
            raise ValueError("Meta candidate dunder access forbidden")
    spec = importlib.util.spec_from_file_location("os_meta_" + digest(path)[:16], path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    harness = module.AgentHarness(None)
    if not all(callable(getattr(harness, name, None)) for name in ("transform_request", "transform_response")):
        raise ValueError("missing Meta request/response interface")
    return module


def initialize(split, manifest, plugin, solver_base):
    global TASK, INTERFACE, TOKENIZER, PLUGIN, SOLVER_BASE
    TASK, INTERFACE = task(split, manifest=manifest)
    PLUGIN = validate_meta(plugin) if plugin else None
    SOLVER_BASE = solver_base.rstrip("/")
    from transformers import AutoTokenizer
    TOKENIZER = AutoTokenizer.from_pretrained(os.environ.get("LIFE_SOLVER_TOKENIZER", "Qwen/Qwen3-4B"), local_files_only=True)


class ContextLimit(ValueError):
    pass


class Session:
    def __init__(self, index):
        self.history, self.calls, self.rewards, self.turn = [], [], [], 0
        self.tools = copy.deepcopy(TASK.tools)
        self.context = {"description": TASK.problem_configs[index]["config"].description}
        self.harness = PLUGIN.AgentHarness(copy.deepcopy(self.context)) if PLUGIN else None
        self.controller = SimpleNamespace(env_output=SimpleNamespace(history=[]))

    def set_tools(self, tools):
        self.tools = copy.deepcopy(tools)

    def set_full_history(self, enabled):
        pass

    def cover(self, messages):
        self.history = copy.deepcopy(messages)

    def inject(self, item):
        if isinstance(item, list):
            for value in item:
                self.inject(value)
        elif isinstance(item, dict) and "role" in item:
            self.history.append(copy.deepcopy(item))
        elif hasattr(item, "reward"):
            self.rewards.append(item.model_dump(mode="json"))

    async def action(self):
        import requests
        self.turn += 1
        fixed = {"model": "qwen3-4b", "max_tokens": 4096, "temperature": 0.0,
                 "chat_template_kwargs": {"enable_thinking": False}}
        body = {**fixed, "messages": copy.deepcopy(self.history), "tools": copy.deepcopy(self.tools),
                "tool_choice": "auto", "parallel_tool_calls": False, "stream": False}
        if self.harness:
            body = self.harness.transform_request(copy.deepcopy(body))
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not isinstance(body.get("tools"), list):
            raise ValueError("invalid Meta transformed request")
        body.update(copy.deepcopy(fixed), stream=False, parallel_tool_calls=False)
        tokens = TOKENIZER.apply_chat_template(body["messages"], tools=body["tools"], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False)
        if len(tokens) + 4096 > 32768:
            raise ContextLimit("solver_context_limit_32768")
        call = {"turn": self.turn, "remaining": TASK.round_limit - self.turn + 1,
                "request": copy.deepcopy(body)}
        self.calls.append(call)
        for attempt in range(3):
            try:
                response = requests.post(SOLVER_BASE + "/chat/completions", json=body,
                            headers={"Authorization": "Bearer EMPTY"}, timeout=600)
                if response.status_code == 400 and "context" in response.text.lower():
                    raise ContextLimit("solver_context_limit_32768")
                response.raise_for_status()
                data = response.json()
                if not data.get("choices"):
                    raise ValueError("solver response missing choices")
                break
            except ContextLimit:
                raise
            except (requests.RequestException, ValueError) as exc:
                call.setdefault("transport_errors", []).append({"attempt": attempt + 1, "type": type(exc).__name__})
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        raw = data["choices"][0]["message"]
        message = {k: copy.deepcopy(v) for k, v in raw.items() if k in {"role", "content", "tool_calls"} and v is not None}
        message.setdefault("role", "assistant")
        if self.harness:
            message = self.harness.transform_response(copy.deepcopy(message), copy.deepcopy(self.history))
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise ValueError("invalid Meta transformed response")
        call.update(response=data, usage=data.get("usage", {}), transformed=copy.deepcopy(message))
        self.history.append(copy.deepcopy(message))
        return SimpleNamespace(messages=[message])


def episode(index):
    session = Session(index)
    started = time.monotonic()

    async def run():
        try:
            return await TASK.start_sample(index, session)
        finally:
            await close_controller(TASK)

    try:
        result = asyncio.run(run())
        status = result.status.value
        private = result.result or {}
        infra = status.replace(" ", "_") in {"unknown", "task_error", "cancelled", "running"}
        success = int(private.get("result") == 1 and not infra)
        row = {"status": status, "success": success, "infrastructure_error": infra,
               "harness_trace": trace_events(private.get("harness_trace", {}))}
        if infra and "solver_context_limit_32768" in str(private.get("error", "")):
            row.update(status="context_limit", success=0, infrastructure_error=False)
    except ContextLimit:
        row = {"status": "context_limit", "success": 0, "infrastructure_error": False}
    except Exception:
        row = {"status": "infrastructure_error", "success": 0, "infrastructure_error": True,
               "exception": traceback.format_exc()}
    if any(call.get("transport_errors") and "response" not in call for call in session.calls):
        row.update(success=0, infrastructure_error=True)
    row.update(index=index, public_task_context=session.context, history=session.history, calls=session.calls,
               turns=session.turn, turn_limit=TASK.round_limit, rewards=session.rewards,
               tokens=sum(call.get("usage", {}).get("total_tokens", 0) for call in session.calls),
               wall_seconds=time.monotonic()-started, interface=INTERFACE)
    return row


def evaluate(root, method, split, phase, ids, solver_base, *, manifest=None, plugin=None):
    root = Path(root).resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    verify_sources(protocol["native_source_hashes"])
    if method not in {"meta", "life"} or (manifest is not None) != (method == "life") or (plugin is not None) != (method == "meta"):
        raise ValueError("explicit method and candidate required")
    if split == "test":
        seal = json.loads((root / "training_seal.json").read_text())
        if phase != "final_test" or ids != protocol["native_spec"]["test_ids"]:
            raise ValueError("only complete final TEST permitted")
        if seal["completed_rounds"] != {m: 6 for m in protocol["methods"]}:
            raise ValueError("all eighteen optimization rounds must finish before TEST")
        if seal["candidates"][method] != digest(manifest or plugin):
            raise ValueError("TEST candidate differs from sealed TRAIN selection")
        for state_path, sha in seal["state_hashes"].items():
            if digest(state_path) != sha:
                raise ValueError("training state changed after seal")
    elif split != "train" or not ids or not set(ids) <= {i for b in protocol["batches"] for i in b["task_ids"]}:
        raise ValueError("native TRAIN task outside frozen schedule")
    if len(set(ids)) != len(ids) or any(type(i) is not int for i in ids):
        raise ValueError("duplicate or invalid native task IDs")
    out = root / method / "evals" / phase
    if out.exists():
        raise FileExistsError("never blindly replay a partial or paid evaluation")
    config = {"domain": "os", "method": method, "split": split, "phase": phase, "ids": ids,
              "candidate_sha256": digest(manifest or plugin), "protocol_sha256": digest(root / "protocol.json"),
              "evaluator_sha256": digest(__file__), "interface_version": hooks.INTERFACE_VERSION,
              "workers": 50, "max_context": 32768, "max_output": 4096, "enable_thinking": False}
    save(out / "config.json", config)
    with ProcessPoolExecutor(max_workers=min(50, len(ids)), mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize, initargs=(split, str(manifest) if manifest else None,
                                             str(plugin) if plugin else None, solver_base)) as pool:
        futures = {pool.submit(episode, index): index for index in ids}
        for future in as_completed(futures):
            row = future.result()
            if row["index"] != futures[future]:
                raise ValueError("native task identity mismatch")
            save(out / f"episode_{row['index']}.json", row)
            print(json.dumps({"method": method, "phase": phase, **{k: row[k] for k in
                             ("index", "status", "success", "infrastructure_error")}}), flush=True)
    verify_sources(protocol["native_source_hashes"])
    if config["candidate_sha256"] != digest(manifest or plugin):
        raise ValueError("candidate changed during evaluation")
    rows = [json.loads((out / f"episode_{i}.json").read_text()) for i in ids]
    summary = {**config, "n": len(ids), "successes": sum(r["success"] for r in rows),
               "accuracy": sum(r["success"] for r in rows)/len(ids),
               "infra_errors": sum(r["infrastructure_error"] for r in rows),
               "per_task": {str(r["index"]): r["success"] for r in rows},
               "tokens": sum(r["tokens"] for r in rows), "turns": sum(r["turns"] for r in rows)}
    if manifest:
        _, records = hooks.manifest_records(manifest)
        summary["chain_hashes"] = {"manifest": digest(manifest), **{r["artifact_path"]: r["sha256"] for r in records}}
    save(out / "summary.json", summary)
    return summary, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--method", choices=("meta", "life"), required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--solver-base", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--plugin", type=Path)
    args = parser.parse_args()
    score, _ = evaluate(args.root, args.method, args.split, args.phase, json.loads(args.ids),
                         args.solver_base, manifest=args.manifest, plugin=args.plugin)
    print(json.dumps(score))


if __name__ == "__main__":
    main()
