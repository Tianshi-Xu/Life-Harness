"""Frozen-task AgentBench evaluator using native FourHookSession composition."""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import multiprocessing
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agentbench_chain import inject_task_module, layer_paths, manifest_records
from protocol import (DOMAINS, HERE, HISTORICAL_ROOT, ROOT, ROUNDS, atomic_json,
                      digest, object_digest, task_ids, validate_protocol)
from evolution.life_optimizer_v25.context import ContextLimit

AGENTBENCH = ROOT / "AgentBench"
TASK = None
DOMAIN = None
MANIFEST_PATH: Path | None = None
TOKENIZER = None
HISTORICAL = validate_protocol(check_files=False)


class TokenizerProxy:
    """Use the core tokenizer without changing ALFWorld's existing packages."""
    def __init__(self):
        self.process = subprocess.Popen([str(AGENTBENCH/'.native/core/bin/python'),'-u',
            str(HERE/'tokenizer_worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            text=True,bufsize=1)

    def apply_chat_template(self,messages,**kwargs):
        self.process.stdin.write(json.dumps({'messages':messages,'tools':kwargs['tools']})+'\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Exact local tokenizer process exited')
        value = json.loads(line)
        if 'error' in value:
            raise ValueError(value['error'])
        return range(value['length'])


def chain_hashes(path: Path) -> dict[str, str]:
    root, records = manifest_records(path)
    return {"manifest": digest(path.resolve()),
            **{record["artifact_path"]: digest(root / record["artifact_path"])
               for record in records}}


def _episode_valid(path: Path, index: Any, config_sha256: str) -> bool:
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return (row.get("episode_schema_version") == 1 and row.get("index") == index
            and row.get("config_sha256") == config_sha256
            and row.get("success") in (0, 1)
            and isinstance(row.get("infrastructure_error"), bool))


def _jsonable(value: Any) -> Any:
    def convert(item: Any) -> Any:
        dump = getattr(item, "model_dump", None)
        if callable(dump):
            return dump(mode="json")
        if isinstance(item, SimpleNamespace):
            return vars(item)
        return str(item)
    return json.loads(json.dumps(value, default=convert))


def validate_cached_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    config_path = output_dir / "config.json"
    if not config_path.is_file():
        raise RuntimeError("cached evaluation configuration is missing")
    config = json.loads(config_path.read_text())
    config_body = {key: value for key, value in config.items() if key != "config_sha256"}
    if (config.get("config_sha256") != object_digest(config_body)
            or any(summary.get(key) != value for key, value in config.items())):
        raise RuntimeError("cached summary configuration mismatch")
    ids = config["ids"]
    if any(not _episode_valid(output_dir / f"episode_{index}.json", index,
                              config["config_sha256"]) for index in ids):
        raise RuntimeError("cached summary contains invalid episodes")
    rows = [json.loads((output_dir / f"episode_{index}.json").read_text()) for index in ids]
    expected = {"n": len(rows), "successes": sum(row["success"] for row in rows),
                "accuracy": sum(row["success"] for row in rows) / len(rows),
                "infra_errors": sum(row["infrastructure_error"] for row in rows),
                "tokens": sum(row["tokens"] for row in rows),
                "turns": sum(row["turns"] for row in rows),
                "per_task": {str(row["index"]): row["success"] for row in rows}}
    if any(summary.get(key) != value for key, value in expected.items()):
        raise RuntimeError("cached summary does not match episode outcomes")


def _enable_all(config: Any) -> None:
    config.enabled = True
    for layer in ("h2", "h3", "h4", "h5"):
        setattr(config, f"{layer}_enabled", True)


def initialize(domain: str, split: str, chain_manifest: str) -> None:
    global TASK, DOMAIN, MANIFEST_PATH, TOKENIZER
    DOMAIN = domain
    MANIFEST_PATH = Path(chain_manifest)
    os.chdir(AGENTBENCH)
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError:
        if domain != 'alfworld':
            raise
        TOKENIZER = TokenizerProxy()
    else:
        TOKENIZER = AutoTokenizer.from_pretrained(
            os.environ.get("LIFE_SOLVER_TOKENIZER", "Qwen/Qwen3-4B"), local_files_only=True
        )
    if domain == "webshop":
        sys.path.insert(0, str(AGENTBENCH / ".native" / "webshop-src"))
    snapshot = HISTORICAL_ROOT / "task_snapshot"
    sys.path.insert(0, str(AGENTBENCH))
    sys.path.insert(0, str(snapshot))
    import yaml
    config = yaml.safe_load((snapshot / f"configs/tasks/{domain}.yaml").read_text())["default"]["parameters"]
    config.update(name=f"{domain}-life-{split}", concurrency=1)
    if domain == "alfworld":
        import src.server.tasks.alfworld.task as task_module
        inject_task_module(domain, task_module, MANIFEST_PATH)
        config.update(data_path=str(AGENTBENCH / "data/alfworld"),
                      config_path=str(snapshot / "src/server/tasks/alfworld/configs/base_config.yaml"),
                      prompts_path=str(snapshot / "src/server/tasks/alfworld/prompts/alfworld_multiturn_plan_first.json"),
                      split="train_valid" if split == "train" else "new_std",
                      enabled=False, h2=False, h3=False, h4=False, h5=False)
        TASK = task_module.ALFWorld(**config)
    elif domain == "dbbench":
        import src.server.tasks.dbbench.task as task_module
        inject_task_module(domain, task_module, MANIFEST_PATH)
        config.update(data_file=str(AGENTBENCH / f"data/dbbench/{'db_out_new.jsonl' if split == 'train' else 'standard.jsonl'}"),
                      env_driver="native_mysql", env_options={"host": "127.0.0.1", "port": 13306},
                      db_password="", harness={"enabled": False, "h2": False, "h3": False,
                                               "h4": False, "h5": False})
        TASK = task_module.DBBenchTask(**config)
    else:
        import src.server.tasks.webshop.task as task_module
        inject_task_module(domain, task_module, MANIFEST_PATH)
        config.update(start=0, end=1021, enabled=False, h2=False, h3=False, h4=False, h5=False)
        TASK = task_module.WebShop(**config)
    _enable_all(TASK.harness_config)


def _bounded_request_messages(history: list[dict[str, Any]], tools: list[dict[str, Any]],
                              max_tokens: int) -> list[dict[str, Any]]:
    messages = copy.deepcopy(history)
    if TOKENIZER is None:
        raise RuntimeError('Solver tokenizer not initialized')
    prompt_budget = 32768 - max_tokens - 256

    def token_count(value: list[dict[str, Any]]) -> int:
        return len(TOKENIZER.apply_chat_template(
            value, tools=tools, tokenize=True, add_generation_prompt=True, enable_thinking=False
        ))

    if token_count(messages) <= prompt_budget:
        return messages
    if DOMAIN != 'webshop':
        raise ContextLimit('Current history exceeds total context limit 32768')
    anchors = messages[:2]
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in messages[2:]:
        if message.get("role") == "assistant" and current:
            groups.append(current)
            current = []
        current.append(message)
    if current:
        groups.append(current)
    while len(groups) > 1 and token_count(anchors + [item for group in groups for item in group]) > prompt_budget:
        groups.pop(0)
    bounded = anchors + [item for group in groups for item in group]
    if token_count(bounded) > prompt_budget:
        raise ContextLimit("WebShop current interaction exceeds total context limit 32768")
    return bounded


class EpisodeSession:
    def __init__(self):
        self.history: list[Any] = []
        self.calls: list[dict[str, Any]] = []
        self.rewards: list[Any] = []
        self.turn = 0
        self.tools = copy.deepcopy(TASK.tools)
        self.controller = SimpleNamespace(env_output=SimpleNamespace(history=[]))

    def set_tools(self, tools):
        self.tools = copy.deepcopy(tools)

    def set_full_history(self, _value):
        return None

    def cover(self, history):
        self.history = copy.deepcopy(history)

    def inject(self, item):
        if isinstance(item, list):
            for value in item:
                self.inject(value)
        elif isinstance(item, dict) and "role" in item:
            self.history.append(copy.deepcopy(item))
        elif hasattr(item, "reward"):
            self.rewards.append(item.model_dump(mode="json"))

    def sync_action(self):
        import requests
        self.turn += 1
        fixed = HISTORICAL["agent"]
        body = {key: copy.deepcopy(value) for key, value in fixed.items() if key != "url"}
        tools = copy.deepcopy(self.tools)
        messages = _bounded_request_messages(
            self.history, tools, int(body.get("max_tokens", 4096))
        )
        body.update(messages=messages, tools=tools, tool_choice="auto",
                    parallel_tool_calls=False)
        call = {"turn": self.turn, "request": copy.deepcopy(body)}
        self.calls.append(call)
        for attempt in range(3):
            try:
                response = requests.post(fixed["url"], headers={"Authorization": "Bearer EMPTY"},
                                         json=body, timeout=240)
                response.raise_for_status()
                data = response.json()
                break
            except Exception as exc:
                call.setdefault("transport_errors", []).append(str(exc))
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        raw = data["choices"][0]["message"]
        message = {key: raw[key] for key in ("role", "content", "tool_calls")
                   if key in raw and raw[key] is not None}
        message.setdefault("role", "assistant")
        call.update(response=data, usage=data.get("usage", {}))
        self.history.append(copy.deepcopy(message))
        return SimpleNamespace(messages=[message])

    async def action(self):
        return self.sync_action()


async def _run_db(index: int, session: EpisodeSession):
    try:
        return await TASK.start_sample(index, session)
    finally:
        if TASK.env_controller_background_task:
            TASK.env_controller_background_task.cancel()
            TASK.env_controller_background_task = None


def episode(index: int) -> dict[str, Any]:
    started = time.time()
    session = EpisodeSession()
    try:
        result = (TASK.sync_start_sample(index, session) if DOMAIN in ("alfworld", "webshop")
                  else asyncio.run(_run_db(index, session)))
        status = result.status.value
        raw = result.result or {}
        if DOMAIN == "alfworld":
            success = bool(raw.get("result") == 1)
        elif DOMAIN == "dbbench":
            success = bool(raw.get("is_correct", False))
        else:
            success = float(raw.get("reward", 0) or 0) >= 1.0
        infra = status.replace(" ", "_") in {"task_error", "cancelled", "unknown", "running"}
        infra = infra or any("transport_errors" in call and "response" not in call
                             for call in session.calls)
        output = {"index": index, "success": int(success), "status": status,
                  "infrastructure_error": infra, "task_result_private": raw}
    except ContextLimit as exc:
        output = {"index": index, "success": 0, "status": "context_limit",
                  "infrastructure_error": False, "failure_kind": "context_limit", "detail": str(exc)}
    except Exception:
        output = {"index": index, "success": 0, "status": "infrastructure_error",
                  "infrastructure_error": True, "exception": traceback.format_exc()}
    if output["infrastructure_error"]:
        output["success"] = 0
    output.update(wall_seconds=time.time() - started, turns=session.turn,
                  history=session.history, calls=session.calls, rewards=session.rewards,
                  tokens=sum(call.get("usage", {}).get("total_tokens", 0)
                             for call in session.calls))
    return output
