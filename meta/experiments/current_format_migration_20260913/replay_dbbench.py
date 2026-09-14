"""Replay the 300 frozen Meta DBBench replies through the migrated Task."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
import asyncio
import copy
import json
import os
import re
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AB = ROOT / "AgentBench"
SOURCE = HERE.parent / "meta_agentbench_test_20260913"
sys.path.insert(0, str(AB))
os.chdir(AB)

from agentrl.worker.task import Session


TASK = None


def normalize_ephemeral_names(value):
    """Remove the per-session MySQL database UUID from otherwise equal errors."""
    if isinstance(value, str):
        return re.sub(r"dbbench_[0-9a-f_]+", "dbbench_<session>", value)
    if isinstance(value, list):
        return [normalize_ephemeral_names(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_ephemeral_names(item) for key, item in value.items()}
    return value


def initialize():
    global TASK
    import yaml
    from src.server.tasks.dbbench.task import DBBenchTask

    config = yaml.safe_load((AB / "configs/tasks/dbbench.yaml").read_text())["default"]["parameters"]
    config.update(
        name="current-format-replay",
        concurrency=1,
        data_file=str(AB / "data/dbbench/standard.jsonl"),
        env_driver="native_mysql",
        env_options={"host": "127.0.0.1", "port": 13306},
        db_password="",
        harness={"enabled": True, "h2": True, "h3": True, "h4": True, "h5": True, "h5_top_k": 1},
    )
    TASK = DBBenchTask(**config)


class Replay(Session):
    def __init__(self, row):
        super().__init__(0)
        self.row = row
        self.n = 0
        self.input_differences = []
        self.raw_input_differences = []
        self.tool_differences = []

    async def action(self, *args):
        if self.n >= len(self.row["calls"]):
            raise RuntimeError("Migrated task requested an extra frozen response")
        call = self.row["calls"][self.n]
        public = [
            copy.deepcopy(item)
            for item in self.history
            if isinstance(item, dict) and "role" in item
        ]
        if public != call["request"]["messages"]:
            self.raw_input_differences.append(self.n + 1)
        if normalize_ephemeral_names(public) != normalize_ephemeral_names(call["request"]["messages"]):
            self.input_differences.append(self.n + 1)
        if self.tools != call["request"]["tools"]:
            self.tool_differences.append(self.n + 1)
        raw = call["response"]["choices"][0]["message"]
        raw = {
            key: value
            for key, value in raw.items()
            if key in ("role", "content", "tool_calls") and value is not None
        }
        raw.setdefault("role", "assistant")
        self.history.append(copy.deepcopy(raw))
        self.n += 1
        return SimpleNamespace(messages=[raw])


async def run_one(index, row):
    session = Replay(row)
    try:
        output = await TASK.start_sample(index, session)
        raw = output.result or {}
        return {
            "index": index,
            "expected_success": row["success"],
            "success": int(bool(raw.get("is_correct", False))),
            "expected_status": row["status"],
            "status": output.status.value,
            "expected_turns": len(row["calls"]),
            "turns": session.n,
            "input_differences": session.input_differences,
            "raw_input_differences": session.raw_input_differences,
            "tool_differences": session.tool_differences,
        }
    finally:
        if TASK.env_controller_background_task:
            TASK.env_controller_background_task.cancel()
            TASK.env_controller_background_task = None


def replay(index):
    row = json.loads((SOURCE / "dbbench/evals/meta_retained" / f"episode_{index}.json").read_text())
    try:
        return asyncio.run(run_one(index, row))
    except Exception as exc:
        return {
            "index": index,
            "expected_success": row["success"],
            "success": 0,
            "status": "replay_error",
            "expected_status": row["status"],
            "expected_turns": len(row["calls"]),
            "turns": -1,
            "input_differences": [],
            "raw_input_differences": [],
            "tool_differences": [],
            "error": repr(exc),
        }


if __name__ == "__main__":
    indices = json.loads((SOURCE / "manifest.json").read_text())["dbbench"]["batches"][0]["indices"]
    rows = []
    with ProcessPoolExecutor(max_workers=4, initializer=initialize) as pool:
        for future in as_completed([pool.submit(replay, index) for index in indices]):
            rows.append(future.result())
    rows.sort(key=lambda item: item["index"])
    result = {
        "n": len(rows),
        "successes": sum(item["success"] for item in rows),
        "expected_successes": sum(item["expected_success"] for item in rows),
        "outcome_mismatches": [item["index"] for item in rows if item["success"] != item["expected_success"]],
        "status_mismatches": [item["index"] for item in rows if item["status"] != item["expected_status"]],
        "turn_mismatches": [item["index"] for item in rows if item["turns"] != item["expected_turns"]],
        "replay_errors": [item["index"] for item in rows if "error" in item],
        "model_input_differences": sum(len(item["input_differences"]) for item in rows),
        "raw_model_input_differences": sum(len(item["raw_input_differences"]) for item in rows),
        "tool_input_differences": sum(len(item["tool_differences"]) for item in rows),
        "new_model_calls": 0,
        "episodes": rows,
    }
    (HERE / "replay_dbbench.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "episodes"}))
