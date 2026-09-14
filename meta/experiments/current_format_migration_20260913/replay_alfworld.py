"""Replay frozen Meta-eval replies through the four-hook release adapter."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace
import copy
import json
import os
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AB = ROOT / "AgentBench"
TEST = HERE.parent / "meta_agentbench_test_20260913"
sys.path.insert(0, str(AB))
os.chdir(AB)

from agentrl.worker.task import Session

TASK = None


def normalize_h4_transport(messages):
    """Treat adjacent user guidance as one bounded H4 payload.

    The legacy Task could inject recovery and step guidance as two adjacent user
    messages.  The current hook contract returns at most one hint, so the bridge
    joins those texts with a newline.  This preserves text and ordering while
    changing only message boundaries.
    """
    normalized = []
    for item in messages:
        item = copy.deepcopy(item)
        if (
            normalized
            and item.get("role") == "user"
            and normalized[-1].get("role") == "user"
        ):
            normalized[-1]["content"] = (
                str(normalized[-1].get("content") or "")
                + "\n"
                + str(item.get("content") or "")
            )
        else:
            normalized.append(item)
    return normalized


def initialize():
    global TASK
    import yaml
    from src.server.tasks.alfworld.task import ALFWorld

    cfg = yaml.safe_load((AB / "configs/tasks/alfworld.yaml").read_text())["default"]["parameters"]
    cfg.update(
        name="meta-release-replay",
        concurrency=1,
        enabled=True,
        h2=True,
        h3=True,
        h4=True,
        h5=True,
        split="new_std",
    )
    for key in ("data_path", "config_path", "prompts_path"):
        cfg[key] = str(AB / cfg[key].removeprefix("/app/"))
    TASK = ALFWorld(**cfg)


class Replay(Session):
    def __init__(self, row):
        super().__init__(0)
        self.row = row
        self.n = 0
        self.input_differences = []
        self.raw_input_differences = []

    def sync_action(self, *args):
        call = self.row["calls"][self.n]
        public = [
            copy.deepcopy(item)
            for item in self.history
            if isinstance(item, dict) and "role" in item
        ]
        expected = call["request"]["messages"]
        if public != expected:
            self.raw_input_differences.append(self.n + 1)
        if normalize_h4_transport(public) != normalize_h4_transport(expected):
            self.input_differences.append(self.n + 1)
        assert self.tools == call["request"]["tools"], "tools mismatch"
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


def replay(index):
    row = json.loads((TEST / "alfworld/evals/meta_efficiency" / f"episode_{index}.json").read_text())
    session = Replay(row)
    output = TASK.sync_start_sample(index, session)
    success = int(output.result.get("result") == 1) if output.result else 0
    return {
        "index": index,
        "expected_success": row["success"],
        "success": success,
        "expected_status": row["status"],
        "status": output.status.value,
        "expected_turns": len(row["calls"]),
        "turns": session.n,
        "input_differences": session.input_differences,
        "raw_input_differences": session.raw_input_differences,
    }


if __name__ == "__main__":
    ids = json.loads((TEST / "manifest.json").read_text())["alfworld"]["batches"][0]["indices"]
    rows = []
    with ProcessPoolExecutor(max_workers=4, initializer=initialize) as pool:
        for future in as_completed([pool.submit(replay, index) for index in ids]):
            rows.append(future.result())
    rows.sort(key=lambda item: item["index"])
    result = {
        "n": len(rows),
        "successes": sum(item["success"] for item in rows),
        "expected_successes": sum(item["expected_success"] for item in rows),
        "outcome_mismatches": [item["index"] for item in rows if item["success"] != item["expected_success"]],
        "status_mismatches": [item["index"] for item in rows if item["status"] != item["expected_status"]],
        "turn_mismatches": [item["index"] for item in rows if item["turns"] != item["expected_turns"]],
        "model_input_differences": sum(len(item["input_differences"]) for item in rows),
        "raw_model_input_differences": sum(len(item["raw_input_differences"]) for item in rows),
        "new_model_calls": 0,
        "episodes": rows,
    }
    (HERE / "replay_alfworld.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "episodes"}))
