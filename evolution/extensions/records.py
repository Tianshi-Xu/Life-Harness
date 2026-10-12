"""Durable request accounting; unavailable provider usage stays unknown."""
import json
import os
import re
import time
import uuid
from pathlib import Path


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.rows = []
        if self.path.exists():
            self.rows = [json.loads(line) for line in self.path.read_text().splitlines() if line]

    def append(self, row):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.rows.append(row)

    def start(self, role, **scope):
        request_id = uuid.uuid4().hex
        self.append({"event": "started", "request_id": request_id, "role": role,
                     "time": time.time(), **scope})
        return request_id

    def finish(self, request_id, usage, *, usage_source="provider", error_type=None):
        if usage_source not in {"provider", "local_no_api"}:
            raise ValueError("usage_source must be provider or local_no_api")
        self.append({"event": "finished", "request_id": request_id, "usage": usage,
                     "usage_source": usage_source, "error_type": error_type, "time": time.time()})

    def summary(self, role=None, round_number=None, phase=None):
        starts = {r["request_id"]: r for r in self.rows if r["event"] == "started"
                  and (role is None or r["role"] == role)
                  and (round_number is None or r.get("round") == round_number)
                  and (phase is None or r.get("phase") == phase)}
        ends = {r["request_id"]: r for r in self.rows if r["event"] == "finished"}
        result = {"calls": len(starts), "known_usage_calls": 0, "unknown_usage_calls": 0,
                  "pending_calls": 0, "local_no_api_calls": 0, "input_tokens": 0,
                  "output_tokens": 0, "total_tokens": 0}
        for request_id in starts:
            end = ends.get(request_id)
            if end is None:
                result["pending_calls"] += 1
                continue
            if end["usage_source"] == "local_no_api":
                result["local_no_api_calls"] += 1
                continue
            usage = end.get("usage") or {}
            values = [usage.get("prompt_tokens"), usage.get("completion_tokens")] if isinstance(usage, dict) else [None, None]
            if any(type(v) is not int or v < 0 for v in values):
                result["unknown_usage_calls"] += 1
                continue
            result["known_usage_calls"] += 1
            result["input_tokens"] += values[0]
            result["output_tokens"] += values[1]
            result["total_tokens"] += sum(values)
        result["accounting_complete"] = result["unknown_usage_calls"] == result["pending_calls"] == 0
        result["scope"] = "known provider input+output; missing usage excluded, local calls separately labeled"
        return result

    def roles(self):
        return sorted({r["role"] for r in self.rows if r["event"] == "started"})

    def call(self, role, request, **scope):
        """Optional native simulator/judge metering; private response stays with the adapter."""
        if role in {"solver", "life_optimizer"} or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", role):
            raise ValueError("use a separate native model role, such as simulator or judge")
        request_id = self.start(role, **scope)
        try:
            result = request()
            if not isinstance(result, dict):
                raise TypeError("recorded request must return a JSON response dictionary")
        except Exception as exc:
            self.finish(request_id, None, error_type=type(exc).__name__)
            raise
        self.finish(request_id, result.get("usage"))
        return result
