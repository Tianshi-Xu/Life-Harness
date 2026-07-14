from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class HarnessEventLogger:
    def __init__(self) -> None:
        self.path: Path | None = None

    def initialize(self, logs_directory: str) -> None:
        self.path = Path(logs_directory) / "harness_events.jsonl"

    def log(self, layer: str, step: int, trigger: str, intervention: str, **data: Any) -> None:
        if self.path is None:
            return
        safe_data = {
            key: ("<REDACTED>" if key in {"password", "access_token"} else value)
            for key, value in data.items()
        }
        row = {
            "layer": layer,
            "step": step,
            "trigger": trigger,
            "intervention": intervention,
            "data": safe_data,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

