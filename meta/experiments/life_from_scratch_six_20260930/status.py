"""Read-only status for the six-domain Life experiment."""
from __future__ import annotations

import json
import os
from pathlib import Path

from protocol import DOMAINS, HERE


def _process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def rows(root: Path = HERE) -> list[dict]:
    result = []
    for domain in DOMAINS:
        path = root / domain / "state.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        rounds = state.get("rounds", [])
        lock = root / domain / ".lock"
        try:
            pid = int(lock.read_text().strip()) if lock.exists() else None
        except ValueError:
            pid = None
        result.append({"domain": domain, "status": state.get("status", "not_started"),
                       "running": _process_alive(pid), "pid": pid,
                       "rounds": len(rounds), "batch": state.get("batch", 0),
                       "accepted_layers": len([item for item in state.get("accepted", [])
                                               if item.get("round", 0) > 0]),
                       "last": rounds[-1] if rounds else None})
    return result


def main() -> None:
    for item in rows():
        activity = f"running pid={item['pid']}" if item["running"] else "idle"
        print(f"{item['domain']:9} {item['status']:11} {activity:20} rounds={item['rounds']} "
              f"batch={item['batch']} accepted={item['accepted_layers']}")
        if item["last"]:
            last = item["last"]
            print(f"  r{last['round']}: {last['incumbent_score']:.3f} -> "
                  f"{last['candidate_score']:.3f} accepted={last['accepted']} "
                  f"test={last['test_score']:.3f}")


if __name__ == "__main__":
    main()
