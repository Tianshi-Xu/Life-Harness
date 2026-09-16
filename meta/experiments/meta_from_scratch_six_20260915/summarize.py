"""Build compact, reviewable tables from the six per-domain state files."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DOMAINS = ("airline", "retail", "telecom", "alfworld", "dbbench", "webshop")


def load_states(require_complete: bool) -> dict[str, dict]:
    states = {}
    for domain in DOMAINS:
        path = ROOT / domain / "state.json"
        if not path.exists():
            raise SystemExit(f"missing state: {path}")
        state = json.loads(path.read_text())
        if require_complete and (state.get("status") != "complete" or len(state.get("rounds", [])) != 6):
            raise SystemExit(f"{domain} is not complete: status={state.get('status')} rounds={len(state.get('rounds', []))}")
        if not state.get("from_scratch") or state.get("released_h_flags") != []:
            raise SystemExit(f"{domain} is not a clean identity-start run")
        states[domain] = state
    return states


def rows(states: dict[str, dict]) -> list[dict]:
    output = []
    for domain in DOMAINS:
        for row in states[domain].get("rounds", []):
            if row.get("released_h_flags") != []:
                raise SystemExit(f"released H flags found in {domain} round {row.get('iteration')}")
            output.append({
                "domain": domain,
                "round": row["iteration"],
                "train_batch": row["train_batch"],
                "train_n": row["train_n"],
                "baseline_candidate": row["baseline_candidate"],
                "baseline_score": row["baseline_score"],
                "candidate": row["candidate"]["name"],
                "candidate_score": row["candidate_score"],
                "accepted": row["accepted"],
                "wins": row["wins"],
                "losses": row["losses"],
                "next_batch_changes": row["next_batch_changes"],
                "retained_candidate": row["retained_candidate"],
                "test_n": row["test_n"],
                "test_score": row["test_score"],
            })
    return output


def markdown(states: dict[str, dict], flat: list[dict]) -> str:
    lines = [
        "# Results",
        "",
        "All runs start from the identity implementation. Released Life-Harness H2-H5 flags are empty in every round. Candidate acceptance uses training score only; the complete fixed test split is evaluated after every round and hidden from subsequent proposals.",
        "",
        "| Domain | Round | Batch | Train baseline | Train candidate | Accepted | Retained | Test |",
        "|---|---:|---:|---:|---:|:---:|---|---:|",
    ]
    for row in flat:
        lines.append(
            f"| {row['domain']} | {row['round']} | {row['train_batch']} | "
            f"{row['baseline_score']:.4f} | {row['candidate_score']:.4f} | "
            f"{'yes' if row['accepted'] else 'no'} | {row['retained_candidate']} | "
            f"{row['test_score']:.4f} ({row['test_n']}) |"
        )
    lines.extend(["", "## Domain summary", "", "| Domain | Rounds | Accepted | Final retained | Final test | Best test |", "|---|---:|---:|---|---:|---:|"])
    for domain in DOMAINS:
        rs = states[domain].get("rounds", [])
        if not rs:
            lines.append(f"| {domain} | 0 | 0 | baseline_zero.py | — | — |")
            continue
        lines.append(
            f"| {domain} | {len(rs)} | {sum(bool(r['accepted']) for r in rs)} | "
            f"{states[domain]['incumbent']} | {rs[-1]['test_score']:.4f} | "
            f"{max(r['test_score'] for r in rs):.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    states = load_states(require_complete=not args.allow_partial)
    flat = rows(states)
    payload = {
        "protocol": {
            "from_scratch": True,
            "released_h_flags": [],
            "rounds_per_domain": 6,
            "test_after_every_round": True,
            "selection_uses_test": False,
        },
        "domains": states,
    }
    (ROOT / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    with (ROOT / "rounds.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]) if flat else ["domain", "round"])
        writer.writeheader()
        writer.writerows(flat)
    (ROOT / "RESULTS.md").write_text(markdown(states, flat))
    print(f"wrote {len(flat)} round rows")


if __name__ == "__main__":
    main()
