#!/usr/bin/env python3
"""Compute WebShop success rate / average reward per model x mode.

Usage: python scripts/analyze_webshop.py [outputs_root]
"""
import glob
import json
import os
import sys


def analyze_dir(path):
    total = success = 0
    rewards = []
    errors = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            # task error detection: status field or result
            status = r.get("status") or r.get("output", {}).get("status")
            res = r.get("output", {}).get("result", {})
            if not isinstance(res, dict):
                res = {}
            reward = res.get("reward", 0)
            try:
                reward = float(reward)
            except (TypeError, ValueError):
                reward = 0.0
            rewards.append(reward)
            if status and "error" in str(status).lower():
                errors += 1
            if reward >= 1.0:
                success += 1
    avg = sum(rewards) / len(rewards) if rewards else 0
    return total, success, avg, errors


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    models = ["gpt-5.5", "gemini-3.1-pro-preview", "claude-opus-4-8"]
    modes = {"run": "webshop-std", "baseline": "webshop-std-baseline"}
    print(f"{'model':30s} {'mode':10s} {'succ/total':>12s} {'rate':>7s} {'avg_reward':>11s} {'errs':>5s}")
    print("-" * 80)
    for model in models:
        for mode, task in modes.items():
            dirs = glob.glob(os.path.join(root, model, "webshop", f"*-{mode}"))
            if not dirs:
                print(f"{model:30s} {mode:10s} {'(no data)':>12s}")
                continue
            d = sorted(dirs)[-1]
            path = os.path.join(d, model, task, "runs.jsonl")
            if not os.path.exists(path):
                print(f"{model:30s} {mode:10s} {'(no runs.jsonl)':>12s}  {path}")
                continue
            total, success, avg, errors = analyze_dir(path)
            rate = success / total * 100 if total > 0 else 0
            print(f"{model:30s} {mode:10s} {success:>5}/{total:<6d} {rate:6.1f}% {avg:11.3f} {errors:5d}")


if __name__ == "__main__":
    main()
