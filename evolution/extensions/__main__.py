"""Run a custom benchmark with Life-Harness, or execute the offline toy demo."""
import argparse
import importlib
from pathlib import Path

from .runner import ExtensionRunner, evaluate


def load_adapter(reference):
    module, separator, name = reference.partition(":")
    if not separator or not module or not name:
        raise ValueError("adapter reference must be module:ClassOrFactory")
    return getattr(importlib.import_module(module), name)()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="offline scripted toy; no model, tokenizer or API needed")
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--rounds", type=int, default=2)
    evolve = commands.add_parser("evolve", help="runs model optimization and native evaluation")
    evolve.add_argument("--adapter", required=True)
    evolve.add_argument("--output", type=Path, required=True)
    evolve.add_argument("--rounds", type=int, default=6)
    scoring = commands.add_parser("evaluate", help="score a frozen harness; solver calls only")
    scoring.add_argument("--adapter", required=True)
    scoring.add_argument("--manifest", type=Path, required=True)
    scoring.add_argument("--output", type=Path, required=True)
    scoring.add_argument("--split", choices=("train", "test"), default="test")
    args = parser.parse_args()
    if args.command == "demo":
        from .examples import ToyAdapter, offline_proposer
        root = ExtensionRunner(ToyAdapter(), args.output, rounds=args.rounds, offline_proposer=offline_proposer).run()
    elif args.command == "evolve":
        root = ExtensionRunner(load_adapter(args.adapter), args.output, rounds=args.rounds).run()
    else:
        score = evaluate(load_adapter(args.adapter), args.manifest, args.output, split=args.split)
        if score["infra_errors"]:
            raise RuntimeError("native infrastructure errors; inspect the saved evaluation records")
        root = args.output
    print(root)


if __name__ == "__main__":
    main()
