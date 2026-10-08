"""Read-only artifact audit; emit a receipt without exposing API credentials."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from .backend import HERE, MANIFEST, MANIFEST_SHA256
from .connection import credentials
from .records import write_json
from .upstream import source_fingerprint


def audit(experiment):
    experiment = Path(experiment).resolve()
    if experiment.parent != HERE / "experiments":
        raise ValueError("audit one integration experiment, not arbitrary paths")
    source = source_fingerprint()
    nexau = HERE / "vendor/NexAU"
    if subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=nexau):
        raise ValueError("tracked NexAU source changed")
    framework_files = [Path(p) for p in subprocess.check_output(["git", "ls-files", "nexau"],
        cwd=nexau, text=True).splitlines() if p.endswith(".py")]
    for runtime in (HERE / ".venv/lib/python3.13/site-packages",
                    HERE / "runtime_rootfs/runtime/lib/python3.13/site-packages"):
        for relative in framework_files:
            if hashlib.sha256((runtime / relative).read_bytes()).digest() != hashlib.sha256((nexau / relative).read_bytes()).digest():
                raise ValueError(f"installed NexAU source differs from pin: {relative}")
    if hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise ValueError("benchmark manifest changed")
    measured = []
    for iteration in sorted((experiment / "runs").glob("iteration_*")):
        jobs = sorted((iteration / "input/benchmark").glob("20*"))
        for job in jobs:
            if not (job / "result.json").is_file():
                continue
            records = [json.loads(p.read_text()) for p in sorted(job.glob("native_*/native_record.json"))]
            receipt = json.loads((job / "workspace_receipt.json").read_text())
            trace_count = len(list(job.glob("native_*/agent/nexau_in_memory_tracer.json")))
            if len(records) != trace_count:
                raise ValueError("completed job lacks genuine runtime traces")
            measured.append({"iteration": iteration.name, "job": str(job.relative_to(experiment)),
                "tasks": len(records), "successes": sum(r["success"] for r in records),
                "infrastructure_errors": sum(r["infrastructure_error"] for r in records),
                "workspace_sha256": receipt["workspace_sha256"], "genuine_traces": trace_count})
    key, _ = credentials()
    secret = key.encode()
    checked, matches = 0, []
    for root in (HERE / "smokes", HERE / "experiments", HERE / "runtime_homes",
                 HERE / "runtime_rootfs/episodes"):
        for path in root.rglob("*"):
            if not path.is_file() or path.is_symlink() or path.suffix not in {
                    ".json", ".jsonl", ".yaml", ".yml", ".md", ".txt", ".log", ".py"}:
                continue
            checked += 1
            if secret in path.read_bytes():
                matches.append(str(path.relative_to(HERE)))
    result = {"official_ahe_commit": source["commit"], "official_ahe_tracked_files": len(source["tracked_sha256"]),
        "official_ahe_tracked_source_clean": True, "nexau_tracked_source_clean": True,
        "nexau_installed_and_isolated_source_files_matched": len(framework_files),
        "nexau_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=nexau, text=True).strip(),
        "manifest_sha256": MANIFEST_SHA256, "measured_training_jobs": measured,
        "credential_artifacts_checked": checked, "credential_matches": matches,
        "scope": "current derived artifacts; cannot prove historical model-visible exposure was zero"}
    write_json(experiment / "verification_receipt.json", result)
    if matches:
        raise ValueError("credential found in derived artifacts; see receipt paths, never print content")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.experiment), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
