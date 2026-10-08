"""Preserve invalid unsealed DB9 evidence and restore the pre-round-nine state."""
import shutil
import protocol


def main():
    root = protocol.HERE
    archive = root / "runtime_cap_archive/dbbench"
    state_path = root / "dbbench/state.json"
    state = protocol.load_json(state_path)
    if state["round"] != 9 or state["rounds"][-1]["accepted"]:
        raise RuntimeError("repair only applies to the unaccepted invalid DB9 round")
    folder = root / "dbbench/evals/candidate_09"
    summary = protocol.load_json(folder / "summary.json")
    log = root / "dbbench/candidate_09.log"
    if (summary["n"] != 50 or summary["infra_errors"] != 50
            or log.read_text().count("ValueError: H4/H5 require at most one hint of at most 120 words") != 50):
        raise RuntimeError("invalid round is not exactly the known Task word-cap failure")
    if (root / "training_seal.json").exists() or archive.exists():
        raise RuntimeError("refuse to rewrite sealed or already archived data")
    unaffected = {}
    for domain in protocol.DOMAINS:
        for path in sorted((root / domain / "evals").glob("*/summary.json")):
            if path.parent == folder:
                continue
            row = protocol.load_json(path)
            if row["infra_errors"] != 0:
                raise RuntimeError("another evaluation requires separate error audit")
            unaffected[str(path.parent.relative_to(root))] = {
                p.name: protocol.digest(p) for p in sorted(path.parent.glob("*.json"))}
    protocol.atomic_json(root / "unaffected_evidence.json", unaffected)
    archive.mkdir(parents=True)
    shutil.copy2(state_path, archive / "state_before_repair.json")
    shutil.move(str(folder), str(archive / "candidate_09"))
    shutil.move(str(log), str(archive / "candidate_09.log"))
    state["rounds"] = state["rounds"][:8]
    state["round"] = 8
    state["status"] = "running"
    state["batch"] = state["rounds"][-1]["next_batch"]
    state["baseline_phase"] = "batch_04_incumbent"
    state["baseline_summary"] = protocol.load_json(root / "dbbench/evals/batch_04_incumbent/summary.json")
    protocol.atomic_json(state_path, state)
    import runtime_support
    runtime_support.bind_amendment()
    print("Archived invalid DB9; retained all unaffected episodes and cached proposal; restored round 8.")


if __name__ == "__main__":
    main()
