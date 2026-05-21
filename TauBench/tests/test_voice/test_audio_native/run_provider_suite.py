#!/usr/bin/env python3
"""Run the provider suite and print a sorted, glanceable summary.

Results are grouped by provider, then ordered by test stage.
Output is written to provider_suite_results.txt atomically (the previous
file remains readable until the new run finishes).

Usage (from repo root):
    uv run tests/test_voice/test_audio_native/run_provider_suite.py
"""

import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

SUITE_DIR = Path(__file__).parent
SUITE = str(SUITE_DIR / "test_provider_suite.py")
RESULTS_FILE = SUITE_DIR / "provider_suite_results.txt"

TEST_ORDER = {
    "connect_disconnect": 1,
    "reconnect_after_disconnect": 2,
    "single_turn_reply": 3,
    "multi_turn_reply": 4,
    "tool_call_round_trip": 5,
    "barge_in": 6,
}


def run_and_summarize() -> int:
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
        xml_path = f.name

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                SUITE,
                f"--junitxml={xml_path}",
                "-n",
                "4",
                "-p",
                "no:warnings",
                "-q",
            ],
            capture_output=True,
        )

        tree = ET.parse(xml_path)
        root = tree.getroot()
    finally:
        Path(xml_path).unlink(missing_ok=True)

    results = []
    for tc in root.iter("testcase"):
        name = tc.get("name", "")
        test_base = name.split("[")[0].replace("test_", "")
        params = name.split("[")[1].rstrip("]") if "[" in name else ""
        provider = params.split("-")[0] if params else "unknown"
        order = TEST_ORDER.get(test_base, 9)

        failure = tc.find("failure")
        skip = tc.find("skipped")
        if skip is not None:
            status, detail = "⏭️ SKIP", skip.get("message", "")
        elif failure is not None:
            status, detail = "❌ FAIL", failure.get("message", "")
        else:
            status, detail = "✅ PASS", ""

        short = name.replace(f"[{provider}-", "[").replace(f"[{provider}]", "")
        results.append((provider, order, status, short, detail, params))

    results.sort(key=lambda r: (r[0], r[1], r[5]))

    ts = root.find("testsuite")
    total = int(ts.get("tests", 0))
    failures = int(ts.get("failures", 0))
    skipped = int(ts.get("skipped", 0))
    errors = int(ts.get("errors", 0))
    elapsed = float(ts.get("time", 0))
    passed = total - failures - skipped - errors

    lines = []
    lines.append(f"Provider Suite — {datetime.now():%Y-%m-%d %H:%M %Z}")
    lines.append("")

    current = ""
    for provider, _, status, short, _, _ in results:
        if provider != current:
            if current:
                lines.append("")
            lines.append(f"[{provider}]")
            current = provider
        lines.append(f"  {status}  {short}")

    fail_details = [(s, d) for _, _, st, s, d, _ in results if "FAIL" in st]
    if fail_details:
        lines.append("")
        lines.append("Failure details:")
        for short, detail in fail_details:
            lines.append(f"  {short}: {detail}")

    lines.append("")
    lines.append(
        f"{passed} passed, {failures} failed, {skipped} skipped, "
        f"{errors} errors in {elapsed:.0f}s"
    )

    output = "\n".join(lines)
    print(output)

    # Atomic write: write to temp file then rename, so the previous results
    # file remains readable until the new one is complete.
    tmp = RESULTS_FILE.with_suffix(".tmp")
    tmp.write_text(output + "\n")
    tmp.rename(RESULTS_FILE)

    return 1 if failures or errors else 0


if __name__ == "__main__":
    sys.exit(run_and_summarize())
