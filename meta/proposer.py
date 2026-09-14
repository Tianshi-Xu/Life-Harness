"""Proposer wrapper: drives the qoder CLI as the harness-proposing agent.

Adapted from meta-harness' claude_wrapper.py. The outer loop only depends on
the file contract this produces: the proposer must write
``<run_dir>/pending_eval.json`` plus candidate plugin files under
``<run_dir>/candidates/`` before the CLI process exits.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ALLOWED_TOOLS = ("Read", "Write", "Edit", "Glob", "Grep")


@dataclass
class ProposerResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool


def run(
    prompt: str,
    cwd: Path,
    model: str | None = None,
    timeout: int = 5400,
    permission_mode: str = "accept_edits",
    allowed_tools: tuple[str, ...] = DEFAULT_ALLOWED_TOOLS,
    log_path: Path | None = None,
    add_dirs: list[Path] | None = None,
) -> ProposerResult:
    """Run one headless qoder session and return its result.

    ``accept_edits`` auto-approves file edits while Bash stays denied, so the
    proposer can read code/traces and write candidate files but cannot execute
    anything (evaluation is the outer loop's job). ``add_dirs`` extends the
    session workspace beyond ``cwd`` (writes outside the workspace are blocked
    by the permission gate).
    """
    if shutil.which("qoder") is None:
        raise RuntimeError("qoder CLI not found on PATH")

    cmd = [
        "qoder",
        "-p", prompt,
        "-w", str(cwd),
        "-o", "json",
        "--permission-mode", permission_mode,
    ]
    for d in add_dirs or []:
        cmd += ["--add-dir", str(d)]
    # --allowed-tools grants permission but does NOT restrict the tool set
    # in current Qoder versions. --tools is the actual availability allowlist.
    # The following --disallowed-tools option terminates the variadic list.
    cmd += ["--tools", *allowed_tools, "--disallowed-tools", "Bash",
            "--disallowed-tools", "Agent", "--disallowed-tools", "Task"]
    for tool in allowed_tools:
        cmd += ["--allowed-tools", tool]
    if model:
        cmd += ["-m", model]

    timed_out = False
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=str(cwd)
        )
        exit_code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        timed_out = True
        exit_code = -1
        stdout = (e.stdout or "") if isinstance(e.stdout, str) else ""
        stderr = (e.stderr or "") if isinstance(e.stderr, str) else ""

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            json.dumps(
                {
                    "cmd": cmd,
                    "exit_code": exit_code,
                    "timed_out": timed_out,
                    "stdout": stdout,
                    "stderr": stderr,
                },
                indent=2,
            )
        )
    return ProposerResult(exit_code, stdout, stderr, timed_out)
