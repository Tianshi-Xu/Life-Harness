"""Diagnostic only: show the original Debugger result before its original parser."""
import sys
from agent_debugger_core.runtime import runner
from agent_debugger_core.cli.adb import main

original = runner._parse_run_output


def inspect_result(value):
    print("ORIGINAL_DEBUGGER_RAW_RESULT:", repr(value)[:6000], file=sys.stderr)
    return original(value)


if __name__ == "__main__":
    runner._parse_run_output = inspect_result
    raise SystemExit(main())
