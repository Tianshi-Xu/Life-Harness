"""Continue the frozen six-round run with corrected inherited evidence lookup."""
from __future__ import annotations

import json
import sys

import continuation_support
import driver
import protocol
import runner
import status


def main():
    continuation_support.bind_amendment()
    continuation_support.install()
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    if command == "status":
        for row in status.rows():
            print(json.dumps(row), flush=True)
    elif command == "verify":
        protocol.validate_protocol()
        for domain in protocol.DOMAINS:
            folder = protocol.HERE / domain
            state = json.loads((folder / "state.json").read_text())
            driver.verify_active(folder, state)
            driver.verify_training_history(folder, state)
            print(json.dumps({"verified": domain, "round": state["round"]}), flush=True)
    elif command == "report":
        print(json.dumps(status.final_report(), indent=2), flush=True)
    else:
        runner.main()


if __name__ == "__main__":
    main()
