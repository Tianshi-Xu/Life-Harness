"""Task-owned environment loop connected to a genuine NexAU solver process."""
import copy
import json
import os
import selectors
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"


class NativeSession:
    def __init__(self, workspace, trace_dir, tools, agent_config="code_agent.yaml", timeout=900,
                 capability_policy=None):
        self.workspace = str(workspace)
        self.trace_dir = Path(trace_dir)
        self.tools = copy.deepcopy(tools)
        self.agent_config = agent_config
        self.timeout = timeout
        self.history = []
        self.rewards = []
        self.turn = 0
        self.controller = SimpleNamespace(env_output=SimpleNamespace(history=[]))
        self.process = None
        self.cursor = 0
        self.pending = None
        self.capability_policy = copy.deepcopy(capability_policy)

    def inject(self, item):
        if isinstance(item, list):
            for message in item:
                self.inject(message)
        elif isinstance(item, dict) and "role" in item:
            self.history.append(copy.deepcopy(item))
        elif hasattr(item, "reward"):
            self.rewards.append(item.model_dump(mode="json"))

    def cover(self, history):
        self.history = copy.deepcopy(history)

    def set_tools(self, tools):
        self.tools = copy.deepcopy(tools)

    def set_full_history(self, value):
        pass

    def start(self):
        if self.capability_policy:
            from .capabilities import validate_workspace
            validate_workspace(self.workspace, self.capability_policy, self.agent_config)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        native_read, solver_write = os.pipe()
        solver_read, native_write = os.pipe()
        from .isolation import case_root, enter
        isolated, case, logical, uid = case_root(self.workspace, self.trace_dir)
        # Do not pass proposer/user credentials into candidate solver code.
        env = {k: v for k, v in os.environ.items() if k in
               ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "NO_PROXY", "no_proxy")}
        env.update(AHE_RPC_READ_FD=str(solver_read), AHE_RPC_WRITE_FD=str(solver_write))
        env.update(PYTHONPATH="/adapter:/upstream", PATH="/runtime/bin:/usr/bin:/bin",
                   HOME=logical + "/workspace/.home", TMPDIR="/tmp")
        self.log = (self.trace_dir / "nexau.txt").open("w")
        self.isolated_case = case
        self.process = subprocess.Popen(["/runtime/bin/python", "-u", "-m", "ahe_native.nexau_worker",
            "--workspace", logical + "/workspace", "--trace-dir", logical + "/output",
            "--agent-config", self.agent_config],
            preexec_fn=lambda: enter(str(isolated), logical + "/workspace", uid), env=env,
            pass_fds=(solver_read, solver_write), stdin=subprocess.DEVNULL,
            stdout=self.log, stderr=subprocess.STDOUT)
        os.close(solver_read)
        os.close(solver_write)
        self.read_fd = native_read
        self.writer = os.fdopen(native_write, "w", buffering=1)
        self.buffer = b""
        self.send({"kind": "init", "messages": self.history,
                   "capability_policy": self.capability_policy})
        self.cursor = len(self.history)

    def send(self, value):
        self.writer.write(json.dumps(value, ensure_ascii=False) + "\n")
        self.writer.flush()

    def receive(self, timeout=None):
        with selectors.DefaultSelector() as selector:
            selector.register(self.read_fd, selectors.EVENT_READ)
            while b"\n" not in self.buffer:
                if not selector.select(self.timeout if timeout is None else timeout):
                    raise TimeoutError("NexAU native episode RPC timed out")
                data = os.read(self.read_fd, 65536)
                if not data:
                    raise RuntimeError("NexAU process ended before delivering an action; inspect nexau.txt")
                self.buffer += data
        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line)

    def sync_action(self):
        if self.process is None:
            self.start()
        elif self.pending:
            self.send({"kind": "feedback" if self.pending == "invoke" else "continue",
                       "messages": self.history[self.cursor:]})
            self.cursor = len(self.history)
        value = self.receive()
        kind = value["kind"]
        if kind == "invoke":
            message = {"role": "assistant", "content": "", "tool_calls": [{
                "id": value["id"], "type": "function", "function": {
                    "name": value["name"], "arguments": json.dumps(value["arguments"], ensure_ascii=False)}}]}
        elif kind == "final":
            message = {"role": "assistant", "content": value["content"]}
        else:
            raise RuntimeError(value.get("error", f"unexpected solver event: {kind}"))
        self.turn += 1
        self.history.append(message)
        self.cursor = len(self.history)
        self.pending = kind
        return SimpleNamespace(messages=[message])

    async def action(self):
        return self.sync_action()

    def close(self):
        if self.process is None:
            return
        try:
            self.send({"kind": "stop"})
            self.process.wait(timeout=15)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self.process.terminate()
            self.process.wait(timeout=10)
        finally:
            os.close(self.read_fd)
            try:
                self.writer.close()
            except BrokenPipeError:
                pass
            self.log.close()
            import shutil
            for path in (self.isolated_case / "output").glob("*.json"):
                shutil.copy2(path, self.trace_dir / path.name)
            self.process = None
