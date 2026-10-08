"""Episode-local environment RPC called by evolving NexAU tool implementations."""
import json
import os
import threading


class EpisodeFinished(BaseException):
    pass


class Channel:
    def __init__(self, read_fd, write_fd):
        self.reader = os.fdopen(read_fd, "r", buffering=1)
        self.writer = os.fdopen(write_fd, "w", buffering=1)
        self.lock = threading.RLock()
        self.counter = 0

    def send(self, message):
        self.writer.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.writer.flush()

    def receive(self):
        line = self.reader.readline()
        if not line:
            raise EpisodeFinished()
        value = json.loads(line)
        if value.get("kind") == "stop":
            raise EpisodeFinished()
        return value

    def invoke(self, name, arguments):
        # Native environments are sequential. Local/internal NexAU tools and
        # subagents retain their original execution path.
        with self.lock:
            budget = getattr(self, "dispatch_budget", None)
            if budget is not None:
                budget.consume_tool(name)
            self.counter += 1
            self.send({"kind": "invoke", "id": f"native_{self.counter}",
                       "name": name, "arguments": arguments})
            response = self.receive()
            if response.get("kind") != "feedback":
                raise RuntimeError("unexpected native environment response")
            messages = response["messages"]
            tool_outputs = [m.get("content", "") for m in messages if m["role"] == "tool"]
            other_outputs = [m.get("content", "") for m in messages if m["role"] != "tool"]
            if other_outputs:
                return json.dumps({"tool_results": tool_outputs,
                                   "environment_messages": other_outputs}, ensure_ascii=False)
            return "\n".join(str(x) for x in tool_outputs)


CHANNEL = None


def invoke(name, arguments):
    if CHANNEL is None:
        raise RuntimeError("native environment is not connected")
    return CHANNEL.invoke(name, arguments)
