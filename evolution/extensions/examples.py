"""Offline integration demo: strict tool arguments and a scripted repair bundle.

Neither the solver nor the proposal provider is an LLM. Scores demonstrate
plumbing on this toy environment, not Life-Harness benchmark performance.
"""
import json

from .adapter import EpisodeSetup, ModelResponse, StepResult
from .records import save


class ToyEpisode:
    def __init__(self, quantity):
        self.quantity, self.accepted = quantity, False
        instruction = f"Submit quantity {quantity} using the submit tool."
        self.setup = EpisodeSetup(
            messages=[{"role": "user", "content": instruction}],
            tools=[{"type": "function", "function": {"name": "submit", "description": "Submit an integer quantity.",
                "parameters": {"type": "object", "properties": {"quantity": {"type": "integer"}},
                               "required": ["quantity"], "additionalProperties": False}}}],
            public_context={"instruction": instruction}, max_turns=2)

    def complete(self, messages, tools):
        # Read only the public task instruction; deliberately demonstrate a type mismatch.
        instruction = next(m["content"] for m in messages if m.get("content", "").startswith("Submit quantity "))
        quantity = instruction.split()[2]
        return ModelResponse({"role": "assistant", "tool_calls": [{"id": "submit_call", "type": "function",
            "function": {"name": "submit", "arguments": json.dumps({"quantity": quantity})}}]},
            usage_source="local_no_api")

    def execute(self, message):
        try:
            call = message["tool_calls"][0]
            arguments = json.loads(call["function"]["arguments"])
            self.accepted = (call["function"]["name"] == "submit" and set(arguments) == {"quantity"}
                             and type(arguments["quantity"]) is int and arguments["quantity"] == self.quantity)
            feedback = "Submitted." if self.accepted else "Validation error: quantity must be an integer."
            return StepResult([{"role": "tool", "tool_call_id": call["id"], "content": feedback}], done=True)
        except (KeyError, IndexError, ValueError):
            return StepResult([{"role": "user", "content": "Invalid action."}], done=True)

    def grade(self):
        return self.accepted

    def close(self):
        pass


class ToyAdapter:
    name = "extension_toy"
    public_context_fields = ("instruction",)
    tasks = {"train": {"train_two": 2, "train_four": 4}, "test": {"test_seven": 7, "test_nine": 9}}

    def task_ids(self, split):
        return list(self.tasks[split])

    def open_episode(self, task_id, split):
        return ToyEpisode(self.tasks[split][task_id])

    def fingerprint(self):
        return {"version": 1, "tasks": self.tasks, "solver": "local_scripted_no_api", "max_turns": 2}


def offline_proposer(workspace, output, round_number):
    inventory = json.loads((workspace / "inventory.json").read_text())
    operation = {"operation_id": "coerce_quantity", "mechanism_id": "integer_representation",
        "op": "modify" if inventory["records"] else "add", "target_layer": "h2",
        "hypothesis": "Convert numeric string quantities into the declared native integer representation.",
        "artifact": "artifacts/integer_quantity.py"}
    save(output / "bundle.json", {"schema_version": 2, "bundle_id": f"demo_round_{round_number:02d}",
        "hypothesis": operation["hypothesis"], "operations": [operation]})
    (output / "artifacts").mkdir()
    (output / operation["artifact"]).write_text('''from json import loads, dumps
TARGET_LAYER = "h2"
class Harness:
    def h2(self, message, history):
        for call in message.get("tool_calls", []):
            arguments = loads(call["function"]["arguments"])
            quantity = arguments.get("quantity")
            if isinstance(quantity, str) and quantity.isdecimal():
                arguments["quantity"] = int(quantity)
                call["function"]["arguments"] = dumps(arguments)
        return message
''')
    return output / "bundle.json"
