"""Empty H2: no repair, gate, recovery rule or forced action."""
TARGET_LAYER = 'h2'


class Harness:
    def bind_shared_state(self, state):
        self.state = state

    def bind_task_context(self, context):
        self.context = context

    def h2(self, message, history):
        return message

    def h2_no_tool_message(self, message, remaining):
        return None

    def drain_h2_messages(self):
        return []
