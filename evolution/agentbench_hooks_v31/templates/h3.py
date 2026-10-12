"""Empty H3: no tool-description or public-message conditioning."""
TARGET_LAYER = 'h3'


class Harness:
    def bind_shared_state(self, state):
        self.state = state

    def bind_task_context(self, context):
        self.context = context

    def h3(self, tools):
        return tools

    def h3_message(self, message):
        return message
