"""Empty H4: no monitor, planning hint, budget policy or recovery rule."""
TARGET_LAYER = 'h4'


class Harness:
    def bind_shared_state(self, state):
        self.state = state

    def bind_task_context(self, context):
        self.context = context

    def h4(self, messages, remaining):
        return []
