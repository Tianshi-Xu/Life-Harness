"""Empty H5: no skill bank, retrieval policy or cold-start instruction."""
TARGET_LAYER = 'h5'


class Harness:
    def bind_shared_state(self, state):
        self.state = state

    def bind_task_context(self, context):
        self.context = context

    def h5(self, messages):
        return []
