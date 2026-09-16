"""Identity runtime: the from-scratch Meta-Harness iteration-0 anchor."""

def register():
    return None

class AgentHarness:
    def __init__(self, public_context=None):
        self.public_context = public_context
    def transform_request(self, request):
        return request
    def transform_response(self, response, public_history):
        return response
