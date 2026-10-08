"""H3: record the agent tool contract in shared state and note lookup/ID formats."""
TARGET_LAYER = 'h3'

NOTES = {
    'get_customer_by_phone': (
        " Pass the phone number exactly as the customer states it, including dashes and any other "
        "formatting (for example 555-123-2002); do not strip punctuation."),
    'get_customer_by_name': (
        " Use the date of birth the customer provides; never invent or guess one."),
    'get_details_by_id': (
        " A line ID returns that line's status, suspension start date, contract end date, roaming "
        "flag and data usage; look up the line attached to the number the customer gave before "
        "concluding that the fault is on the device."),
}


class Harness:
    state = None
    context = None

    def bind_shared_state(self, state):
        self.state = state

    def bind_task_context(self, context):
        self.context = context

    def h3(self, tools):
        if not isinstance(tools, list):
            return []
        names = []
        for tool in tools:
            if not isinstance(tool, dict) or not isinstance(tool.get('function'), dict):
                continue
            function = tool['function']
            name = function.get('name')
            if not isinstance(name, str):
                continue
            names.append(name)
            note = NOTES.get(name)
            if note:
                head = function.get('description')
                function['description'] = (head if isinstance(head, str) else '') + note
        if isinstance(self.state, dict):
            self.state['agent_tool_names'] = names
        return tools
