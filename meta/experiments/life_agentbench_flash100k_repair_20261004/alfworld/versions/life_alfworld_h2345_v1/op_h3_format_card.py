"""H3 ALFWorld format card: one verbatim-argument sentence on take_action, one system-message block."""
from copy import deepcopy

TARGET_LAYER = 'h3'

DESC_SUFFIX = ' Submit one action string copied verbatim from the most recent AVAILABLE ACTIONS list.'
SYSTEM_NOTE = ('\n\nAction protocol: reply with exactly one take_action call per turn and copy its '
               'action argument verbatim from the newest AVAILABLE ACTIONS list; an action outside '
               'that list changes nothing; never repeat a survey action whose observation did not '
               'change.')
MARKERS = ('AVAILABLE ACTIONS', 'Interact with a household')


class Harness:
    def __init__(self):
        self.state = {}
        self.context = None
        self._conditioned = False

    def bind_shared_state(self, state):
        self.state = state

    def bind_task_context(self, context):
        self.context = context

    def h3(self, tools):
        if not isinstance(tools, list):
            return tools
        result = deepcopy(tools)
        for tool in result:
            if not isinstance(tool, dict):
                continue
            function = tool.get('function')
            if not isinstance(function, dict):
                continue
            if function.get('name') != 'take_action':
                continue
            description = function.get('description')
            if not isinstance(description, str):
                continue
            if DESC_SUFFIX.strip() in description:
                continue
            function['description'] = description + DESC_SUFFIX
        return result

    def h3_message(self, message):
        if not isinstance(message, dict) or self._conditioned:
            return message
        if message.get('role') != 'system':
            return message
        content = message.get('content')
        if not isinstance(content, str):
            return message
        if not any(marker in content for marker in MARKERS):
            return message
        message['content'] = content + SYSTEM_NOTE
        self._conditioned = True
        return message
