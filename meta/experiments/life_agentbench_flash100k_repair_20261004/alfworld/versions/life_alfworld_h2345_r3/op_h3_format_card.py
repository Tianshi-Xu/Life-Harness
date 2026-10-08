"""H3 ALFWorld format card: verbatim-argument description plus the goal protocol note."""
from copy import deepcopy

TARGET_LAYER = 'h3'

DESC_SUFFIX = (' Submit one action string copied verbatim from the most recent AVAILABLE ACTIONS '
               'list; anything outside that list changes nothing.')
SYSTEM_NOTE = (
    '\n\nAction protocol: one take_action call per turn, its action copied verbatim from the newest '
    'AVAILABLE ACTIONS list, and an action outside that list changes nothing. The task names one '
    'goal object type and one goal receptacle type, plus a count if it says two: a count of two '
    'needs two different objects of the named type in the receptacle, and a closed receptacle must '
    'be opened before an object is put into it. Never repeat an action whose observation did not '
    'change; switch to a different action from the newest list instead. For a look-at or examine '
    'goal the step is to turn on the lamp named in the task while holding the goal object; '
    'examining the goal object itself is not that step.')
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
