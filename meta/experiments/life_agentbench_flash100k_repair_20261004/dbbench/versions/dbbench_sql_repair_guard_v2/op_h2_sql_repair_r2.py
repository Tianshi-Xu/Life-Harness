"""H2: keep the SQL action executable - quote public identifiers, close broken string literals,
and break a failed-statement loop with the recorded MySQL evidence instead of a generic nudge."""
from json import dumps, loads
from re import IGNORECASE, escape, finditer

TARGET_LAYER = 'h2'

IDENTIFIER_CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'

RESERVED_COLUMNS = (
    'rank', 'rows', 'row', 'groups', 'over', 'window', 'recursive', 'system',
    'of', 'lead', 'lag', 'current', 'usage', 'condition', 'return', 'function',
    'array', 'cube', 'except', 'intersect', 'lateral',
)

NUDGE = (
    'No tool call was detected, so nothing was executed. Call execute_sql with exactly one '
    'single-line SQL statement, wrapping the table name and every column name in backticks, '
    'or call commit_final_answer if the result is already known.'
)

SYNTAX_ADVICE = ('Close every string value and backtick-quote every table and column name, then change '
                 'the failing expression instead of resubmitting an identical statement.')
DEFAULT_ADVICE = 'Change the failing expression instead of resubmitting an identical statement.'


class Harness:
    state = None
    context = None

    def bind_shared_state(self, state):
        self.state = state

    def bind_task_context(self, context):
        self.context = context

    def h2(self, message, history):
        if not isinstance(message, dict):
            return message
        calls = message.get('tool_calls')
        if not isinstance(calls, list) or not calls:
            return message
        table, columns = self._identifiers()
        guard = self._guard()
        repaired = []
        changed = False
        repeated = False
        for call in calls:
            replacement = call
            if isinstance(call, dict):
                function = call.get('function')
                if isinstance(function, dict) and function.get('name') == 'execute_sql':
                    query = _query_of(function.get('arguments'))
                    if isinstance(query, str):
                        if isinstance(guard, dict) and guard.get('signature') == _signature(query):
                            repeated = True
                        better = repair_query(query, table, columns)
                        if better != query:
                            changed = True
                            replacement = dict(call)
                            replacement['function'] = dict(function)
                            replacement['function']['arguments'] = dumps({'query': better})
            repaired.append(replacement)
        if changed:
            self._clear_guard()
            result = dict(message)
            result['tool_calls'] = repaired
            return result
        repeats = guard.get('repeats') if isinstance(guard, dict) else None
        if repeated and isinstance(repeats, int) and repeats >= 1:
            self._clear_guard()
            result = dict(message)
            result['_harness_control'] = {
                'action': 'suppress_tool',
                'messages': [{'role': 'user', 'content': _loop_message(guard)}],
            }
            return result
        return message

    def h2_no_tool_message(self, message, remaining):
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            return None
        if message.get('tool_calls'):
            return None
        if isinstance(remaining, int) and remaining < 1:
            return None
        return NUDGE

    def _identifiers(self):
        table, columns = _context_identifiers(self.context)
        if (not table) and not columns and isinstance(self.state, dict):
            stored = self.state.get('sql_identifiers')
            if isinstance(stored, dict):
                if isinstance(stored.get('table'), str):
                    table = stored['table']
                if isinstance(stored.get('columns'), list):
                    columns = [name for name in stored['columns'] if isinstance(name, str)]
        return table, columns

    def _guard(self):
        if isinstance(self.state, dict):
            guard = self.state.get('sql_loop_guard')
            if isinstance(guard, dict):
                return guard
        return None

    def _clear_guard(self):
        if isinstance(self.state, dict):
            self.state.pop('sql_loop_guard', None)


def _query_of(arguments):
    if not isinstance(arguments, str):
        return None
    try:
        value = loads(arguments)
    except ValueError:
        return None
    if isinstance(value, dict) and isinstance(value.get('query'), str):
        return value['query']
    return None


def _signature(query):
    return ' '.join(query.split()).lower()


def repair_query(query, table, columns):
    """Double the word-internal apostrophes of an unbalanced statement, then quote identifiers."""
    if not isinstance(query, str) or not query:
        return query
    return _quote_identifiers(_close_literals(query), table, columns)


def _scan(query):
    """Mask the quoted/backticked spans and collect the unescaped single quotes of the statement."""
    mask = [False] * len(query)
    apostrophes = []
    state = ''
    index = 0
    length = len(query)
    while index < length:
        character = query[index]
        if state:
            mask[index] = True
            if character == '\\' and index + 1 < length:
                mask[index + 1] = True
                index += 2
                continue
            if character == state:
                if index + 1 < length and query[index + 1] == state:
                    mask[index + 1] = True
                    index += 2
                    continue
                state = ''
        else:
            if character == '`' or character == '"':
                state = character
                mask[index] = True
            elif character == "'":
                apostrophes.append(index)
        index += 1
    return mask, apostrophes


def _close_literals(query):
    if not isinstance(query, str) or "'" not in query:
        return query
    mask, apostrophes = _scan(query)
    if len(apostrophes) % 2 == 0:
        return query
    fixes = []
    for position in apostrophes:
        if position == 0 or position + 1 >= len(query) or mask[position]:
            continue
        if query[position - 1].isalnum() and query[position + 1].isalnum():
            fixes.append(position)
    if not fixes or len(fixes) % 2 == 0:
        return query
    result = query
    for position in reversed(fixes):
        result = result[:position] + "'" + result[position:]
    return result


def _advice(kind, table):
    if kind == 'unknown_column':
        if isinstance(table, str) and table:
            return ('Use the exact column names printed in the schema card, and if a name is still '
                    'rejected run one SHOW COLUMNS FROM `' + table + '`; probe and rewrite the statement '
                    'with the names it returns.')
        return ('Use the exact column names printed in the schema card, and if a name is still rejected '
                'run one SHOW COLUMNS probe and rewrite the statement with the names it returns.')
    if kind == 'missing_table':
        if isinstance(table, str) and table:
            return ('Use the exact table name `' + table + '` from the schema card and change the failing '
                    'expression instead of resubmitting an identical statement.')
        return ('Use the exact table name from the schema card and change the failing expression instead '
                'of resubmitting an identical statement.')
    if kind == 'syntax':
        return SYNTAX_ADVICE
    return DEFAULT_ADVICE


def _loop_message(guard):
    kind = guard.get('kind') if isinstance(guard, dict) else None
    error = guard.get('error') if isinstance(guard, dict) else None
    table = guard.get('table') if isinstance(guard, dict) else None
    parts = ['This statement already failed and MySQL will reject it again.']
    if isinstance(error, str) and error:
        parts.append('MySQL said: ' + error + '.')
    parts.append(_advice(kind, table))
    parts.append('Do not resubmit it unchanged.')
    return ' '.join(parts)


def _context_identifiers(context):
    table = None
    columns = []
    if isinstance(context, dict):
        table_map = context.get('table')
        if isinstance(table_map, dict):
            if isinstance(table_map.get('table_name'), str):
                table = table_map['table_name']
            info = table_map.get('table_info')
            if isinstance(info, dict):
                listed = info.get('columns')
                if isinstance(listed, list):
                    for column in listed:
                        if isinstance(column, dict) and isinstance(column.get('name'), str):
                            columns.append(column['name'])
        text = context.get('add_description')
        if isinstance(text, str):
            if not table:
                marker = text.find('table is ')
                if marker >= 0:
                    head = text[marker + 9:]
                    stop = head.find(', and')
                    table = (head[:stop] if stop >= 0 else head).strip()
            if not columns and 'headers of this table are' in text:
                head = text.split('headers of this table are')[-1].strip()
                if head.endswith('.'):
                    head = head[:-1]
                columns = [piece.strip() for piece in head.split(',') if piece.strip()]
    return table, columns


def _needs_quote(name):
    if not isinstance(name, str) or not name:
        return False
    for character in name:
        if character not in IDENTIFIER_CHARS:
            return True
    return name.lower() in RESERVED_COLUMNS


def _protected(query):
    mask = []
    quote = None
    for character in query:
        if quote is None:
            if character in '\'"`':
                quote = character
                mask.append(True)
            else:
                mask.append(False)
        else:
            mask.append(True)
            if character == quote:
                quote = None
    return mask


def _refuses(query, end):
    stripped = query[end:].lstrip()
    if stripped.startswith('('):
        return True
    if stripped[:2].upper() == 'BY':
        after = stripped[2:3]
        if (not after) or (after not in IDENTIFIER_CHARS):
            return True
    return False


def _quote_identifiers(query, table, columns):
    if not isinstance(query, str) or not query:
        return query
    candidates = []
    if isinstance(table, str) and table:
        candidates.append(table)
    if isinstance(columns, list):
        candidates.extend(columns)
    names = []
    for name in candidates:
        if isinstance(name, str) and name and name not in names and _needs_quote(name):
            names.append(name)
    if not names:
        return query
    mask = _protected(query)
    spans = []
    for name in names:
        pattern = '(?<![A-Za-z0-9_`])' + escape(name) + '(?![A-Za-z0-9_`])'
        for found in finditer(pattern, query, IGNORECASE):
            start = found.start()
            end = found.end()
            inside = False
            for index in range(start, end):
                if mask[index]:
                    inside = True
            if inside or _refuses(query, end):
                continue
            spans.append((start, end, name))
    spans.sort()
    chosen = []
    last = -1
    for start, end, name in spans:
        if start >= last:
            chosen.append((start, end, name))
            last = end
    if not chosen:
        return query
    result = query
    for start, end, name in reversed(chosen):
        result = result[:start] + '`' + name + '`' + result[end:]
    return result
