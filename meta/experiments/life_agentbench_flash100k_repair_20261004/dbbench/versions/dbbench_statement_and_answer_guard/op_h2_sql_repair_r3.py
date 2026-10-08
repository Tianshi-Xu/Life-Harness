"""H2: keep the SQL action executable - close a broken string literal, quote public identifiers, replace the
line break (or backslash escape) inside a public header with the real database name, and suppress a modifying
statement that the public history already shows as successfully executed."""
from json import dumps, loads
from re import IGNORECASE, escape, finditer

TARGET_LAYER = 'h2'

IDENTIFIER_CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'

RESERVED_COLUMNS = (
    'rank', 'rows', 'row', 'groups', 'over', 'window', 'recursive', 'system',
    'of', 'lead', 'lag', 'current', 'usage', 'condition', 'return', 'function',
    'array', 'cube', 'except', 'intersect', 'lateral',
)

MUTATION_HEADS = ('insert', 'update', 'delete', 'replace')

ERROR_MARKERS = (
    '1064', '1054', '1146', 'sql syntax', 'unknown column', "doesn't exist",
    'does not exist', 'error', 'failed', 'denied',
)

NUDGE = (
    'No tool call was detected, so nothing was executed. Call execute_sql with exactly one '
    'single-line SQL statement, wrapping the table name and every column name in backticks, '
    'or call commit_final_answer if the result is already known.'
)

SYNTAX_ADVICE = ('Close every string value and backtick-quote every table and column name, then change '
                 'the failing expression instead of resubmitting an identical statement.')
DEFAULT_ADVICE = 'Change the failing expression instead of resubmitting an identical statement.'

APPLIED_MESSAGE = ('This exact statement was already executed successfully, so running it again would apply the '
                   'same change a second time (an INSERT would add a duplicate row). It has not been executed '
                   'again. Confirm the current rows with one SELECT over the same filter, then commit_final_answer.')


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
        table, columns, renames = self._schema()
        guard = self._guard()
        repaired = []
        changed = False
        repeated = False
        applied = False
        for call in calls:
            replacement = call
            if isinstance(call, dict):
                function = call.get('function')
                if isinstance(function, dict) and function.get('name') == 'execute_sql':
                    query = _query_of(function.get('arguments'))
                    if isinstance(query, str):
                        signature = _signature(query)
                        if isinstance(guard, dict) and guard.get('signature') == signature:
                            repeated = True
                        if (not applied and _mutation_head(query) is not None
                                and _already_applied(history, signature)):
                            applied = True
                        better = repair_query(query, table, columns, renames)
                        if better != query:
                            changed = True
                            replacement = dict(call)
                            replacement['function'] = dict(function)
                            replacement['function']['arguments'] = dumps({'query': better}, ensure_ascii=False)
            repaired.append(replacement)
        if changed:
            self._clear_guard()
            result = dict(message)
            result['tool_calls'] = repaired
            return result
        if applied:
            result = dict(message)
            result['_harness_control'] = {
                'action': 'suppress_tool',
                'messages': [{'role': 'user', 'content': APPLIED_MESSAGE}],
            }
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

    def _schema(self):
        table, columns, renames = _context_schema(self.context)
        if not columns and isinstance(self.state, dict):
            stored = self.state.get('sql_identifiers')
            if isinstance(stored, dict):
                if not table and isinstance(stored.get('table'), str):
                    table = stored['table']
                listed = stored.get('columns')
                if isinstance(listed, list):
                    columns = [name for name in listed if isinstance(name, str)]
                pairs = stored.get('renames')
                if isinstance(pairs, list):
                    for pair in pairs:
                        if (isinstance(pair, list) and len(pair) == 2
                                and isinstance(pair[0], str) and isinstance(pair[1], str)
                                and pair[0] != pair[1]):
                            renames.append((pair[0], pair[1]))
        return table, columns, renames

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


def _mutation_head(query):
    if not isinstance(query, str):
        return None
    pieces = query.split()
    if not pieces:
        return None
    head = pieces[0].lower()
    if head in MUTATION_HEADS:
        return head
    return None


def _error_text(text):
    lowered = text.lower()
    for marker in ERROR_MARKERS:
        if marker in lowered:
            return True
    return False


def _already_applied(history, signature):
    """True when an identical execute_sql call appears earlier in the public history and its tool result was
    not an error message."""
    if not isinstance(history, list) or not signature:
        return False
    for index in range(len(history)):
        message = history[index]
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            continue
        calls = message.get('tool_calls')
        if not isinstance(calls, list):
            continue
        match = False
        for call in calls:
            if isinstance(call, dict):
                function = call.get('function')
                if isinstance(function, dict) and function.get('name') == 'execute_sql':
                    query = _query_of(function.get('arguments'))
                    if isinstance(query, str) and _signature(query) == signature:
                        match = True
        if not match:
            continue
        for follower in history[index + 1:]:
            if isinstance(follower, dict) and follower.get('role') == 'tool':
                text = follower.get('content')
                if isinstance(text, str) and not _error_text(text):
                    return True
                break
    return False


def _card_name(name):
    """The real database name behind a public header that carries a line break or a backslash escape."""
    if not isinstance(name, str) or not name:
        return name
    fixed = name
    if '\\' in name:
        replaced = name.replace('\\n', ' ').replace('\\r', ' ').replace('\\t', ' ')
        if '\\' not in replaced:
            fixed = replaced
    cleaned = ' '.join(fixed.split())
    if not cleaned:
        return name
    return cleaned


def _context_schema(context):
    raw_table = None
    raw_columns = []
    if isinstance(context, dict):
        table_map = context.get('table')
        if isinstance(table_map, dict):
            if isinstance(table_map.get('table_name'), str):
                raw_table = table_map['table_name']
            info = table_map.get('table_info')
            if isinstance(info, dict):
                listed = info.get('columns')
                if isinstance(listed, list):
                    for column in listed:
                        if isinstance(column, dict) and isinstance(column.get('name'), str):
                            raw_columns.append(column['name'])
        text = context.get('add_description')
        if isinstance(text, str):
            if not raw_table:
                marker = text.find('table is ')
                if marker >= 0:
                    head = text[marker + 9:]
                    stop = head.find(', and')
                    raw_table = (head[:stop] if stop >= 0 else head).strip()
            if not raw_columns and 'headers of this table are' in text:
                head = text.split('headers of this table are')[-1].strip()
                if head.endswith('.'):
                    head = head[:-1]
                raw_columns = [piece.strip() for piece in head.split(',') if piece.strip()]
    table = _card_name(raw_table) if isinstance(raw_table, str) else None
    columns = []
    renames = []
    for raw in raw_columns:
        card = _card_name(raw)
        columns.append(card)
        if card != raw:
            renames.append((raw, card))
    return table, columns, renames


def repair_query(query, table, columns, renames):
    """Replace a broken public header with the real name, double the word-internal apostrophes of an unbalanced
    statement, then quote identifiers."""
    if not isinstance(query, str) or not query:
        return query
    renamed = _rename_identifiers(_close_literals(query), renames)
    return _quote_identifiers(renamed, table, columns)


def _quoting(query):
    marks = [''] * len(query)
    quote = ''
    for index in range(len(query)):
        character = query[index]
        if quote:
            marks[index] = quote
            if character == quote:
                quote = ''
        elif character == "'" or character == '"' or character == '`':
            quote = character
            marks[index] = character
    return marks


def _rename_identifiers(query, renames):
    if not isinstance(query, str) or not query or not renames:
        return query
    marks = _quoting(query)
    spans = []
    for raw, card in renames:
        if not raw or not card or raw == card:
            continue
        pattern = '(?<![A-Za-z0-9_])' + escape(raw) + '(?![A-Za-z0-9_])'
        for found in finditer(pattern, query, IGNORECASE):
            start = found.start()
            end = found.end()
            mark = marks[start]
            if mark == "'" or mark == '"':
                continue
            if mark == '`':
                spans.append((start, end, card))
            else:
                spans.append((start, end, '`' + card + '`'))
    if not spans:
        return query
    spans.sort()
    chosen = []
    last = -1
    for start, end, text in spans:
        if start >= last:
            chosen.append((start, end, text))
            last = end
    result = query
    for start, end, text in reversed(chosen):
        result = result[:start] + text + result[end:]
    return result


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
            if character == "'" or character == '"' or character == '`':
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
