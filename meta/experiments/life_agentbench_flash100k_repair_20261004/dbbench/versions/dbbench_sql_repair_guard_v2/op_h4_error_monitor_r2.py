"""H4: watch new public tool results, record the failing statement with its MySQL evidence in shared
state, and steer one concrete repair."""
from json import loads

TARGET_LAYER = 'h4'

LABELS = {
    'missing_table': 'the table name is unknown to MySQL',
    'unknown_column': 'a column name is unknown to MySQL',
    'syntax': 'MySQL reported a syntax error',
}

LIMIT = 160


class Harness:
    state = None
    context = None
    cursor = 0
    seen = None

    def bind_shared_state(self, state):
        self.state = state

    def bind_task_context(self, context):
        self.context = context

    def h4(self, messages, remaining):
        if not isinstance(messages, list):
            return []
        cursor = self.cursor
        if not isinstance(cursor, int) or cursor < 0 or cursor > len(messages):
            cursor = 0
        fresh = messages[cursor:]
        self.cursor = len(messages)
        kind = None
        error = None
        for message in fresh:
            if not isinstance(message, dict) or message.get('role') != 'tool':
                continue
            text = message.get('content')
            if not isinstance(text, str):
                continue
            found = _error_kind(text)
            if found is not None:
                kind = found
                error = _trim(text)
        if kind is None:
            self._clear_guard()
            return []
        query = _last_query(messages)
        signature = _signature(query) if isinstance(query, str) else ''
        counts = self.seen
        if not isinstance(counts, dict):
            counts = {}
            self.seen = counts
        count = counts.get(signature, 0) + 1
        counts[signature] = count
        table, columns = self._identifiers()
        self._set_guard(signature, count, kind, error, table)
        return [_hint(kind, count, table, columns, error)]

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

    def _set_guard(self, signature, count, kind, error, table):
        if isinstance(self.state, dict):
            self.state['sql_loop_guard'] = {
                'signature': signature, 'repeats': count, 'kind': kind,
                'error': error, 'table': table,
            }

    def _clear_guard(self):
        if isinstance(self.state, dict):
            self.state.pop('sql_loop_guard', None)


def _trim(text):
    return ' '.join(text.split())[:LIMIT]


def _error_kind(text):
    lowered = text.lower()
    if '1064' in lowered or 'sql syntax' in lowered:
        return 'syntax'
    if '1054' in lowered or 'unknown column' in lowered:
        return 'unknown_column'
    if '1146' in lowered or "doesn't exist" in lowered or 'does not exist' in lowered:
        return 'missing_table'
    return None


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


def _last_query(messages):
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            continue
        calls = message.get('tool_calls')
        if not isinstance(calls, list):
            continue
        for call in reversed(calls):
            if isinstance(call, dict):
                function = call.get('function')
                if isinstance(function, dict) and function.get('name') == 'execute_sql':
                    query = _query_of(function.get('arguments'))
                    if isinstance(query, str):
                        return query
    return None


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
        return ('Close every string value and backtick-quote every table and column name, then change the '
                'failing expression instead of resubmitting an identical statement.')
    return 'Change the failing expression instead of resubmitting an identical statement.'


def _hint(kind, count, table, columns, error):
    label = LABELS.get(kind, 'MySQL rejected the statement')
    if isinstance(count, int) and count > 1:
        head = ('Recovery: this same statement has now failed ' + str(count) + ' times - ' + label + '.')
    else:
        head = 'Recovery: your last SQL statement failed - ' + label + '.'
    parts = [head]
    if isinstance(error, str) and error:
        parts.append('MySQL said: ' + error + '.')
    names = []
    if isinstance(columns, list):
        for name in columns:
            if isinstance(name, str) and name:
                names.append(name)
    if isinstance(table, str) and table:
        listed = ', '.join('`' + name + '`' for name in names[:12])
        line = 'Use the exact public identifiers: table `' + table + '`'
        if listed:
            line = line + '; columns ' + listed
        parts.append(line + '.')
    parts.append(_advice(kind, table))
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
