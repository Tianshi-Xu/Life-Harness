"""H3: publish the public schema card with the identifier rule, the preview rule and the
value-convention rule for the task at hand."""

TARGET_LAYER = 'h3'

CARD_TITLE = 'SCHEMA CARD (from the public task context).'
SAMPLE_TITLE = 'sample rows (preview of the first rows only):'
DESC_ADDENDUM = (' Submit exactly one single-line statement and wrap the table name and every column '
                 'name in backticks (for example `No.`, `Shirt sponsor`, `Rank`).')
IDENTIFIER_RULE = ('Identifier rule: in every SQL statement, write the table name and every column name as a '
                   'backtick-quoted identifier exactly as listed above; this covers names containing spaces or '
                   'dots (for example "No.") and columns that are MySQL keywords (for example "Rank").')
PREVIEW_RULE = ('Preview rule: the sample rows above are only the first rows of the table, never the whole '
                'table; before you choose a value that has to continue an existing sequence (the next rank, '
                'episode number or id), read that column back from the database with one statement and build '
                'the value from what it returns.')
VALUE_RULE = ('Value rule: reproduce the convention of the same column in the sample rows - keep a leading '
              'word such as "elected", a trailing period, the date format and plain digits without added '
              'thousands separators - and when the task statement says the values are provided as strings, '
              'quote every value including numbers and years.')


class Harness:
    state = None
    context = None
    conditioned = False

    def bind_shared_state(self, state):
        self.state = state

    def bind_task_context(self, context):
        self.context = context

    def h3(self, tools):
        if not isinstance(tools, list):
            return tools
        result = []
        for tool in tools:
            if isinstance(tool, dict):
                function = tool.get('function')
                if (isinstance(function, dict) and function.get('name') == 'execute_sql'
                        and isinstance(function.get('description'), str)):
                    tool = dict(tool)
                    tool['function'] = dict(function)
                    tool['function']['description'] = function['description'] + DESC_ADDENDUM
            result.append(tool)
        return result

    def h3_message(self, message):
        if not isinstance(message, dict) or message.get('role') != 'user':
            return message
        if self.conditioned:
            return message
        content = message.get('content')
        if not isinstance(content, str):
            return message
        table, columns, kinds, rows = self._schema()
        if not table and not columns:
            return message
        self.conditioned = True
        if isinstance(self.state, dict):
            self.state['sql_identifiers'] = {'table': table, 'columns': list(columns)}
        result = dict(message)
        result['content'] = content + _card(table, columns, kinds, rows)
        return result

    def _schema(self):
        table, columns = _context_identifiers(self.context)
        kinds = {}
        rows = []
        context = self.context
        if isinstance(context, dict):
            table_map = context.get('table')
            if isinstance(table_map, dict):
                info = table_map.get('table_info')
                if isinstance(info, dict):
                    listed = info.get('columns')
                    if isinstance(listed, list):
                        for column in listed:
                            if isinstance(column, dict) and isinstance(column.get('name'), str):
                                label = column.get('type')
                                kinds[column['name']] = label if isinstance(label, str) else ''
                    listed_rows = info.get('rows')
                    if isinstance(listed_rows, list):
                        rows = listed_rows
        return table, columns, kinds, rows


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


def _samples(rows):
    samples = []
    for row in rows:
        if not isinstance(row, list):
            continue
        cells = []
        for cell in row:
            if isinstance(cell, str):
                text = cell
            elif cell is None:
                text = ''
            else:
                text = str(cell)
            if len(text) > 40:
                text = text[:40]
            cells.append(text)
        samples.append(' | '.join(cells))
        if len(samples) >= 3:
            break
    return samples


def _card(table, columns, kinds, rows):
    lines = [CARD_TITLE]
    if isinstance(table, str) and table:
        lines.append('table: `' + table + '`')
    if columns:
        listed = []
        for name in columns:
            label = kinds.get(name)
            listed.append('`' + name + '`' + ((' ' + label) if label else ''))
        lines.append('columns: ' + ', '.join(listed))
    samples = _samples(rows)
    if samples:
        lines.append(SAMPLE_TITLE)
        for sample in samples:
            lines.append('  - ' + sample)
    lines.append(IDENTIFIER_RULE)
    lines.append(PREVIEW_RULE)
    lines.append(VALUE_RULE)
    return '\n\n' + '\n'.join(lines)
