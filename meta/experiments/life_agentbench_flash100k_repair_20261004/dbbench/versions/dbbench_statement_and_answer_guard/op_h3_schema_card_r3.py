"""H3: publish the public schema card with the de-escaped identifiers, the reported sample rows as a preview,
the task-type line and the standing statement rules (identifier, numeric cast, boolean bracketing,
literal-to-column, modify-then-verify) for the task at hand."""

TARGET_LAYER = 'h3'

CARD_TITLE = 'SCHEMA CARD (from the public task context).'
SAMPLE_TITLE = 'sample rows (preview of the first rows only):'
DESC_ADDENDUM = (' Submit exactly one single-line statement and wrap the table name and every column '
                 'name in backticks (for example `No.`, `Shirt sponsor`, `Rank`).')
IDENTIFIER_RULE = ("Identifier rule: in every SQL statement write the table name and every column name as a "
                   "backtick-quoted identifier exactly as printed above; a header of the public task information "
                   "that contains a line break or a backslash escape such as \\n comes from a line break of the "
                   "source document, and the real name in this database is the same text with that break "
                   "replaced by a space.")
NUMERIC_RULE = ("Numeric rule: the type label printed next to a column is only the label of the source table; in "
                "this database most values arrive as text, and ORDER BY, MAX, MIN, SUM and AVG on such a column "
                "compare character strings (ORDER BY `Total` ASC can return the text '12' before '3', and "
                "SUM(`Votes`) over values like '7,660' can return 7). Whenever you order or aggregate a column "
                "whose values look numeric, force a number first: use CAST(REPLACE(`Total`, ',', '') AS INTEGER) "
                "for whole numbers, and CAST(REPLACE(`Amount`, ',', '') AS REAL) when the values carry a "
                "decimal point.")
BOOLEAN_RULE = ("Boolean rule: never rely on the default precedence of AND over OR - bracket every OR group "
                "explicitly, so WHERE `Introduced` = '1997' AND (`Retired` IS NULL OR `Retired` = '') changes "
                "only the rows of 1997.")
FILTER_RULE = ("Filter rule: filter on the column whose printed rows actually contain the literal of the "
               "question (the level '-6' sits in `To par`, not in `Score`); if that literal never appears in "
               "the chosen column, the filter matches nothing and a modifying statement silently changes no "
               "row.")
MODIFY_RULE = ("Modify rule: INSERT, UPDATE and DELETE return no rows, so an empty reply is normal and says "
               "nothing about the effect. After one modifying statement, run one SELECT with the same filter "
               "to confirm which rows changed, never execute an identical modifying statement twice, and only "
               "then commit.")
INSERT_LINE = ("Task type line: this task adds rows to the table, so the graded effect is the database change - "
               "build one INSERT statement with the exact column names and the values described by the task, "
               "instead of answering that the described entry cannot be found.")
UPDATE_LINE = ("Task type line: this task changes rows that already exist, so the graded effect is the database "
               "change - make the change with one statement and confirm it with one SELECT instead of only "
               "describing it.")
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
        table, columns, renames, kinds, rows, types = self._schema()
        if not table and not columns:
            return message
        self.conditioned = True
        if isinstance(self.state, dict):
            self.state['sql_identifiers'] = {
                'table': table,
                'columns': list(columns),
                'renames': [[raw, card] for raw, card in renames],
            }
        result = dict(message)
        result['content'] = content + _card(table, columns, kinds, rows, types)
        return result

    def _schema(self):
        table, columns, renames = _context_schema(self.context)
        kinds = {}
        rows = []
        types = []
        context = self.context
        if isinstance(context, dict):
            listed_types = context.get('type')
            if isinstance(listed_types, list):
                types = listed_types
            table_map = context.get('table')
            if isinstance(table_map, dict):
                info = table_map.get('table_info')
                if isinstance(info, dict):
                    listed = info.get('columns')
                    if isinstance(listed, list):
                        for column in listed:
                            if isinstance(column, dict) and isinstance(column.get('name'), str):
                                label = column.get('type')
                                kinds[_card_name(column['name'])] = label if isinstance(label, str) else ''
                    listed_rows = info.get('rows')
                    if isinstance(listed_rows, list):
                        rows = listed_rows
        return table, columns, renames, kinds, rows, types


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


def _type_lines(types):
    if not isinstance(types, list):
        return []
    lowered = []
    for item in types:
        if isinstance(item, str):
            lowered.append(item.lower())
    for item in lowered:
        if 'insert' in item:
            return [INSERT_LINE]
    for item in lowered:
        if 'update' in item or 'delete' in item:
            return [UPDATE_LINE]
    return []


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


def _card(table, columns, kinds, rows, types):
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
    for line in _type_lines(types):
        lines.append(line)
    lines.append(IDENTIFIER_RULE)
    lines.append(NUMERIC_RULE)
    lines.append(BOOLEAN_RULE)
    lines.append(FILTER_RULE)
    lines.append(MODIFY_RULE)
    lines.append(PREVIEW_RULE)
    lines.append(VALUE_RULE)
    return '\n\n' + '\n'.join(lines)
