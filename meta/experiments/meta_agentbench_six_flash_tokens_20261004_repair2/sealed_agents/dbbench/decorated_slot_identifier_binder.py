"""Meta-Harness candidate: canonicalize `execute_sql` identifiers to the episode's own schema.

Three cooperating rewrite steps, all applied to SQL *code* regions only (single-quoted and
backticked regions stay byte-identical, so genuine string literals are never touched):

  pass A (bind) - a pre-emptive, error-free binder.  Any bare single-token identifier and
               any backticked span whose separator-insensitive form uniquely equals the
               separator-insensitive form of a declared/observed schema name is rewritten
               to that true spelling (backticked when it cannot stand bare).  This catches
               spellings the exact-match passes never see, because the solver swapped the
               name's separators (underscores for spaces) or wrote the name in backticks
               with a wrong spelling.  No MySQL error is required first.

  pass S (slot) - a pre-emptive binder for *decorated* table-slot tokens.  A bare token
               that contains a hyphen can never be a legal MySQL identifier: the grammar
               reads `-` as the minus operator, so any statement that uses such a token in
               a table slot (the operand of FROM / INTO / UPDATE / JOIN / TABLE / DESCRIBE /
               DESC / USING) is a guaranteed syntax error (1064), and a solver that keeps
               re-issuing it burns the episode to the turn cap.  When the token's
               separator-insensitive fingerprint resolves uniquely to one of the episode's
               own true names (an exact fingerprint match, or a fingerprint that prefixes
               exactly one longer true-name fingerprint, minimum length 3), the token is
               that name mis-written and is rewritten to the true spelling in backticks.
               Because the operand could never have parsed, this can only turn a guaranteed
               failure into a valid statement; it can never invalidate a working query.

  pass 0 (rescue) - a token written in double quotes is, under MySQL's default sql_mode
               (ANSI_QUOTES off), a *string literal*, not an identifier.  When a
               double-quoted span (a) is being used as a boolean-predicate operand - it is
               introduced by WHERE / AND / OR / NOT / ON / HAVING, i.e. it sits where the
               grammar wants an identifier expression - and (b) its text is exactly one of
               the episode's own schema names and that name cannot stand bare (it contains
               a non-word character or is a reserved word), the span is almost certainly an
               identifier that was mis-quoted; it is rewritten to the true name in
               backticks.  Restricting the rewrite to predicate operands keeps double-quoted
               text that may be a genuine literal (right of a comparison / LIKE / IN /
               REGEXP / BETWEEN) and identifier text in non-predicate slots (assignment
               targets, CASE arms, column lists) byte-identical.

  pass 1 (quote) - every true name that cannot stand bare in MySQL - it contains a
               character outside ``[A-Za-z0-9_]`` or is a reserved word - is backtick
               quoted where it appears unquoted, through one longest-first,
               case-insensitive alternation, with guards so that function calls
               (``name(``) and ``... BY`` clauses are left alone.

  pass 2 (rebind) - identifiers MySQL itself has rejected (``1054 ... Unknown column
               'x'``) are matched back against the same vocabulary by a
               punctuation-insensitive comparison (exact, then prefix), and the bare token
               is replaced by the resolved true name in backticks.

The vocabulary (the "true names" of the episode) is collected from the two public sources
available at runtime: (a) the task blurb that hands over a table name and a comma
separated header list, and (b) any schema-listing result the solver has already
retrieved (``SHOW COLUMNS`` / ``DESCRIBE`` / ``SHOW TABLES`` / ``SHOW CREATE TABLE``),
paired with the tool message that answers it.  Every mined name is textually normalised: a
line break that a tabular cell serialised as a literal backslash + ``n`` becomes a space
and whitespace runs collapse, so a header that reaches the prompt as ``Header\\n(suffix)``
denotes the single identifier ``Header (suffix)``.  Only *ASCII* whitespace is normalised:
a declared name that contains a non-breaking space or another non-ASCII space keeps it
byte-exact, so the spelling this harness binds to is the spelling MySQL stored.

Everything is schema-driven, per-episode and task-agnostic: no task identifiers, no
answer tables, no cross-episode state.  Any exception inside the harness returns the
input object unchanged.

Contract: plain ``AgentHarness``, synchronous hooks, ordinary OpenAI-style dicts in and
out, per-episode instance state only (public metadata observed from the request /
history), imports limited to ``re``/``json``.
"""

import json
import re

# ---------------------------------------------------------------------------
# declared-schema extraction (task blurb)
# ---------------------------------------------------------------------------

_TABLE_INFO_RE = re.compile(
    r"The name of this table is (.+?), and the headers of this table are "
    r"(.*?)(?:\n\s*Question:|\Z)",
    re.I | re.S,
)

# A tabular cell holding a line break is serialised as a literal backslash + 'n' in the
# public blurb text; that is a transport artefact, not part of the identifier.
_LITERAL_NEWLINE_RE = re.compile(r"\\n")
# Only ASCII whitespace is collapsed: re's ``\s`` and str.strip() also match U+00A0 and
# other non-ASCII spaces, which would silently rewrite a real non-breaking space inside a
# declared name into an ASCII space and make the bound spelling wrong.
_ASCII_WS_RUN_RE = re.compile(r"[ \t\r\n\x0b\x0c]+")


def _normalize_name(name):
    """Repair the transport spelling of a mined identifier and strip it."""
    if not isinstance(name, str):
        return ""
    name = _LITERAL_NEWLINE_RE.sub(" ", name)
    name = _ASCII_WS_RUN_RE.sub(" ", name)
    return name.strip(" \t\r\n\x0b\x0c")


# ---------------------------------------------------------------------------
# which true names actually need quoting
# ---------------------------------------------------------------------------

# Modest MySQL-8.0 reserved-word set: only words that could realistically show up as
# an identifier in a tabular dataset (plus the analytics-function keywords).  Words
# such as date / time / year / name / status / score / position / record / points /
# class / total / value / type are deliberately absent - they are not reserved, and
# one of them shows up as a *column data type* in a CREATE TABLE in the corpus.
_RESERVED_WORDS = frozenset(
    """
    rank rows row range groups grouping over window partition
    order group key keys index usage system match interval
    lead lag first_value last_value nth_value row_number
    dense_rank percent_rank cume_dist ntile
    recursive lateral cube rollup of empty member array except intersect
    """.split()
)

_SIMPLE_NAME_RE = re.compile(r"[A-Za-z0-9_]+\Z")

# guards used inside the replacement callbacks
_OPEN_PAREN_RE = re.compile(r"\s*\(")              # identifier used as a function call
_BY_CLAUSE_RE = re.compile(r"[ \t\r\n]+BY\b", re.I)  # ORDER BY / GROUP BY / PARTITION BY

_QUOTE_CHARS = frozenset("'\"`")


def _needs_quoting(name):
    """True iff `name` must be backtick-quoted to be a usable MySQL identifier."""
    if not _SIMPLE_NAME_RE.match(name):
        return True
    return name.lower() in _RESERVED_WORDS


# ---------------------------------------------------------------------------
# observed-schema extraction (schema-listing tool results)
# ---------------------------------------------------------------------------

# Only the statement shapes that enumerate schema objects are considered.
_SCHEMA_LIST_RE = re.compile(
    r"SHOW\s+(?:FULL\s+)?COLUMNS|DESCRIBE|DESC\s|SHOW\s+TABLES|SHOW\s+CREATE\s+TABLE",
    re.I,
)

# element 1 of a listing row is the column type; accept only plausible type keywords.
_TYPE_RE = re.compile(
    r"^(?:tinyint|smallint|mediumint|int|integer|bigint|decimal|dec|numeric|fixed|"
    r"float|double|real|bit|bool|boolean|serial|char|varchar|binary|varbinary|"
    r"tinyblob|blob|mediumblob|longblob|tinytext|text|mediumtext|longtext|"
    r"enum|set|date|datetime|timestamp|time|year|json|geometry)(?![a-z])"
)

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", "\\": "\\", "'": "'", '"': '"'}


def _parse_tuple(text, index):
    """Parse one parenthesised python-repr tuple starting at `text[index]` == '(' .

    Returns ``(values, next_index)`` with values a list of str (quoted scalars) or None
    (bare scalars such as ``None``/numbers); ``(None, index)`` if it is not well formed.
    """
    values = []
    i = index + 1
    n = len(text)
    while True:
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n:
            return None, index
        if text[i] == ")":
            return values, i + 1
        if text[i] in "'\"":
            quote = text[i]
            i += 1
            buf = []
            while i < n and text[i] != quote:
                if text[i] == "\\" and i + 1 < n:
                    buf.append(_ESCAPES.get(text[i + 1], text[i + 1]))
                    i += 2
                    continue
                buf.append(text[i])
                i += 1
            if i >= n:
                return None, index
            i += 1
            values.append("".join(buf))
        else:
            j = i
            while j < n and text[j] not in ",)":
                j += 1
            values.append(None)
            i = j
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i < n and text[i] == ",":
            i += 1
            continue
        if i < n and text[i] == ")":
            return values, i + 1
        return None, index


def _parse_tuples(text):
    """Parse every top-level tuple of a python-repr result into a list of value lists."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "(":
            values, nxt = _parse_tuple(text, i)
            if values is None:
                return out
            out.append(values)
            i = nxt
            continue
        if ch in "'\"":
            quote = ch
            i += 1
            while i < n and text[i] != quote:
                i += 2 if (text[i] == "\\" and i + 1 < n) else 1
            i += 1
            continue
        i += 1
    return out


# ---------------------------------------------------------------------------
# solver-visible rejection messages
# ---------------------------------------------------------------------------

_UNKNOWN_COLUMN_RE = re.compile(r"Unknown column\s+['\"]([^'\"]+)['\"]", re.I)

_TOKEN_STRIP_RE = re.compile(r"[^a-z0-9]")


# ---------------------------------------------------------------------------
# SQL code-region splitter
# ---------------------------------------------------------------------------

def _split_regions(sql):
    """Split SQL into ``(is_code, text)`` regions.

    Code regions are everything outside single-quoted, double-quoted and backticked
    regions; backslash escapes (and doubled quote characters) inside those regions are
    honoured so that a literal never leaks into the code stream.
    """
    regions = []
    n = len(sql)
    i = 0
    start = 0
    while i < n:
        ch = sql[i]
        if ch in _QUOTE_CHARS:
            if i > start:
                regions.append((True, sql[start:i]))
            j = i + 1
            while j < n:
                cj = sql[j]
                if cj == "\\":
                    j += 2
                    continue
                if cj == ch:
                    if j + 1 < n and sql[j + 1] == ch:
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            regions.append((False, sql[i:j]))
            start = j
            i = j
            continue
        i += 1
    if start < n:
        regions.append((True, sql[start:]))
    return regions


def _normalize_token(name):
    """Punctuation/case-insensitive fingerprint of an identifier."""
    if not isinstance(name, str):
        return ""
    return _TOKEN_STRIP_RE.sub("", name.lower())


# characters treated as interchangeable separators inside a schema name
_IDENT_SEPARATORS = frozenset(
    " \t\r\n\x0b\x0c\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008"
    "\u2009\u200a\u200b\u2028\u2029\u202f\u205f\u3000"
    "_-().,/\\'\"`"
)
_LOOSE_MIN = 3
_BARE_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# words that must never be treated as a schema name when they appear BARE in SQL text
_SQL_KEYWORDS = frozenset(
    """
    select from where group by order having limit offset insert into values update
    set delete create table drop alter add and or not in like between is null as
    distinct count sum avg min max join left right inner outer on using union all
    asc desc case when then else end exists replace ignore explain describe show
    with of for to distinct
    """.split()
)


def _loose(name):
    """Separator-insensitive fingerprint of an identifier.

    Case-folded with the interchangeable separators (whitespace incl. U+00A0, underscore,
    hyphen, dot, brackets, comma, slash, quote characters) removed.  Symbols such as
    ``#`` and ``%`` are deliberately KEPT: they are part of the name, so a bare word
    must never be folded onto a declared name like ``Sr. #`` or ``Win %``.
    """
    return "".join(ch for ch in name.lower() if ch not in _IDENT_SEPARATORS)


# ---------------------------------------------------------------------------
# pass S: bind decorated (never-legal-bare) identifiers in table-name slots
# ---------------------------------------------------------------------------

# A table-name slot: one of the table-introducing keywords followed by whitespace and
# one unquoted operand.  The operand character class includes '.' and '-' so that a
# decorated spelling is consumed whole (matching only its first hyphen-separated piece
# would leave a dangling suffix behind).
_SLOT_RE = re.compile(
    r"(?i)\b(?:FROM|INTO|UPDATE|JOIN|TABLE|DESCRIBE|DESC|USING)(\s+)([`A-Za-z0-9_$][A-Za-z0-9_$.\-]*)"
)

_SLOT_MIN = 3   # ignore fingerprints too short to be evidence of a schema name


# ---------------------------------------------------------------------------
# pass 0: rescue identifiers that were (mis)written in double quotes
# ---------------------------------------------------------------------------

# Pass 0 only fires when the double-quoted span is being introduced as a boolean-predicate
# operand, i.e. the text just before it ends with one of these keywords.  Elsewhere a
# double-quoted span may be a genuine string literal (right of a comparison) or a
# non-predicate identifier slot (SET target, CASE arm, column list) and is left alone.
_PREDICATE_TAIL_RE = re.compile(r"(?:\bWHERE|\bAND|\bOR|\bNOT|\bON|\bHAVING)\s*\Z", re.I)


class AgentHarness:
    """Identity runtime + schema-identifier canonicalization for `execute_sql`."""

    def __init__(self, public_context=None):
        self.public_context = public_context
        # per-episode state: public metadata only
        self._names = {}         # lower-cased name -> first spelling seen (columns + tables)
        self._tables = set()     # lower-cased observed/declared *table* names
        self._needed = {}        # lower-cased name -> spelling, must be backtick-quoted
        self._rejected = {}      # lower-cased MySQL-rejected token -> spelling as written
        self._seen_text = set()
        self._seen_pairs = set()
        self._seen_rejections = set()
        self._version = 0
        self._pattern = None
        self._pattern_version = -1
        self._loose_index = None
        self._loose_version = -1

    # ------------------------------------------------------------------ hooks

    def transform_request(self, request):
        # the schema blurb and earlier schema listings may be visible in the request
        try:
            self._absorb(request)
        except Exception:
            pass
        return request

    def transform_response(self, response, public_history):
        try:
            self._absorb(public_history)
            if not isinstance(response, dict):
                return response
            calls = response.get("tool_calls")
            if not isinstance(calls, list) or not calls:
                return response
            if not self._needed and not self._rejected:
                return response
            for call in calls:
                if not isinstance(call, dict):
                    continue
                func = call.get("function")
                if not isinstance(func, dict) or func.get("name") != "execute_sql":
                    continue
                try:
                    self._repair_call(func)
                except Exception:
                    continue
            return response
        except Exception:
            return response

    # ------------------------------------------------------- step 1: extraction

    def _absorb(self, obj):
        """Walk an arbitrary nested public structure and mine schema facts."""
        if isinstance(obj, str):
            self._absorb_text(obj)
        elif isinstance(obj, dict):
            for value in obj.values():
                self._absorb(value)
        elif isinstance(obj, (list, tuple)):
            if obj and all(
                isinstance(item, dict) and isinstance(item.get("role"), str) for item in obj
            ):
                self._absorb_messages(list(obj))
            else:
                for item in obj:
                    self._absorb(item)

    def _absorb_messages(self, messages):
        """Mine the blurb from messages and pair schema-listing calls with their results."""
        pending = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if isinstance(content, str):
                if role == "tool":
                    self._mine_rejection(content)
                else:
                    self._absorb_text(content)
            elif content is not None:
                self._absorb(content)
            if role == "assistant":
                for call in self._tool_calls(message):
                    query = self._call_query(call)
                    if query and _SCHEMA_LIST_RE.search(query):
                        pending.append(call.get("id"))
            elif role == "tool" and pending and isinstance(content, str):
                call_id = message.get("tool_call_id")
                if call_id in pending:
                    pending.remove(call_id)
                elif call_id is None and len(pending) == 1:
                    pending.pop()
                else:
                    continue
                self._mine_schema_listing(content)

    def _absorb_text(self, text):
        if not text or text in self._seen_text:
            return
        self._seen_text.add(text)
        if "name of this table is" not in text.lower():
            return
        for match in _TABLE_INFO_RE.finditer(text):
            self._remember(match.group(1), is_table=True)
            headers = match.group(2).split(",")
            for index, header in enumerate(headers):
                self._remember(header)
                # the sentence-terminating period is captured into the last header,
                # so also accept that header without its final period.
                if index == len(headers) - 1:
                    bare = header.strip()
                    if bare.endswith(".") and len(bare) > 1:
                        self._remember(bare[:-1])

    def _mine_schema_listing(self, text):
        """Collect observed column / table names from one schema-listing result."""
        if not text or "(" not in text or text in self._seen_pairs:
            return
        if len(self._seen_pairs) > 512:
            self._seen_pairs.clear()
        self._seen_pairs.add(text)
        for values in _parse_tuples(text):
            if len(values) == 1 and isinstance(values[0], str):
                self._remember(values[0], is_table=True)
                continue
            if len(values) < 2:
                continue
            name, type_name = values[0], values[1]
            if not isinstance(name, str) or not isinstance(type_name, str):
                continue
            if _TYPE_RE.match(type_name.strip().lower()):
                self._remember(name)

    def _mine_rejection(self, text):
        """Collect identifiers MySQL has already rejected in a public tool result."""
        if "unknown column" not in text.lower() or text in self._seen_rejections:
            return
        if len(self._seen_rejections) > 512:
            self._seen_rejections.clear()
        self._seen_rejections.add(text)
        for match in _UNKNOWN_COLUMN_RE.finditer(text):
            token = match.group(1).strip()
            if token:
                self._rejected.setdefault(token.lower(), token)

    def _remember(self, name, is_table=False):
        name = _normalize_name(name)
        if not name:
            return
        key = name.lower()
        if is_table:
            self._tables.add(key)
        if key not in self._names:
            self._names[key] = name
            self._version += 1
            if _needs_quoting(name):
                self._needed[key] = name
                self._pattern = None

    # ------------------------------------------------- step 2: pass 1 quoting

    def _compiled(self):
        if self._pattern is not None and self._pattern_version == self._version:
            return self._pattern
        if not self._needed:
            self._pattern = None
            self._pattern_version = self._version
            return None
        names = sorted(self._needed, key=lambda s: (-len(s), s))
        body = "|".join(re.escape(name) for name in names)
        self._pattern = re.compile(r"(?<![A-Za-z0-9_`])(" + body + r")(?![A-Za-z0-9_])", re.I)
        self._pattern_version = self._version
        return self._pattern

    def _sub(self, match):
        tail = match.string[match.end():]
        if _BY_CLAUSE_RE.match(tail):
            return match.group(0)
        # A name followed by "(" is a function call -- unless it is the declared table
        # name, which is exactly the `INSERT INTO <table> (...)` / `CREATE TABLE <table>
        # (...)` shape and must still be quoted.
        if _OPEN_PAREN_RE.match(tail) and match.group(0).lower() not in self._tables:
            return match.group(0)
        declared = self._needed.get(match.group(0).lower())
        if declared is None:
            return match.group(0)
        return "`" + declared + "`"

    # ------------------------------------ step 2b: pass 0 double-quote rescue

    def _skip_quoted(self, sql, i, ch):
        """Index just past the quoted region that starts at ``sql[i]`` == ``ch``."""
        n = len(sql)
        j = i + 1
        while j < n:
            cj = sql[j]
            if cj == "\\":
                j += 2
                continue
            if cj == ch:
                if j + 1 < n and sql[j + 1] == ch:
                    j += 2
                    continue
                return j + 1
            j += 1
        return n

    def _rescue_double_quoted(self, sql):
        """Backtick a double-quoted span used as a predicate operand whose text is a true
        schema name.

        Under MySQL's default sql_mode a double-quoted token is a string literal, so a
        solver that writes a column name in double quotes inside a WHERE / ON predicate
        gets a silently always-false comparison.  When the text inside the span is exactly
        one of the episode's true names and that name cannot stand bare, the span is
        rewritten to the true name in backticks.  Spans not introduced by a predicate
        keyword are left byte-identical.
        """
        if '"' not in sql or not self._names:
            return sql
        n = len(sql)
        out = []
        start = 0
        i = 0
        changed = False
        while i < n:
            ch = sql[i]
            if ch in "'`":
                i = self._skip_quoted(sql, i, ch)
                continue
            if ch == '"':
                end = self._skip_quoted(sql, i, '"')
                if end <= i + 1 or sql[end - 1] != '"':
                    break
                inner = sql[i + 1:end - 1]
                name = self._names.get(_normalize_name(inner).lower())
                if (
                    name is not None
                    and _needs_quoting(name)
                    and _PREDICATE_TAIL_RE.search(sql[start:i])
                ):
                    out.append(sql[start:i])
                    out.append("`" + name + "`")
                    start = end
                    changed = True
                i = end
                continue
            i += 1
        if not changed:
            return sql
        out.append(sql[start:])
        return "".join(out)

    # --------------------------------------------- step 3: pass 2 rebinding

    def _resolve(self, token):
        """Resolve a rejected token to exactly one true name, or None."""
        target = _normalize_token(token)
        if not target:
            return None
        exact = [name for name in self._names.values() if _normalize_token(name) == target]
        if len(exact) == 1:
            return exact[0]
        if len(target) >= 3:
            near = {}
            for name in self._names.values():
                folded = _normalize_token(name)
                if len(folded) < 3:
                    continue
                if folded.startswith(target) or target.startswith(folded):
                    near[name.lower()] = name
            if len(near) == 1:
                return next(iter(near.values()))
        return None

    def _rebind(self):
        """Build the pass-2 alternation: lower-cased rejected token -> true spelling."""
        mapping = {}
        for key, written in self._rejected.items():
            resolved = self._resolve(written)
            if resolved is None or resolved.lower() == key:
                continue
            mapping[key] = resolved
        if not mapping:
            return None
        names = sorted(mapping, key=lambda s: (-len(s), s))
        body = "|".join(re.escape(name) for name in names)
        pattern = re.compile(r"(?<![A-Za-z0-9_`])(" + body + r")(?![A-Za-z0-9_])", re.I)
        return pattern, mapping

    def _sub_rebind(self, match, mapping):
        # a token that starts a parenthesised identifier must not be spliced
        if _OPEN_PAREN_RE.match(match.string[match.end():]):
            return match.group(0)
        token = match.group(0)
        if token.lower() in self._names:
            # already the true name: quoting it is pass 1's job
            return token
        resolved = mapping.get(token.lower())
        if resolved is None:
            return token
        return "`" + resolved + "`"

    # ------------------------------------- step 2c: pass A separator-insensitive binding

    def _loose_map(self):
        """Cached map: separator-insensitive form -> the true spellings that share it."""
        if self._loose_index is not None and self._loose_version == self._version:
            return self._loose_index
        index = {}
        for name in self._names.values():
            key = _loose(name)
            if len(key) < _LOOSE_MIN:
                continue
            index.setdefault(key, set()).add(name)
        self._loose_index = index
        self._loose_version = self._version
        return index

    def _bind_name(self, index, text, bare):
        """Resolve `text` to exactly one true spelling, or None when it must be left alone."""
        if bare and text.lower() in _SQL_KEYWORDS:
            return None
        key = _loose(text)
        if len(key) < _LOOSE_MIN:
            return None
        candidates = index.get(key)
        if not candidates or len(candidates) != 1:
            return None
        true = next(iter(candidates))
        if true == text:
            return None
        if true.lower() == text.lower() and not _needs_quoting(true):
            return None
        return true

    def _bind_bare(self, chunk, index):
        def repl(match):
            token = match.group(0)
            true = self._bind_name(index, token, bare=True)
            if true is None:
                return token
            tail = chunk[match.end():]
            if tail[:1] == ".":
                return token
            if _OPEN_PAREN_RE.match(tail) and true.lower() not in self._tables:
                return token
            if _BY_CLAUSE_RE.match(tail):
                return token
            return "`" + true + "`" if _needs_quoting(true) else true

        return _BARE_IDENT_RE.sub(repl, chunk)

    def _bind_identifiers(self, sql):
        """Bind bare identifier tokens and backticked spans to the episode's true names."""
        if not self._names:
            return sql
        index = self._loose_map()
        if not index:
            return sql
        parts = []
        changed = False
        for is_code, chunk in _split_regions(sql):
            if is_code:
                new = self._bind_bare(chunk, index)
            elif len(chunk) > 2 and chunk[0] == "`" and chunk[-1] == "`":
                inner = chunk[1:-1]
                true = self._bind_name(index, inner, bare=False)
                new = "`" + true + "`" if true is not None else chunk
            else:
                new = chunk
            if new != chunk:
                changed = True
            parts.append(new)
        return "".join(parts) if changed else sql

    # ---------------------------------- step 2d: pass S decorated table-slot binding

    def _bind_table_slots(self, sql):
        """Bind a decorated table-slot token to the episode's own true table name.

        A bare token that contains `-` cannot be a legal MySQL identifier (the grammar
        reads it as a minus operator), so it makes the statement a guaranteed syntax error.
        Resolve it against the episode's true names (exact separator-insensitive
        fingerprint, else exactly one true name whose fingerprint the token fingerprint
        prefixes) and replace the token with that spelling in backticks.  Ambiguous or
        unknown tokens, already backticked operands, and tokens that equal the resolved
        name stay byte-identical.
        """
        if not self._names:
            return sql
        index = self._loose_map()
        if not index:
            return sql

        def repl(match):
            head = match.group(0)[: -len(match.group(2))]
            token = match.group(2)
            if token.startswith("`") or "-" not in token:
                return match.group(0)
            key = _loose(token)
            if len(key) < _SLOT_MIN:
                return match.group(0)
            candidates = index.get(key)
            if candidates and len(candidates) == 1:
                true = next(iter(candidates))
            else:
                prefixes = [n for k, names in index.items() if k.startswith(key) for n in names]
                if len(prefixes) != 1:
                    return match.group(0)
                true = prefixes[0]
            if true == token:
                return match.group(0)
            return head + "`" + true + "`" if _needs_quoting(true) else head + true

        return self._apply(sql, lambda chunk: _SLOT_RE.sub(repl, chunk))

    # -------------------------------------------------------- rewrite plumbing

    def _apply(self, sql, replace):
        parts = []
        changed = False
        for is_code, chunk in _split_regions(sql):
            if is_code:
                new = replace(chunk)
                if new != chunk:
                    changed = True
                parts.append(new)
            else:
                parts.append(chunk)
        return "".join(parts) if changed else sql

    def _rewrite(self, sql):
        try:
            out = self._bind_table_slots(sql)
            out = self._bind_identifiers(out)
            out = self._rescue_double_quoted(out)
            pattern = self._compiled()
            if pattern is not None:
                out = self._apply(out, lambda chunk: pattern.sub(self._sub, chunk))
            rebind = self._rebind()
            if rebind is not None:
                rebind_pattern, mapping = rebind
                out = self._apply(
                    out,
                    lambda chunk: rebind_pattern.sub(
                        lambda match: self._sub_rebind(match, mapping), chunk
                    ),
                )
            return out
        except Exception:
            return sql

    # ------------------------------------------------- step 4: tool call repair

    def _tool_calls(self, message):
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            return [call for call in calls if isinstance(call, dict)]
        return []

    def _call_query(self, call):
        """Peek at the `query` argument of an execute_sql call without mutating it."""
        func = call.get("function")
        if not isinstance(func, dict) or func.get("name") != "execute_sql":
            return None
        args = func.get("arguments")
        try:
            payload = json.loads(args) if isinstance(args, str) else args
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        query = payload.get("query")
        return query if isinstance(query, str) else None

    def _repair_call(self, func):
        args = func.get("arguments")
        as_text = isinstance(args, str)
        if as_text:
            payload = json.loads(args)
        elif isinstance(args, dict):
            payload = args
        else:
            return
        if not isinstance(payload, dict):
            return
        query = payload.get("query")
        if not isinstance(query, str) or not query:
            return
        rewritten = self._rewrite(query)
        if rewritten == query:
            return
        new_payload = dict(payload)
        new_payload["query"] = rewritten
        func["arguments"] = (
            json.dumps(new_payload, separators=(",", ":")) if as_text else new_payload
        )
