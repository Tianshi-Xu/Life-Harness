"""Meta-Harness candidate: materialize a text-expressed tool call, then canonicalize
`execute_sql` identifiers to the episode's own schema.

Two cooperating mechanisms, both synchronous, both exception-safe.

  A. materialization - the host executes ``response["tool_calls"]``.  Some solver turns
     come back with an EMPTY ``tool_calls`` list and express the intended call as prose
     instead (an XML-ish ``<tool_call>`` blob, or a fenced SQL code block).  The host
     rejects such a turn and the model burns a turn re-asking (in the worst case it loops
     to the episode cap).  This mechanism reads the textual intent and turns it into a
     real tool call so the turn is not wasted:
       * XML path: inside the ``<tool_call> ... </tool_call>`` span, first try
         ``json.loads``.  If the span is not well-formed JSON - the common case, because a
         model writing an SQL literal that itself contains a quote mangles the escaping -
         fall back to a *lenient* extractor that does not depend on well-formed JSON:
         the tool name comes from ``"name": "..."`` and the statement from
         ``"query": "`` up to the LAST double quote of the remainder, which keeps an
         embedded quote that lives inside the value.  The consumed span is removed from
         the content, surrounding prose is kept.
       * fenced path: with no XML span, the first fenced code block whose body starts with
         a data statement keyword is executed as the query (content untouched).
     A recovered statement is handed to the canonicalizer below, because materialization
     happens BEFORE the repair loop.

  B. identifier canonicalization (three rewrite passes over SQL *code* regions only;
     single-quoted and backticked regions stay byte-identical, so genuine string literals
     are never touched):

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
denotes the single identifier ``Header (suffix)``.

Everything is schema-driven, per-episode and task-agnostic: no task identifiers, no
answer tables, no cross-episode state.  Any exception inside the harness returns the
input object unchanged; if no statement can be recovered the response is left
byte-identical (a fabricated call is never invented, and a ``commit_final_answer``
materialization is skipped unless an ``answers`` list can actually be parsed).

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
_WS_RUN_RE = re.compile(r"\s+")


def _normalize_name(name):
    """Repair the transport spelling of a mined identifier and strip it."""
    if not isinstance(name, str):
        return ""
    name = _LITERAL_NEWLINE_RE.sub(" ", name)
    name = _WS_RUN_RE.sub(" ", name)
    return name.strip()


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
# text-expressed tool-call patterns (mechanism A)
# ---------------------------------------------------------------------------

# <tool_call> ... </tool_call>; a missing closing tag runs to the end of the text.
_XML_CALL_RE = re.compile(r"<tool_call[^>]*>(.*?)(?:</tool_call>|\Z)", re.I | re.S)

# the first fenced code block, with an optional info string and optional newline
_FENCE_RE = re.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\r?\n?(.*?)```", re.S)

# lenient, well-formedness-independent probes inside an XML span
_XML_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')
_XML_QUERY_RE = re.compile(r'"query"\s*:\s*"')
_XML_ANSWERS_RE = re.compile(r'"answers"\s*:\s*')

# first word of a fenced body that makes it read as a statement to execute
# (compared upper-cased against the upper-cased first word of the body)
_SQL_LEAD_WORDS = frozenset(
    """SELECT INSERT UPDATE DELETE REPLACE WITH SHOW DESCRIBE EXPLAIN
    ALTER CREATE DROP TRUNCATE""".split()
)

_TOOL_NAMES = frozenset(("execute_sql", "commit_final_answer"))


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


# ---------------------------------------------------------------------------
# pass 0: rescue identifiers that were (mis)written in double quotes
# ---------------------------------------------------------------------------

# Pass 0 only fires when the double-quoted span is being introduced as a boolean-predicate
# operand, i.e. the text just before it ends with one of these keywords.  Elsewhere a
# double-quoted span may be a genuine string literal (right of a comparison) or a
# non-predicate identifier slot (SET target, CASE arm, column list) and is left alone.
_PREDICATE_TAIL_RE = re.compile(r"(?:\bWHERE|\bAND|\bOR|\bNOT|\bON|\bHAVING)\s*\Z", re.I)


class AgentHarness:
    """Identity runtime + text-tool-call materialization + schema identifier repair."""

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
        self._materialized = 0   # per-episode counter, keeps materialized ids unique

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
            try:
                self._absorb(public_history)
            except Exception:
                pass
            if not isinstance(response, dict):
                return response
            # mechanism A runs first, so a recovered statement is subsequently
            # canonicalized by the identifier passes below
            self._materialize(response)
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

    # ------------------------------------------- mechanism A: materialization

    def _materialize(self, response):
        """Turn a text-expressed tool call into a real one, in place.

        Does nothing when the reply is not a dict, already carries tool calls, has no
        non-empty string content, or yields no recoverable statement.  Any exception
        leaves the response unchanged.
        """
        try:
            if not isinstance(response, dict):
                return
            if response.get("tool_calls"):
                return
            content = response.get("content")
            if not isinstance(content, str) or not content:
                return
            span = _XML_CALL_RE.search(content)
            if span is not None:
                name, sql, answers = self._parse_xml_payload(span.group(1).strip())
                if name is None:
                    return
                if name == "commit_final_answer":
                    call = self._make_call(name, {"answers": answers})
                else:
                    call = self._make_call(name, {"query": sql})
                if call is None:
                    return
                response["tool_calls"] = [call]
                response["content"] = (
                    content[:span.start()] + content[span.end():]
                ).strip()
                return
            # no XML span: maybe a fenced SQL block
            fence = _FENCE_RE.search(content)
            if fence is None:
                return
            body = fence.group(2).strip()
            if not body:
                return
            if body.split(None, 1)[0].upper() not in _SQL_LEAD_WORDS:
                return
            call = self._make_call("execute_sql", {"query": body})
            if call is None:
                return
            response["tool_calls"] = [call]
            return
        except Exception:
            return

    def _parse_xml_payload(self, inner):
        """Recover ``(name, sql, answers)`` from the inside of a <tool_call> span.

        ``name`` is None when nothing usable could be lifted out.  The structured JSON
        read is preferred; a lenient, well-formedness-independent extractor backs it up,
        because a statement value containing a quote routinely breaks the escaping.
        """
        name = None
        sql = None
        answers = None
        try:
            parsed = json.loads(inner)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            raw_name = parsed.get("name")
            if isinstance(raw_name, str) and raw_name:
                name = raw_name
            else:
                name = "execute_sql"
            args = parsed.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = None
            if isinstance(args, dict):
                query = args.get("query")
                if isinstance(query, str) and query:
                    sql = query
                recovered = args.get("answers")
                if isinstance(recovered, list):
                    answers = recovered
        usable = name is not None and (
            (name == "execute_sql" and sql) or (name == "commit_final_answer" and answers)
        )
        if not usable:
            match = _XML_NAME_RE.search(inner)
            if match is not None and match.group(1):
                name = match.group(1)
            elif name is None:
                name = "execute_sql"
            if name == "execute_sql" and not sql:
                sql = self._lenient_sql(inner)
            if name == "commit_final_answer" and answers is None:
                answers = self._lenient_answers(inner)
        if name not in _TOOL_NAMES:
            return None, None, None
        return name, sql, answers

    @staticmethod
    def _lenient_sql(inner):
        """Lift a ``"query": "..."`` value without requiring well-formed JSON.

        The value is taken from just after the ``"query": "`` marker up to the LAST
        double quote of the remainder, so an embedded quote inside the value survives
        (the classic case: an SQL string literal that itself contains a double quote).
        """
        match = _XML_QUERY_RE.search(inner)
        if match is None:
            return None
        rest = inner[match.end():]
        end = rest.rfind('"')
        if end <= 0:
            return None
        sql = rest[:end]
        return sql or None

    @staticmethod
    def _lenient_answers(inner):
        """Lift an ``"answers": [...]`` list with a streaming JSON read, else None."""
        match = _XML_ANSWERS_RE.search(inner)
        if match is None:
            return None
        try:
            value, _ = json.JSONDecoder().raw_decode(inner[match.end():])
        except Exception:
            return None
        return value if isinstance(value, list) else None

    def _make_call(self, name, payload):
        """Build one OpenAI-style tool call, or None if the payload is unwearable."""
        if name not in _TOOL_NAMES or not isinstance(payload, dict):
            return None
        if name == "execute_sql":
            query = payload.get("query")
            if not isinstance(query, str) or not query.strip():
                return None
        else:
            if not isinstance(payload.get("answers"), list):
                return None
        self._materialized += 1
        return {
            "id": "chatcmpl-tool-materialized-%d" % self._materialized,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(payload, separators=(",", ":")),
            },
        }

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
            out = self._rescue_double_quoted(sql)
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
