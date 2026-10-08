"""Meta-Harness candidate: canonicalize `execute_sql` identifiers to the episode's own schema.

One focused mechanism, two cooperating passes, both applied to SQL *code* regions only
(single-quoted, double-quoted and backticked regions stay byte-identical, so string
literals are never touched):

  vocabulary - the true names of the episode are collected from the two public sources
               available at runtime: (a) the task blurb that hands over a table name and
               a comma separated header list, and (b) any schema-listing result the solver
               has already retrieved (``SHOW COLUMNS`` / ``DESCRIBE`` / ``SHOW TABLES`` /
               ``SHOW CREATE TABLE``), paired with the tool message that answers it.  Every
               mined name is textually normalised: a line break that a tabular cell
               serialised as a literal backslash + ``n`` becomes a space and whitespace
               runs collapse, so a header that reaches the prompt as ``Header\\n(suffix)``
               denotes the single identifier ``Header (suffix)``.

  pass 1 (quote) - every true name that cannot stand bare in MySQL - it contains a
               character outside ``[A-Za-z0-9_]`` or is a reserved word - is backtick
               quoted where it appears unquoted, through one longest-first,
               case-insensitive alternation, with guards so that function calls
               (``name(``) and ``... BY`` clauses are left alone.

  pass 2 (rebind) - identifiers MySQL itself has rejected (``1054 ... Unknown column
               'x'``) are matched back against the same vocabulary by a
               punctuation-insensitive comparison (exact, then prefix), and the bare token
               is replaced by the resolved true name in backticks.  This catches the
               near-miss the quoting pass cannot see: a solver that writes a header with
               its punctuation dropped, or with only the leading word of a multi-word
               header, and then loops on the same rejected statement.

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
            out = sql
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
