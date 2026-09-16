"""Runtime that recovers tool calls the model wrote as reply text.

The benchmark executes only function-call style tool calls. The model
occasionally emits the intended invocation as plain reply text instead: an
XML ``<tool_call>{...}</tool_call>`` block whose embedded JSON is malformed
(so the harness rejects it with "your <tool_call> XML was malformed or
truncated" and nothing runs), or a single fenced SQL code block (which the
harness never executes and answers with "No executable tool calls found").
Both forms strand the episode: one burns the whole turn budget without ever
executing a statement, the other wastes a recovery turn. This runtime
promotes exactly one unambiguous text-form invocation into a real
``execute_sql`` tool call, then pushes the recovered statement through the
declared-identifier normalization inherited from the incumbent.

Inherited behavior (rounds 1-2): declared identifiers are backtick-quoted
where they appear bare, double-quoted declared identifiers are canonicalized
to the backticked declared surface, and escape-embedded declared tokens are
matched with whitespace tolerance. Identifiers are parsed from the public
user message. Requests pass through unchanged. Every recovery is guarded:
only when the response carries no tool call at all, only for a single
unambiguous invocation, only for ``execute_sql``, never for a commit, and any
parse failure or exception degrades to identity.
"""

import copy
import json
import re

_DECLARATION_RE = re.compile(
    r"name\s+of\s+this\s+table\s+is\s+(?P<table>.+?)\s*,?\s*and\s+the\s+headers\s+of\s+this\s+table\s+are\s+"
    r"(?P<headers>.+?)(?:\.\s*\n|\n|\Z)",
    re.IGNORECASE | re.DOTALL,
)

_PURE_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\Z")
_SENTENCE_PERIOD_RE = re.compile(r"\.\Z")
_TRAILING_PUNCTUATION_RE = re.compile(r"[^A-Za-z0-9_]+\Z")
_IDENT_EDGE = r"A-Za-z0-9_$"

_QUOTE_CHARS = frozenset(("'", '"', "`"))

# A declared token embedding a literal backslash escape, e.g. a whitespace
# position rendered as a two-character escape sequence instead of a real space.
_ESCAPE_RE = re.compile(r"\\[A-Za-z]")
_WHITESPACE_RUN_RE = re.compile(r"\s+")

# Reserved words whose bare use is rejected by the engine.
_RESERVED_WORDS = frozenset(
    (
        "rank", "order", "group", "key", "index", "usage", "system", "signal",
        "interval", "condition", "current_date", "current_time", "function",
        "references", "partition", "range", "rows", "over", "window",
        "recursive", "cube", "lag", "lead",
    )
)

# --- text-form tool call recovery -----------------------------------------

_TOOL_NAME = "execute_sql"

_XML_BLOCK_RE = re.compile(r"\s*<tool_call>\s*(?P<body>\{.*\})\s*</tool_call>\s*", re.DOTALL)
_JSON_NAME_RE = re.compile(r'"name"\s*:\s*"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"')
_JSON_QUERY_KEY_RE = re.compile(r'"query"\s*:\s*"')
_SQL_FENCE_RE = re.compile(r"```[ \t]*(?P<lang>[A-Za-z]*)[ \t]*\r?\n(?P<body>.*?)```", re.DOTALL)

_SQL_KEYWORDS = (
    "SELECT", "INSERT", "UPDATE", "DELETE", "REPLACE", "WITH", "DESCRIBE",
    "SHOW", "ALTER", "CREATE", "DROP", "EXPLAIN",
)
_SQL_START_RE = re.compile(
    r"^(?:%s)\b" % "|".join(_SQL_KEYWORDS), re.IGNORECASE
)


def _starts_with_sql(text):
    return _SQL_START_RE.match(text.lstrip()) is not None


def _single_statement(sql):
    """True when `sql` contains at most one statement (one trailing `;`)."""
    length = len(sql)
    index = 0
    semicolons = 0
    while index < length:
        char = sql[index]
        if char in _QUOTE_CHARS:
            quote = char
            index += 1
            while index < length:
                if sql[index] == "\\":
                    index += 2
                    continue
                if sql[index] == quote:
                    if index + 1 < length and sql[index + 1] == quote:
                        index += 2
                        continue
                    break
                index += 1
            index += 1
            continue
        if char == ";":
            if sql[index + 1:].strip():
                return False
            semicolons += 1
        index += 1
    return semicolons <= 1


def _xml_form_query(content):
    """SQL from a whole-content ``<tool_call>`` block the harness could not parse.

    A strictly parseable block is left alone: the harness handles it itself,
    and promoting it would risk executing the same statement twice.
    """
    found = _XML_BLOCK_RE.fullmatch(content)
    if found is None:
        return None
    body = found.group("body")
    if "<tool_call>" in body or "</tool_call>" in body:
        # The greedy body swallowed a second block: the reply is ambiguous, so
        # it is left untouched rather than mined across block boundaries.
        return None
    try:
        json.loads(body)
        return None
    except (ValueError, TypeError):
        pass
    name = _JSON_NAME_RE.search(body)
    if name is None or name.group("name") != _TOOL_NAME:
        return None
    query_key = _JSON_QUERY_KEY_RE.search(body)
    if query_key is None:
        return None
    # Only the documented shape {"name": ..., "arguments": {"query": ...}} is
    # accepted; anything else is ambiguous and is left untouched.
    between = "".join(body[name.end():query_key.start()].split())
    if between != ',"arguments":{':
        return None
    stripped = body.rstrip()
    if not stripped.endswith("}}"):
        return None
    tail = stripped[:-2].rstrip()
    if not tail.endswith('"'):
        return None
    # The query value runs from the opening quote consumed by the key pattern
    # to the last quote before the closing braces, preserving SQL-level escapes
    # and embedded quotes verbatim.
    start = query_key.end()
    sql = tail[start:-1]
    if not sql.strip() or not _starts_with_sql(sql) or not _single_statement(sql):
        return None
    return sql


def _fenced_form_query(content):
    """SQL from the single fenced code block in the reply, if unambiguous."""
    blocks = []
    for found in _SQL_FENCE_RE.finditer(content):
        if (found.group("lang") or "").strip().lower() not in ("", "sql", "mysql"):
            continue
        body = found.group("body").strip()
        if body and _starts_with_sql(body):
            blocks.append(body)
    if len(blocks) != 1:
        return None
    if not _single_statement(blocks[0]):
        return None
    return blocks[0]


def _recovered_query(content):
    if not isinstance(content, str) or not content.strip():
        return None
    sql = _xml_form_query(content)
    if sql is None:
        sql = _fenced_form_query(content)
    return sql


def _call_id(text):
    """Deterministic id for the synthesized call (FNV-1a over the statement)."""
    value = 0xCBF29CE484222325
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return "chatcmpl-tool-%016x" % value


# --- inherited declared-identifier normalization ---------------------------


def _message_texts(history):
    if isinstance(history, dict):
        history = history.get("messages")
    if not isinstance(history, (list, tuple)):
        return []
    texts = []
    for message in history:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        text = None
        if isinstance(content, str):
            text = content
        elif isinstance(content, (list, tuple)):
            chunks = []
            for part in content:
                if isinstance(part, str):
                    chunks.append(part)
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
            if chunks:
                text = "\n".join(chunks)
        if text:
            texts.append((message.get("role"), text))
    return texts


def _strip_wrapping(token):
    token = token.strip()
    while len(token) >= 2 and token[0] == token[-1] and token[0] in _QUOTE_CHARS:
        token = token[1:-1].strip()
    return token


def _declared_identifiers(history):
    user_hits = []
    other_hits = []
    for role, text in _message_texts(history):
        found = _DECLARATION_RE.search(text)
        if found is None:
            continue
        if role in (None, "user"):
            user_hits.append(found)
        else:
            other_hits.append(found)
    identifiers = []
    seen = set()
    for match in user_hits or other_hits:
        table = _strip_wrapping(match.group("table"))
        headers = _SENTENCE_PERIOD_RE.sub("", match.group("headers").rstrip())
        for candidate in [table] + headers.split(","):
            token = _strip_wrapping(candidate)
            if not token or _PURE_NUMBER_RE.match(token):
                continue
            if token not in seen:
                seen.add(token)
                identifiers.append(token)
    return identifiers


def _needs_quoting(identifier):
    if identifier.lower() in _RESERVED_WORDS:
        return True
    for char in identifier:
        if char == "_" or (char.isascii() and char.isalnum()):
            continue
        return True
    return False


def _identifier_pattern(identifier):
    return re.compile(
        r"(?<![{0}.]){1}(?![{0}])(?!\()".format(_IDENT_EDGE, re.escape(identifier))
    )


def _spaced_identifier_pattern(variant):
    """Whitespace-tolerant pattern for the normalized form of an escaped token."""
    body = r"\s+".join(re.escape(piece) for piece in variant.split())
    return re.compile(
        r"(?<![{0}.]){1}(?![{0}])(?!\()".format(_IDENT_EDGE, body)
    )


def _escaped_identifier_variant(identifier):
    """Readable single-space form of an identifier embedding backslash escapes.

    Returns None when the declared token has no escape, or when normalizing it
    yields an empty string or leaves it unchanged.
    """
    if _ESCAPE_RE.search(identifier) is None:
        return None
    variant = _WHITESPACE_RUN_RE.sub(" ", _ESCAPE_RE.sub(" ", identifier)).strip()
    if not variant or variant == identifier:
        return None
    return variant


def _build_targets(identifiers):
    declared = set(identifiers)
    targets = []
    for identifier in identifiers:
        if _needs_quoting(identifier):
            targets.append(
                (len(identifier), _identifier_pattern(identifier), "`" + identifier + "`")
            )
            # Declared identifiers may lose trailing punctuation in the model's
            # SQL (e.g. a header ending in a period written without it); map that
            # stripped form back.
            core = _TRAILING_PUNCTUATION_RE.sub("", identifier)
            if len(core) >= 2 and core != identifier and core not in declared:
                targets.append(
                    (len(core), _identifier_pattern(core), "`" + identifier + "`")
                )
        # Escaped tokens are additionally matched with whitespace tolerance, so
        # the readable single-space surface still resolves to the declared name.
        variant = _escaped_identifier_variant(identifier)
        if variant is not None:
            targets.append(
                (len(variant), _spaced_identifier_pattern(variant), "`" + variant + "`")
            )
    targets.sort(key=lambda item: item[0], reverse=True)
    return targets


def _span_end(sql, start):
    """Index of the quote closing the span opened at `start`, or None if open.

    Mirrors the protection scanner used elsewhere: a backslash skips the next
    character and a doubled quote character does not close the span.
    """
    quote = sql[start]
    length = len(sql)
    cursor = start + 1
    while cursor < length:
        current = sql[cursor]
        if current == "\\":
            cursor += 2
            continue
        if current == quote:
            if cursor + 1 < length and sql[cursor + 1] == quote:
                cursor += 2
                continue
            return cursor
        cursor += 1
    return None


def _identifier_context_blocked(sql, start, end):
    """True when a span sits in a qualified-name or call-like context.

    The same edges that keep bare-identifier quoting out of `a.b` and call
    positions keep a quoted span out of them as well.
    """
    if start > 0:
        before = sql[start - 1]
        if before == "." or (before.isascii() and (before.isalnum() or before in "_$")):
            return True
    if end + 1 < len(sql):
        after = sql[end + 1]
        if after == "(" or (after.isascii() and (after.isalnum() or after == "_")):
            return True
    return False


def _requote_double_quoted(sql, declared):
    """Backtick declared identifiers that the model wrapped in double quotes.

    A whole double-quoted span whose content (stripped) case-insensitively
    equals a declared identifier becomes that identifier backticked, so it
    stays an identifier reference instead of turning into a constant. Spans
    inside single-quoted literals or backticked regions are skipped, as are
    spans embedded in a larger token, and an unterminated quote run is left
    untouched.
    """
    if not declared:
        return sql, False
    canonical = {}
    for identifier in declared:
        canonical.setdefault(identifier.lower(), identifier)
    length = len(sql)
    pieces = []
    cursor = 0
    index = 0
    changed = False
    while index < length:
        char = sql[index]
        if char not in _QUOTE_CHARS:
            index += 1
            continue
        end = _span_end(sql, index)
        if end is None:
            break
        if char == '"' and not _identifier_context_blocked(sql, index, end):
            inner = sql[index + 1:end].strip()
            replacement = canonical.get(inner.lower())
            if replacement is not None:
                pieces.append(sql[cursor:index])
                pieces.append("`" + replacement + "`")
                cursor = end + 1
                changed = True
        index = end + 1
    if not changed:
        return sql, False
    pieces.append(sql[cursor:])
    return "".join(pieces), True


def _quoted_mask(sql):
    length = len(sql)
    mask = bytearray(length)
    index = 0
    while index < length:
        char = sql[index]
        if char not in _QUOTE_CHARS:
            index += 1
            continue
        quote = char
        cursor = index + 1
        while cursor < length:
            current = sql[cursor]
            if current == "\\":
                cursor += 2
                continue
            if current == quote:
                if cursor + 1 < length and sql[cursor + 1] == quote:
                    cursor += 2
                    continue
                break
            cursor += 1
        end = cursor if cursor < length else length - 1
        for position in range(index, end + 1):
            mask[position] = 1
        index = end + 1
    return mask


def _quote_identifiers(sql, targets):
    if not targets:
        return sql, False
    protected = _quoted_mask(sql)
    taken = []
    edits = []
    for _, pattern, replacement in targets:
        for match in pattern.finditer(sql):
            start, end = match.span()
            if any(protected[start:end]):
                continue
            if any(start < span_end and end > span_start for span_start, span_end in taken):
                continue
            taken.append((start, end))
            edits.append((start, end, replacement))
    if not edits:
        return sql, False
    edits.sort(key=lambda item: item[0], reverse=True)
    pieces = []
    cursor = len(sql)
    for start, end, replacement in edits:
        pieces.append(sql[end:cursor])
        pieces.append(replacement)
        cursor = start
    pieces.append(sql[:cursor])
    pieces.reverse()
    return "".join(pieces), True


def _normalize_arguments(arguments, declared, targets):
    as_text = False
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (ValueError, TypeError):
            return arguments, False
        as_text = True
    elif isinstance(arguments, dict):
        parsed = arguments
    else:
        return arguments, False
    if not isinstance(parsed, dict):
        return arguments, False
    query = parsed.get("query")
    if not isinstance(query, str):
        return arguments, False
    rewritten, double_quoted_changed = _requote_double_quoted(query, declared)
    rewritten, bare_changed = _quote_identifiers(rewritten, targets)
    if not (double_quoted_changed or bare_changed):
        return arguments, False
    updated = dict(parsed)
    updated["query"] = rewritten
    if as_text:
        return json.dumps(updated), True
    return updated, True


def register():
    return None


class AgentHarness:
    def __init__(self, public_context=None):
        self.public_context = public_context

    def transform_request(self, request):
        return request

    def transform_response(self, response, public_history):
        try:
            if not isinstance(response, dict):
                return response
            tool_calls = response.get("tool_calls")
            has_calls = isinstance(tool_calls, (list, tuple)) and bool(tool_calls)
            identifiers = _declared_identifiers(public_history)
            if has_calls:
                if not identifiers:
                    return response
                targets = _build_targets(identifiers)
                if not targets:
                    return response
                new_calls = copy.deepcopy(list(tool_calls))
                changed = False
                for call in new_calls:
                    if not isinstance(call, dict):
                        continue
                    function = call.get("function")
                    if not isinstance(function, dict):
                        continue
                    if function.get("name") == "commit_final_answer":
                        continue
                    arguments, call_changed = _normalize_arguments(
                        function.get("arguments"), identifiers, targets
                    )
                    if call_changed:
                        function["arguments"] = arguments
                        changed = True
                if not changed:
                    return response
                result = dict(response)
                result["tool_calls"] = new_calls
                return result
            # No function-call style invocation: the model may have written its
            # intended call as reply text instead. Promote at most one.
            sql = _recovered_query(response.get("content"))
            if sql is None:
                return response
            if identifiers:
                targets = _build_targets(identifiers)
                if targets:
                    normalized, _ = _normalize_arguments({"query": sql}, identifiers, targets)
                    if isinstance(normalized, dict):
                        sql = normalized.get("query", sql)
            call = {
                "id": _call_id(sql),
                "type": "function",
                "function": {"name": _TOOL_NAME, "arguments": json.dumps({"query": sql})},
            }
            result = dict(response)
            result["tool_calls"] = [call]
            return result
        except Exception:
            return response
