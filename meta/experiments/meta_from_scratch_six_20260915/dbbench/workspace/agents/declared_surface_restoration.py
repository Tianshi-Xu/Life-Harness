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

The public message's surface forms are treated as authoritative for tokens
the model normalized. A declared identifier that the model re-spelled with
underscores or with a trailing parenthetical group dropped is rebound to the
declared name, and a value literal that lost the question's digit grouping or
percent sign is restored to the question's surface -- but only for statements
that write values, never for statements that define schema, and only when the
question gives a single unambiguous surface for that token.

Inherited behavior (rounds 1-3): declared identifiers are backtick-quoted
where they appear bare, double-quoted declared identifiers are canonicalized
to the backticked declared surface, escape-embedded declared tokens are
matched with whitespace tolerance, and text-form invocations are recovered
into real calls. Identifiers are parsed from the public user message.
Requests pass through unchanged. Every rewrite is guarded: only when the
response carries no tool call at all is a call synthesized, only for a single
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
    """Return (declared targets, normalized-variant targets).

    The first list is the incumbent's: bare declared identifiers that cannot be
    written bare are backticked. The second list holds the declared surface for
    tokens the model re-spelled, and is only ever applied to statements that do
    not define schema.
    """
    declared = set(identifiers)
    base_targets = []
    variant_targets = []
    seen_variants = set()
    for identifier in identifiers:
        if _needs_quoting(identifier):
            base_targets.append(
                (len(identifier), _identifier_pattern(identifier), "`" + identifier + "`")
            )
            # Declared identifiers may lose trailing punctuation in the model's
            # SQL (e.g. a header ending in a period written without it); map that
            # stripped form back.
            core = _TRAILING_PUNCTUATION_RE.sub("", identifier)
            if len(core) >= 2 and core != identifier and core not in declared:
                base_targets.append(
                    (len(core), _identifier_pattern(core), "`" + identifier + "`")
                )
        # Escaped tokens are additionally matched with whitespace tolerance, so
        # the readable single-space surface still resolves to the declared name.
        variant = _escaped_identifier_variant(identifier)
        if variant is not None:
            base_targets.append(
                (len(variant), _spaced_identifier_pattern(variant), "`" + variant + "`")
            )
        normalized, candidates = _surface_variants(identifier)
        others = [other for other in identifiers if other != identifier]
        for candidate in candidates:
            if not _variant_admissible(candidate, normalized, identifier, others):
                continue
            pattern = _identifier_pattern(candidate)
            replacement = "`" + normalized + "`"
            signature = (pattern.pattern, replacement)
            if signature in seen_variants:
                continue
            seen_variants.add(signature)
            variant_targets.append((len(candidate), pattern, replacement))
    base_targets.sort(key=lambda item: item[0], reverse=True)
    variant_targets.sort(key=lambda item: item[0], reverse=True)
    return base_targets, variant_targets


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


# --- public-surface restoration --------------------------------------------

_ASCII_ALNUM = frozenset(
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
)
_NON_KEY_CHAR_RE = re.compile(r"[^0-9a-z]")
_WORD_ONLY_RE = re.compile(r"[A-Za-z0-9]+\Z")
_PARENTHETICAL_TAIL_RE = re.compile(r"^(?P<base>.*?)\s*\([^()]*\)\s*$")

_LEADING_KEYWORD_RE = re.compile(r"\s*(?P<keyword>[A-Za-z_][A-Za-z0-9_]*)")
_WRITER_KEYWORDS = frozenset(("INSERT", "REPLACE", "UPDATE"))
_SCHEMA_KEYWORDS = frozenset(("CREATE", "ALTER", "DROP"))
_VALUES_RE = re.compile(r"\bVALUES\b", re.IGNORECASE)
_SET_RE = re.compile(r"\bSET\b", re.IGNORECASE)
_WHERE_RE = re.compile(r"\bWHERE\b", re.IGNORECASE)
_BARE_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")

_LITERAL_QUOTE_PAIRS = {
    "'": "'",
    '"': '"',
    "\u2018": "\u2019",
    "\u201c": "\u201d",
}
_NUMBER_SURFACE_RES = (
    re.compile(r"\$\d{1,3}(?:,\d{3})+(?:\.\d+)?%?"),
    re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?%?"),
    re.compile(r"\d+(?:\.\d+)?%"),
    re.compile(r"\$\d+(?:\.\d+)?"),
)


def _surface_key(text):
    """Matching key for a token: lowercase, alphanumeric characters only."""
    return _NON_KEY_CHAR_RE.sub("", text.lower())


def _normalized_surface(identifier):
    """Declared surface with literal escape sequences read as whitespace."""
    return _WHITESPACE_RUN_RE.sub(" ", _ESCAPE_RE.sub(" ", identifier)).strip()


def _surface_variants(identifier):
    """(normalized surface, re-spellings the model may have used)."""
    normalized = _normalized_surface(identifier)
    candidates = []
    parenthetical = _PARENTHETICAL_TAIL_RE.match(normalized)
    if parenthetical is not None:
        base = parenthetical.group("base").strip()
        if base:
            candidates.append(base)
    words = normalized.split()
    for size in range(2, len(words) + 1):
        if all(_WORD_ONLY_RE.match(word) for word in words[:size]):
            candidates.append("_".join(words[:size]))
    return normalized, candidates


def _variant_admissible(candidate, normalized, identifier, others):
    """A variant is usable only when no other declared name can claim it."""
    if len(candidate) < 6 or candidate == normalized or candidate == identifier:
        return False
    key = _surface_key(candidate)
    if not key:
        return False
    for other in others:
        other_key = _surface_key(other)
        if not other_key:
            continue
        if key == other_key or key.startswith(other_key) or other_key.startswith(key):
            return False
    return True


def _leading_keyword(sql):
    found = _LEADING_KEYWORD_RE.match(sql)
    return found.group("keyword").upper() if found is not None else None


def _add_surface(surfaces, surface):
    key = _surface_key(surface)
    if not key:
        return
    surfaces.setdefault(key, set()).add(surface)


def _quoted_surfaces(text):
    """Inner text of straight or curly single/double quoted spans."""
    surfaces = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char not in _LITERAL_QUOTE_PAIRS:
            index += 1
            continue
        close = text.find(_LITERAL_QUOTE_PAIRS[char], index + 1)
        if close < 0:
            index += 1
            continue
        inner = text[index + 1:close].strip()
        if 2 <= len(inner) <= 80:
            surfaces.append(inner)
        index = close + 1
    return surfaces


def _question_literals(history):
    """Surface forms the question used, keyed by their alphanumeric spelling.

    Only keys with exactly one distinct surface are kept, so a token the
    question writes in several forms is never rewritten.
    """
    surfaces = {}
    for role, text in _message_texts(history):
        if role not in (None, "user") or "Question:" not in text:
            continue
        for surface in _quoted_surfaces(text):
            _add_surface(surfaces, surface)
        for pattern in _NUMBER_SURFACE_RES:
            for found in pattern.finditer(text):
                _add_surface(surfaces, found.group(0))
    return {key: values for key, values in surfaces.items() if len(values) == 1}


def _is_word_char(char):
    return char.isalnum() or char == "_"


def _keyword_spans(sql, pattern, mask):
    spans = []
    for found in pattern.finditer(sql):
        start, end = found.span()
        if any(mask[start:end]):
            continue
        spans.append((start, end))
    return spans


def _value_region(sql, keyword):
    """Span past which a value-writing statement holds no column references."""
    mask = _quoted_mask(sql)
    if keyword in ("INSERT", "REPLACE"):
        hits = _keyword_spans(sql, _VALUES_RE, mask)
        if not hits:
            return None
        return hits[-1][1], len(sql)
    hits = _keyword_spans(sql, _SET_RE, mask)
    if not hits:
        return None
    start = hits[0][1]
    end = len(sql)
    for where_start, _ in _keyword_spans(sql, _WHERE_RE, mask):
        if where_start >= start:
            end = where_start
            break
    return start, end


def _bare_number_token(sql, index):
    if not sql[index].isdigit():
        return None
    if index > 0:
        before = sql[index - 1]
        if _is_word_char(before) or before in ".'\"`":
            return None
    found = _BARE_NUMBER_RE.match(sql, index)
    if found is None:
        return None
    if found.end() < len(sql):
        after = sql[found.end()]
        if _is_word_char(after) or after in ".'\"`":
            return None
    return found


def _restorable_literal(text, literals):
    """The question's surface for `text`, or None when no rewrite is safe."""
    key = _surface_key(text)
    if not key:
        return None
    surfaces = literals.get(key)
    if not surfaces or len(surfaces) != 1:
        return None
    value = next(iter(surfaces))
    if value == text or "\\" in text:
        return None
    for char in value:
        if char in _QUOTE_CHARS or char == "\\":
            return None
    for char in set(text) | set(value):
        if char in _ASCII_ALNUM:
            continue
        if text.count(char) > value.count(char):
            return None
    if value.strip(".,;: ") == text.strip(".,;: "):
        return None
    return value


def _restore_literals(sql, literals):
    """Rebind value literals to the surface form the question used."""
    if not literals:
        return sql, False
    keyword = _leading_keyword(sql)
    if keyword not in _WRITER_KEYWORDS:
        return sql, False
    region = _value_region(sql, keyword)
    if region is None:
        return sql, False
    start, end = region
    edits = []
    index = start
    while index < end:
        char = sql[index]
        if char in ("`", '"'):
            close = _span_end(sql, index)
            index = end if close is None else close + 1
            continue
        if char == "'":
            close = _span_end(sql, index)
            if close is None:
                break
            value = _restorable_literal(sql[index + 1:close], literals)
            if value is not None:
                edits.append((index, close + 1, "'" + value + "'"))
            index = close + 1
            continue
        if char.isdigit():
            found = _bare_number_token(sql, index)
            if found is None:
                index += 1
                continue
            index = found.end()
            value = _restorable_literal(found.group(0), literals)
            if value is not None:
                edits.append((found.start(), found.end(), "'" + value + "'"))
            continue
        index += 1
    if not edits:
        return sql, False
    pieces = []
    cursor = len(sql)
    for edit_start, edit_end, replacement in sorted(edits, key=lambda item: item[0], reverse=True):
        pieces.append(sql[edit_end:cursor])
        pieces.append(replacement)
        cursor = edit_start
    pieces.append(sql[:cursor])
    pieces.reverse()
    return "".join(pieces), True


def _normalize_arguments(arguments, declared, base_targets, variant_targets, literals=None):
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
    rewritten = query
    changed = False
    if literals:
        restored, literal_changed = _restore_literals(rewritten, literals)
        if literal_changed:
            rewritten = restored
            changed = True
    targets = base_targets
    if variant_targets and _leading_keyword(rewritten) not in _SCHEMA_KEYWORDS:
        targets = sorted(base_targets + variant_targets, key=lambda item: item[0], reverse=True)
    rewritten, double_quoted_changed = _requote_double_quoted(rewritten, declared)
    rewritten, bare_changed = _quote_identifiers(rewritten, targets)
    if not (changed or double_quoted_changed or bare_changed):
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
            literals = _question_literals(public_history)
            if has_calls:
                base_targets, variant_targets = _build_targets(identifiers)
                if not (base_targets or variant_targets or literals):
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
                        function.get("arguments"),
                        identifiers,
                        base_targets,
                        variant_targets,
                        literals,
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
            if identifiers or literals:
                base_targets, variant_targets = _build_targets(identifiers)
                normalized, _ = _normalize_arguments(
                    {"query": sql}, identifiers, base_targets, variant_targets, literals
                )
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
