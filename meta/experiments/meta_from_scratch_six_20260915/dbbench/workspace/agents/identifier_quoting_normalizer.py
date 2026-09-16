"""Runtime that backtick-quotes declared SQL identifiers in outgoing queries.

The prompt declares one table name and its headers in a user message. Any of
those identifiers that the engine cannot accept bare (characters outside
[A-Za-z0-9_], or reserved words) is rewritten with backticks inside outgoing
`execute_sql` arguments. Quoting is skipped inside string literals, inside
already backticked spans, after `.`, and before `(`, so valid SQL is left
untouched. Requests pass through unchanged.
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

# Reserved words whose bare use is rejected by the engine.
_RESERVED_WORDS = frozenset(
    (
        "rank", "order", "group", "key", "index", "usage", "system", "signal",
        "interval", "condition", "current_date", "current_time", "function",
        "references", "partition", "range", "rows", "over", "window",
        "recursive", "cube", "lag", "lead",
    )
)


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


def _build_targets(identifiers):
    declared = set(identifiers)
    targets = []
    for identifier in identifiers:
        if not _needs_quoting(identifier):
            continue
        targets.append((identifier, "`" + identifier + "`"))
        # Declared identifiers may lose trailing punctuation in the model's SQL
        # (e.g. "No." written as "No"); map that stripped form back.
        core = _TRAILING_PUNCTUATION_RE.sub("", identifier)
        if len(core) >= 2 and core != identifier and core not in declared:
            targets.append((core, "`" + identifier + "`"))
    targets.sort(key=lambda item: len(item[0]), reverse=True)
    return targets


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
    for needle, replacement in targets:
        for match in _identifier_pattern(needle).finditer(sql):
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


def _normalize_arguments(arguments, targets):
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
    rewritten, changed = _quote_identifiers(query, targets)
    if not changed:
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
            if not isinstance(tool_calls, (list, tuple)) or not tool_calls:
                return response
            identifiers = _declared_identifiers(public_history)
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
                    function.get("arguments"), targets
                )
                if call_changed:
                    function["arguments"] = arguments
                    changed = True
            if not changed:
                return response
            result = dict(response)
            result["tool_calls"] = new_calls
            return result
        except Exception:
            return response
