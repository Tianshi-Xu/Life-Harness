"""Meta-Harness candidate: quote the *declared* schema identifiers inside `execute_sql`.

Single, focused mechanism.  The task description hands the solver a table name and a
comma separated header list ("The name of this table is X, and the headers of this
table are A,B,C.").  A small solver model happily emits SQL in which those identifiers
are left bare when they are MySQL reserved words (``Rank``) or contain spaces /
punctuation (``Hockey Games``, ``No. Disc``, ``Top 10``, ``Frequency MHz``).  MySQL
then answers 1064 / 1054 / 1146 and the model re-issues the very same statement until
the turn budget is gone.

This runtime repairs exactly that: before an ``execute_sql`` tool call is dispatched it
backtick-quotes the declared identifiers that need quoting, and only those, inside SQL
*code* regions.  Everything else is identity.

Contract: plain ``AgentHarness``, synchronous hooks, ordinary OpenAI-style dicts in and
out, per-episode instance state only (public metadata observed from the request /
history), imports limited to ``re``/``json``.
"""

import json
import re

# ---------------------------------------------------------------------------
# declared-schema extraction
# ---------------------------------------------------------------------------

_TABLE_INFO_RE = re.compile(
    r"The name of this table is (.+?), and the headers of this table are "
    r"(.*?)(?:\n\s*Question:|\Z)",
    re.I | re.S,
)

# ---------------------------------------------------------------------------
# which declared names actually need quoting
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

# guards used inside the replacement callback
_OPEN_PAREN_RE = re.compile(r"\s*\(")          # identifier used as a function call
_BY_CLAUSE_RE = re.compile(r"[ \t\r\n]+BY\b", re.I)  # ORDER BY / GROUP BY / PARTITION BY

_QUOTE_CHARS = frozenset("'\"`")


def _needs_quoting(name):
    """True iff `name` must be backtick-quoted to be a usable MySQL identifier."""
    if not _SIMPLE_NAME_RE.match(name):
        return True
    return name.lower() in _RESERVED_WORDS


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


class AgentHarness:
    """Identity runtime + declared-schema identifier quoting for `execute_sql`."""

    def __init__(self, public_context=None):
        self.public_context = public_context
        # per-episode state: public metadata only
        self._declared = {}   # lower-cased name -> first spelling seen
        self._needed = {}     # lower-cased name -> declared spelling, needs quoting
        self._tables = set()  # lower-cased declared *table* names
        self._seen_text = set()
        self._pattern = None

    # ------------------------------------------------------------------ hooks

    def transform_request(self, request):
        # fallback: the schema blurb may already be visible in the request messages
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
            if not self._needed:
                return response
            pattern = self._compiled()
            if pattern is None:
                return response
            for call in calls:
                if not isinstance(call, dict):
                    continue
                func = call.get("function")
                if not isinstance(func, dict) or func.get("name") != "execute_sql":
                    continue
                try:
                    self._repair_call(func, pattern)
                except Exception:
                    continue
            return response
        except Exception:
            return response

    # --------------------------------------------------- step 1: extraction

    def _absorb(self, obj):
        """Walk an arbitrary nested public structure and mine schema declarations."""
        stack = [obj]
        while stack:
            item = stack.pop()
            if isinstance(item, str):
                self._absorb_text(item)
            elif isinstance(item, dict):
                stack.extend(item.values())
            elif isinstance(item, (list, tuple, set)):
                stack.extend(item)

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

    def _remember(self, name, is_table=False):
        name = name.strip()
        if not name:
            return
        key = name.lower()
        if is_table:
            self._tables.add(key)
        if key in self._declared:
            return
        self._declared[key] = name
        if _needs_quoting(name):
            self._needed[key] = name
            self._pattern = None

    # ------------------------------------------- step 2/3: rewrite machinery

    def _compiled(self):
        if self._pattern is None and self._needed:
            names = sorted(self._needed, key=lambda s: (-len(s), s))
            body = "|".join(re.escape(name) for name in names)
            self._pattern = re.compile(
                r"(?<![A-Za-z0-9_`])(" + body + r")(?![A-Za-z0-9_])", re.I
            )
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

    def _rewrite(self, sql, pattern):
        parts = []
        changed = False
        for is_code, chunk in _split_regions(sql):
            if is_code:
                new = pattern.sub(self._sub, chunk)
                if new != chunk:
                    changed = True
                parts.append(new)
            else:
                parts.append(chunk)
        return "".join(parts) if changed else sql

    # ------------------------------------------------- step 4: tool call repair

    def _repair_call(self, func, pattern):
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
        rewritten = self._rewrite(query, pattern)
        if rewritten == query:
            return
        new_payload = dict(payload)
        new_payload["query"] = rewritten
        func["arguments"] = (
            json.dumps(new_payload, separators=(",", ":")) if as_text else new_payload
        )
