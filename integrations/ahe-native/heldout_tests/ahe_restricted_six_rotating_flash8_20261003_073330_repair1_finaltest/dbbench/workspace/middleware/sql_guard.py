"""Deterministic SQL guard for the DB-Bench native agent (v3).

Pure, synchronous middleware.  It never calls a model, the environment, the
network or the filesystem.  It only:

* reads the question already present in the conversation to learn the exact
  table / column identifiers (episode-local state),
 * repairs ordinary tool arguments before dispatch (adds backticks around
   identifiers that cannot be written unquoted, turns escape renderings such
   as ``\\xa0`` / ``\\n`` inside a backticked identifier into the real
   characters the server actually stores, and makes the read of a
   "list of all ..." question return whole rows instead of the few columns the
   wording happens to name),
 * conditions the feedback the model sees after a statement with SHORT, correct
   notes: a failed statement, a statement that matched no row, a statement
   that modified rows (an INSERT/UPDATE/DELETE is never to be repeated, and the
   verification SELECT must look for the NEW value), and a read that sorted a
   column to pick the row the table lists first/last,* refuses to execute the same statement a third time and, near the end of the
  episode budget, replaces a doomed read with a submission built from the
  evidence already collected - an episode that never calls
  ``commit_final_answer`` scores zero,
 * refuses (once) a submission whose values are a column of a plain
   ``SELECT * ... LIMIT n`` sample, or that drops a column the question needs
   (whole records, or a value for each group),
 * completes the ordinary action when the submitted ``answers`` hold only part
   of the result they were drawn from: the submission is re-issued with every
   cell of that read, in the read's order, for the two question shapes whose
   answer is the whole result (a list of records, or a value for each group),* repairs the normal action when the model answers with plain text or emits a
  tool call inside its text.
Everything is bounded per episode so the agent can never get stuck here.
"""
from __future__ import annotations

import json
import re

from nexau.archs.main_sub.execution.hooks import HookResult, Middleware
from nexau.archs.main_sub.execution.parse_structures import ParsedResponse, ToolCall
from nexau.core.messages import Message, ToolUseBlock

_TABLE_RE = re.compile(
    r"name of th[ei]s? table is\s+(?P<table>[^\n]+?)\s*,\s*and the headers of (?:this|the) tables? are\s+(?P<cols>[^\n]+)",
    re.IGNORECASE,
)
_COLS_RE = re.compile(
    r"headers of (?:this|the) tables? are\s+(?P<cols>[^\n]+)",
    re.IGNORECASE,
)
_ERROR_RE = re.compile(
    r"\b\d{4}\s*\(\d{5}\)|\(\d{5}\)|\berror in your sql\b|\bunknown column\b|\bdoesn't exist\b"
    r"|\bsyntax to use near\b|\byou have an error\b|\bcheck the manual that corresponds\b"
    r"|\btruncated incorrect\b|\bincorrect (?:integer|double|decimal|date|datetime|time) value\b"
    r"|\bunknown table\b|\bunknown database\b|\bduplicate entry\b|\bcolumn count doesn't match\b"
    r"|\bcolumn cannot be null\b|\bout of range value\b|\bdata too long\b|\bevery derived table must have\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[0-9A-Za-z_]")
_PLAIN_RE = re.compile(r"[0-9A-Za-z_]+")
_DML_RE = re.compile(r"^\s*(insert|update|delete|replace|alter|drop|truncate)\b", re.IGNORECASE)
_READ_RE = re.compile(r"^\s*(select|show|desc|describe|explain|with|table)\b", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_CAST_ERR_RE = re.compile(
    r"truncated incorrect|incorrect (?:integer|double|date|datetime|time) value|\b1292\b",
    re.IGNORECASE,
)
_LONG_IDENT_RE = re.compile(r"\b1059\b|is too long", re.IGNORECASE)
_HEX_RE = re.compile(r"\\x([0-9a-fA-F]{2})")
_SET_RE = re.compile(
    r"\bset\s+(?P<col>`[^`]+`|[A-Za-z0-9_ ]+?)\s*=\s*(?P<val>'(?:[^']|'')*'|\"[^\"]*\"|-?\d+(?:\.\d+)?)",
    re.IGNORECASE | re.DOTALL,
)
_INSERT_VALUES_RE = re.compile(r"\bvalues\s*\((?P<vals>.*?)\)\s*;?\s*$", re.IGNORECASE | re.DOTALL)
_INSERT_COLS_RE = re.compile(r"\binto\s+[^()]*\((?P<cols>[^()]*)\)", re.IGNORECASE | re.DOTALL)
_NAME_RE = re.compile(r'"name"\s*:\s*"([A-Za-z0-9_]+)"')
_QUERY_RE = re.compile(r'"query"\s*:\s*("(?:[^"\\]|\\.)*")')
_ANSWERS_RE = re.compile(r'"answers"\s*:\s*(\[.*?\])', re.DOTALL)
_ANSWERS_STR = re.compile(r'"answers"\s*:\s*("(?:[^"\\]|\\.)*")')
_PROBE_RE = re.compile(r"^\s*(?:select|table)\s+\*\s+from\b", re.IGNORECASE)
_QUESTION_MARK = "Question:"
_RECORDS_Q_RE = re.compile(
    r"\b(records?|rows?|entries|entr(?:y|ies)|list of all|all of the)\b", re.IGNORECASE
)
_EACH_Q_RE = re.compile(r"\b(each|per|every)\b", re.IGNORECASE)
_COUNT_Q_RE = re.compile(
    r"\b(how many|number of|count(?:s|ed|ing)?|how often|how many times|per (?:country|group))\b",
    re.IGNORECASE,
)
_XML_TOOL_RE = re.compile(r"<tool_name>\s*([A-Za-z0-9_]+)\s*</tool_name>", re.IGNORECASE)
_XML_PARAM_RE = re.compile(r"<parameter>(.*?)</parameter>", re.DOTALL | re.IGNORECASE)
_XML_QUERY_RE = re.compile(r"<query>(.*?)</query>", re.DOTALL | re.IGNORECASE)
_XML_ANSWERS_RE = re.compile(r"<answers>(.*?)</answers>", re.DOTALL | re.IGNORECASE)

_SQL_TOOL = "execute_sql"
_COMMIT_TOOL = "commit_final_answer"

# MySQL truncates identifiers at 64 characters; anything longer cannot be used.
_MAX_IDENT = 64
_MAX_NUDGES = 3
_MAX_ANSWERS = 10
# The largest read a submission may be completed from; a bigger grid is left alone
# rather than turning one commit into a huge submission.
_MAX_ANSWER_CELLS = 120
# The native environment cuts the episode off after ~15 actions, so the guard
# starts warning after 10 statements and forces a submission before the end.
_BUDGET = 15
_NUDGE_AT = 10
_FORCE_AT = 13
_LAST_CHANCE = 14
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", "\\": "\\"}
_HEX_ESC_RE = re.compile(r"\\x([0-9a-fA-F]{2})")

# A question that asks for a list/set of the table's rows ("find all games ...",
# "provide a list of all skyscrapers ...", "give me every ...").
_ALL_Q_RE = re.compile(r"\b(all|every)\b", re.IGNORECASE)
# A question that asks for a value per group ("for each year, how many ...",
# "grouped by their Position", "how many ... per city").
_GROUP_Q_RE = re.compile(
    r"\bgrouped by\b|\bgroup\s+by\b"
    r"|\bfor each\b[^.]{0,60}\b(?:how many|number of|count)\b"
    r"|\bfor every\b[^.]{0,60}\b(?:how many|number of|count)\b"
    r"|\b(?:how many|number of|count)\b[^.]{0,60}\bfor each\b"
    r"|\bper\b[^.]{0,40}\b(?:how many|number of|count)\b"
    r"|\b(?:how many|number of|count)\b[^.]{0,40}\bper\b",
    re.IGNORECASE,
)
# A question about the row the table lists first/last ("the last listed pennant",
# "the last tournament on the chart").
_LISTED_Q_RE = re.compile(
    r"\b(?:last|first|final|earliest|latest)\b[^.?]{0,60}"
    r"\b(?:listed|chart|table|row|entry|record|item|line)\b"
    r"|\b(?:listed|chart|table)\b[^.?]{0,30}\b(?:last|first|final)\b",
    re.IGNORECASE,
)
# A read whose row is chosen by sorting a column instead of the table's own order.
_ORDER_LIMIT1_RE = re.compile(r"\border\s+by\b[^;]{0,200}?\blimit\s+1\b", re.IGNORECASE)
# A plain single-table SELECT whose projection list can be widened to `*`.
_PLAIN_SELECT_RE = re.compile(
    r"^\s*select\s+(?P<proj>.*?)\s+from\s+(?P<rest>.*)$", re.IGNORECASE | re.DOTALL
)
_WIDE_BLOCKED = (" distinct ", " group by ", " having ", " union ", " join ", " into ", " over ")


def _subsequence(values: list, cells: list) -> bool:
    """True when ``values`` appear in ``cells`` in the same relative order."""
    index = 0
    for value in values:
        while index < len(cells) and cells[index] != value:
            index += 1
        if index >= len(cells):
            return False
        index += 1
    return True


def _unescape(text: str) -> str:
    """Undo the backslash escapes MySQL's raw repr uses inside a value.

    The raw output is a Python repr, so a real no-break space shows up as the
    four characters ``\\xa0`` and a line break inside a cell as ``\\n``; each
    stands for the single character the server actually stores.  Leaving them
    as text (``xa0``) silently corrupts every value copied out of a result.
    """
    if "\\" not in text:
        return text
    text = _HEX_ESC_RE.sub(lambda m: chr(int(m.group(1), 16)), text)
    out = []
    index = 0
    size = len(text)
    while index < size:
        char = text[index]
        if char == "\\" and index + 1 < size:
            nxt = text[index + 1]
            out.append(_ESCAPES.get(nxt, nxt))
            index += 2
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _row_major(answers: list, rows: list):
    """Rebuild a column-major submission of a two-column result in row order.

    A "for each X, how many Y" question returns (label, number) pairs and the
    committed list is compared value by value, so the values must alternate
    label, number, label, number ... in the order the rows came back.  Every
    label followed by every number holds the same values in the wrong order.
    The rewrite is applied only when the answers are exactly the two columns of
    the most recent read, so it can never touch an unrelated answer.
    """
    if not rows or len(rows) < 2:
        return None
    if any(len(row) != 2 for row in rows):
        return None
    flat = [str(item).strip() for item in answers]
    size = len(rows)
    if len(flat) != 2 * size:
        return None
    first = [row[0].strip() for row in rows]
    second = [row[1].strip() for row in rows]
    if flat[:size] != first or flat[size:] != second:
        return None
    rebuilt = []
    for row in rows:
        rebuilt.append(row[0].strip())
        rebuilt.append(row[1].strip())
    return rebuilt


def _split_tuples(text: str) -> list:
    """Return the top-level parenthesised groups of a MySQL raw result."""
    out = []
    depth = 0
    quote = ""
    start = None
    index = 0
    size = len(text or "")
    while index < size:
        char = text[index]
        if quote:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                quote = ""
            index += 1
            continue
        if char in ("'", '"'):
            quote = char
            index += 1
            continue
        if char == "(":
            if depth == 0:
                start = index + 1
            depth += 1
        elif char == ")":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    out.append(text[start:index])
                    start = None
        index += 1
    return out


def _cells(group: str) -> list:
    """Extract the cell values of one result row."""
    cells = []
    index = 0
    size = len(group or "")
    while index < size:
        char = group[index]
        if char in ("'", '"'):
            quote = char
            index += 1
            buffer = []
            while index < size:
                if group[index] == "\\" and index + 1 < size:
                    buffer.append(group[index : index + 2])
                    index += 2
                    continue
                if group[index] == quote:
                    break
                buffer.append(group[index])
                index += 1
            index += 1
            cells.append(_unescape("".join(buffer)))
            continue
        if char.isdigit() or (char == "-" and index + 1 < size and group[index + 1].isdigit()):
            start = index
            index += 1
            while index < size and (group[index].isdigit() or group[index] in ". -"):
                index += 1
            cells.append(group[start:index])
            continue
        index += 1
    return cells


def _rows_from_result(text: str) -> list:
    rows = []
    for group in _split_tuples(text):
        cells = _cells(group)
        if cells:
            rows.append(cells)
    return rows


def _columns_of(rows: list) -> list:
    """Turn result rows into columns (one list of cell strings per column)."""
    if not rows:
        return []
    width = max(len(row) for row in rows)
    columns = []
    for index in range(width):
        columns.append(
            [(row[index].strip() if index < len(row) else "") for row in rows]
        )
    return columns


def _is_probe(query: str) -> bool:
    """True for a plain "look at the data" statement.

    ``SELECT * FROM t [LIMIT n]`` samples arbitrary rows and is never built for
    the question; ``SELECT * FROM t WHERE ...`` is a real query.
    """
    flat = _WS_RE.sub(" ", (query or "").lower())
    if not _PROBE_RE.match(flat):
        return False
    return " where " not in flat


class SqlGuardMiddleware(Middleware):
    """Repair SQL arguments, keep the agent acting, and keep it terminating."""

    def __init__(self, max_nudges: int = _MAX_NUDGES) -> None:
        self.max_nudges = max_nudges
        self.nudges = 0
        self.facts_ready = False
        self.facts_table = ""
        self.facts_cols = []
        self.facts_truncated = False
        self.sql_calls = 0
        self.repairs = 0
        # how many times each statement (normalised) has actually been executed
        self.executed = {}
        self.attempted = set()
        self.probes_sent = set()
        self.ever_read = False
        self.saw_rows = False
        self.last_rows = None
        # every read built for the question (never a plain sample), newest last
        self.answer_reads = []
        # the "first/last row the table lists" hint is allowed once per episode
        self.listed_noted = False
        self.committed = False
        self.gate_used = False
        self.forced = False
        # text of the question itself (used to judge the expected answer shape)
        self.question = ""
        # the last plain "look at the data" inspection and the rows it returned
        self.probe_query = ""
        self.probe_rows = None
        # the shape nudge is allowed at most once per episode
        self.shape_gate_used = False
        # text that explains a replaced statement; attached to its output
        self.pending_note = ""
        # whether any INSERT/UPDATE/DELETE has run in this episode
        self.executed_dml = False
        # whether the one-time "look at the table's own rows" hint was shown
        self.ground_noted = False

    # ------------------------------------------------------------------ facts
    def _scan_facts(self, messages) -> None:
        if not messages:
            return
        if not self.question:
            for message in messages:
                try:
                    text = message.get_text_content()
                except Exception:
                    continue
                if not text or _QUESTION_MARK not in text:
                    continue
                self.question = text.split(_QUESTION_MARK, 1)[1].strip()
                break
        if self.facts_ready:
            return
        for message in messages:
            try:
                text = message.get_text_content()
            except Exception:
                continue
            if not text or "table" not in text.lower():
                continue
            table = ""
            cols_raw = ""
            found = _TABLE_RE.search(text)
            if found:
                table = found.group("table")
                cols_raw = found.group("cols")
            else:
                found = _COLS_RE.search(text)
                if found:
                    cols_raw = found.group("cols")
            if not table and not cols_raw:
                continue
            table = _WS_RE.sub(" ", table or "").strip().strip("`").strip()
            if table and (
                len(table) > _MAX_IDENT
                or "\n" in table
                or "question" in table.lower()
                or "header" in table.lower()
            ):
                if len(table) > _MAX_IDENT:
                    self.facts_truncated = True
                    self.facts_ready = True
                table = ""
            cols = []
            for raw in self._split_columns(cols_raw):
                name = raw.strip().strip("`").strip()
                if not name or len(name) > _MAX_IDENT or "\n" in name:
                    continue
                cols.append(name)
            if not table and not cols:
                continue
            self.facts_table = table
            self.facts_cols = cols
            self.facts_ready = True
            return

    @staticmethod
    def _split_columns(raw: str) -> list:
        parts = []
        current = []
        depth = 0
        for char in raw or "":
            if char == "(":
                depth += 1
            elif char == ")":
                if depth > 0:
                    depth -= 1
            if char == "," and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(char)
        parts.append("".join(current))
        cleaned = []
        for part in parts:
            part = part.strip().strip(".").strip()
            if part:
                cleaned.append(part)
        return cleaned

    def _table_name(self) -> str:
        return self.facts_table or ""

    def _quote_list(self) -> str:
        columns = []
        for col in self.facts_cols:
            if col and col != self.facts_table:
                columns.append("`" + col + "`")
        if not self.facts_table and not columns:
            return ""
        parts = []
        if self.facts_table:
            parts.append("table `" + self.facts_table + "`")
        if columns:
            parts.append("columns " + ", ".join(columns))
        return "; ".join(parts)

    def _probe_statement(self) -> str:
        """The statement to run for a first look at the data."""
        if self.facts_table:
            return "SELECT * FROM `%s` LIMIT 5;" % self.facts_table
        return "SHOW TABLES;"

    def _probe_hint(self) -> str:
        if self.facts_truncated and not self.facts_table:
            return (
                " The table name in the question is longer than MySQL's 64-character identifier "
                "limit, so it is not the real name - use the name `SHOW TABLES;` returns, verbatim."
            )
        return ""

    # ------------------------------------------------------------- sql repair
    def _identifiers(self) -> list:
        seen = set()
        out = []
        for name in [self.facts_table] + list(self.facts_cols):
            if not name or len(name) > _MAX_IDENT:
                continue
            if _PLAIN_RE.fullmatch(name):
                # already writable without quoting
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(name)
        out.sort(key=len, reverse=True)
        return out

    @staticmethod
    def _unescape_identifiers(sql: str) -> str:
        """Turn escape renderings inside backticked names into real characters.

        Raw MySQL output is a Python repr, so a real no-break space shows up as
        ``\\xa0`` and a line break inside a column name as ``\\n``.  When the
        model copies such a name it copies the escape text, and MySQL then
        reports ``1054 Unknown column 'Preceded\\xa0by ...'``.  Repairing the
        argument is deterministic and always correct: the identifier really
        does contain that character.
        """
        if "\\" not in sql:
            return sql
        out = []
        index = 0
        size = len(sql)
        while index < size:
            char = sql[index]
            if char == "`":
                end = sql.find("`", index + 1)
                if end < 0:
                    out.append(sql[index:])
                    break
                inner = sql[index + 1 : end]
                if "\\" in inner:
                    inner = _HEX_RE.sub(lambda m: chr(int(m.group(1), 16)), inner)
                    inner = inner.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
                out.append("`" + inner + "`")
                index = end + 1
                continue
            out.append(char)
            index += 1
        return "".join(out)

    def _quote_identifiers(self, sql: str) -> str:
        if not sql:
            return sql
        identifiers = self._identifiers()
        if not identifiers:
            return sql
        lowered = [(name, name.lower()) for name in identifiers]
        out = []
        index = 0
        size = len(sql)
        while index < size:
            char = sql[index]
            if char == "`":
                end = sql.find("`", index + 1)
                if end < 0:
                    out.append(sql[index:])
                    break
                out.append(sql[index : end + 1])
                index = end + 1
                continue
            if char in ("'", '"'):
                end = index + 1
                while end < size:
                    if sql[end] == "\\":
                        end += 2
                        continue
                    if sql[end] == char:
                        if end + 1 < size and sql[end + 1] == char:
                            end += 2
                            continue
                        end += 1
                        break
                    end += 1
                out.append(sql[index:end])
                index = end
                continue
            hit = ""
            for name, low in lowered:
                if sql[index : index + len(name)].lower() == low:
                    before = sql[index - 1] if index > 0 else " "
                    after = sql[index + len(name)] if index + len(name) < size else " "
                    if not _WORD_RE.match(before) and not _WORD_RE.match(after):
                        hit = name
                        break
            if hit:
                out.append("`" + hit + "`")
                index += len(hit)
                continue
            out.append(char)
            index += 1
        return "".join(out)

    def _repair(self, sql: str) -> str:
        repaired = self._quote_identifiers(self._unescape_identifiers(sql))
        return self._widen_records_select(repaired)

    # ------------------------------------------------------------------- text
    @staticmethod
    def _balanced(text: str, start: int):
        depth = 0
        index = start
        size = len(text)
        quote = ""
        while index < size:
            char = text[index]
            if quote:
                if char == "\\":
                    index += 2
                    continue
                if char == quote:
                    quote = ""
                index += 1
                continue
            if char in ("'", '"'):
                quote = char
                index += 1
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
            index += 1
        return None

    def _decode_call(self, raw: str):
        try:
            obj = json.loads(raw)
        except Exception:
            return None
        if not isinstance(obj, dict):
            return None
        name = obj.get("name") or obj.get("tool_name") or obj.get("tool")
        args = obj.get("arguments")
        if args is None:
            args = obj.get("parameters")
        if args is None:
            args = obj.get("input")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                return None
        if not isinstance(args, dict):
            return None
        if name == _COMMIT_TOOL:
            answers = args.get("answers")
            if isinstance(answers, str):
                answers = [answers]
            if not isinstance(answers, list) or not answers:
                return None
            return ToolCall(tool_name=_COMMIT_TOOL, parameters={"answers": [str(item) for item in answers]})
        if name == _SQL_TOOL:
            query = args.get("query")
            if not isinstance(query, str) or not query.strip():
                return None
            return ToolCall(tool_name=_SQL_TOOL, parameters={"query": query})
        return None

    def _recover_call(self, text: str):
        if not isinstance(text, str):
            return None
        haystack = text.lower()
        if "<tool_call" in haystack:
            cursor = 0
            while True:
                hit = haystack.find("<tool_call", cursor)
                if hit < 0:
                    break
                cursor = hit + 10
                brace = text.find("{", hit)
                if brace < 0:
                    continue
                raw = self._balanced(text, brace)
                if raw is None:
                    continue
                call = self._decode_call(raw)
                if call is not None:
                    return call
            salvaged = self._salvage_call(text[haystack.find("<tool_call") :])
            if salvaged is not None:
                return salvaged
        if "<tool_use" in haystack:
            return self._xml_call(text[haystack.find("<tool_use") :])
        return None

    @staticmethod
    def _xml_call(window: str):
        found = _XML_TOOL_RE.search(window)
        if not found:
            return None
        name = found.group(1)
        body = window[found.end() :]
        block = _XML_PARAM_RE.search(body)
        if not block:
            return None
        inner = block.group(1)
        if name == _SQL_TOOL:
            query = _XML_QUERY_RE.search(inner)
            if not query:
                return None
            value = query.group(1).strip()
            if not value:
                return None
            return ToolCall(tool_name=_SQL_TOOL, parameters={"query": value})
        if name != _COMMIT_TOOL:
            return None
        answers = _XML_ANSWERS_RE.search(inner)
        if not answers:
            return None
        items = []
        for line in answers.group(1).splitlines():
            line = line.strip().strip(",").strip()
            if line[:1] in ("-", "*"):
                line = line[1:].strip()
            line = line.strip().strip('"').strip("'")
            if line:
                items.append(line)
        if not items:
            return None
        return ToolCall(tool_name=_COMMIT_TOOL, parameters={"answers": items})

    @staticmethod
    def _salvage_call(window: str):
        found = _NAME_RE.search(window)
        if not found:
            return None
        name = found.group(1)
        if name == _SQL_TOOL:
            query = _QUERY_RE.search(window)
            if not query:
                return None
            try:
                value = json.loads(query.group(1))
            except Exception:
                return None
            if not isinstance(value, str) or not value.strip():
                return None
            return ToolCall(tool_name=_SQL_TOOL, parameters={"query": value})
        if name != _COMMIT_TOOL:
            return None
        answers = []
        block = _ANSWERS_RE.search(window)
        if block:
            try:
                parsed = json.loads(block.group(1))
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, str) and item:
                        answers.append(item)
        if not answers:
            single = _ANSWERS_STR.search(window)
            if single:
                try:
                    value = json.loads(single.group(1))
                except Exception:
                    value = None
                if isinstance(value, str) and value:
                    answers.append(value)
        if not answers:
            return None
        return ToolCall(tool_name=_COMMIT_TOOL, parameters={"answers": answers})

    # ------------------------------------------------------------------ notes
    def _note(self, body: str) -> str:
        return "\n\n[SQL_GUARD] " + body

    def _budget_line(self) -> str:
        if self.committed or self.sql_calls < _NUDGE_AT:
            return ""
        left = max(_BUDGET - self.sql_calls, 0)
        return (
            " You have already sent %d SQL statements and the episode is cut off after about %d: "
            "if you have the values the question asks for, submit them with commit_final_answer NOW - "
            "an episode that never commits scores zero. (%d statements left)"
            % (self.sql_calls, _BUDGET, left)
        )

    def _error_note(self, query: str, value: str = "") -> str:
        lines = ["The statement above FAILED and did not run."]
        if _LONG_IDENT_RE.search(value or ""):
            lines.append(
                "MySQL cuts identifiers at 64 characters, so the name in the question is longer than "
                "the real table name. Run `SHOW TABLES;` and then `SHOW COLUMNS FROM` the exact name it "
                "returns, and copy that name character for character - do not retype the long name from "
                "the question."
            )
        else:
            listed = self._quote_list()
            if listed:
                lines.append(
                    "Use the question's exact names, each wrapped in backticks: " + listed + "."
                )
            if _CAST_ERR_RE.search(value or ""):
                lines.append(
                    "This column holds text (cells like `2 - 3`), so it must never be CAST to a number: "
                    "filter it with LIKE or with another column."
                )
            else:
                lines.append(
                    "Rewrite the statement (names, backticks, one statement) before sending another."
                )
        lines.append("Never send this same statement unchanged.")
        return self._note(" ".join(lines))

    def _dml_note(self, query: str) -> str:
        """Correct post-modification guidance.

        A verification SELECT that keeps the OLD condition naturally returns []
        after a successful UPDATE, and a SELECT of the whole table may not show
        a freshly inserted row inside its LIMIT.  Both were being read as "the
        modification did not apply", which made episodes re-run the very same
        INSERT/UPDATE until the budget ran out (and, for INSERTs, silently
        create duplicate rows).  The note below states the right check.
        """
        lines = [
            "This statement MODIFIED the database and has been executed ONCE.",
            "Do NOT send it again: a repeated INSERT adds a duplicate row and a repeated "
            "UPDATE/DELETE can change more rows than intended. Editing the same statement only to "
            "change a value is a new statement and is fine.",
        ]
        suggestion = self._verification_select(query)
        if suggestion:
            lines.append(
                "Verify by looking for the NEW value: `%s`. If it returns the new row/values the change "
                "applied - submit with commit_final_answer." % suggestion
            )
        else:
            lines.append(
                "Verify by running one SELECT that looks for the NEW value you just wrote. If it returns "
                "the new row/values the change applied - submit with commit_final_answer."
            )
        lines.append(
            "A SELECT that still uses the OLD value returns `[]` after a successful UPDATE: that is "
            "expected and does NOT mean the change failed."
        )
        return self._note(" ".join(lines))

    def _empty_note(self, query: str) -> str:
        lines = [
            "`[]` means the condition matched NO row - it is not an answer.",
            "If you expected rows, the condition does not match the stored text: cells may carry "
            "surrounding quotes, extra spaces or different punctuation or number formats (a cell can "
            "literally be `\"Vakaras\"` including the quotes, so `LIKE 'V%'` finds nothing while "
            "`LIKE '%V%'` works).",
            "Run `%s` (or `SELECT DISTINCT <column> FROM <table> LIMIT 10;`) and rebuild the "
            "condition from the values you actually see." % self._probe_statement()
            + self._probe_hint(),
            "Then try ONE condition at a time, and try `LIKE '%<part of the text>%'` instead of "
            "`=`: a cell often packs two fields together (a venue and a number in one cell, a name "
            "and its country), so `= 'Target Center'` never matches `Target Center 12,665`.",
        ]
        if _DML_RE.match(query or ""):
            lines.append(
                "If you just deleted those rows, an empty result is expected: check that the rows you "
                "meant to keep are still there and submit."
            )
        return self._note(" ".join(lines))

    def _need_action_note(self) -> str:
        lines = [
            "Your previous reply contained NO tool call, so nothing was executed. Reply with exactly ONE "
            "tool call now: `execute_sql` (one SQL statement) or `commit_final_answer` (submit).",
            "Invoke it through the native function-calling interface - never write `<tool_use>`, "
            "`<tool_call>` or any other tag inside your text, and never write SQL as plain text.",
        ]
        listed = self._quote_list()
        if listed:
            lines.append("The question uses: " + listed + ".")
        return self._note(" ".join(lines))

    def _need_query_note(self) -> str:
        lines = [
            "You tried to commit before reading anything, so the answer cannot come from the database.",
            "First run `execute_sql`, then submit.",
        ]
        listed = self._quote_list()
        if listed:
            lines.append("The question uses: " + listed + ".")
        return self._note(" ".join(lines))

    def _no_rows_note(self) -> str:
        return self._note(
            "You are about to submit but you have never seen a single row: every statement so far "
            "returned `[]`. That is not evidence. Run `%s` first, then submit the value the question "
            "asks for." % self._probe_statement()
            + self._probe_hint()
        )

    def _ground_note(self) -> str:
        """The one-time "look at the table itself" hint.

        Failed episodes repeatedly committed the value of a single query without
        ever seeing the table's own rows, so nothing had checked that the column
        they filtered on stores values the way the question writes them, that the
        rows are ordered the way the question assumes, or that the word "last"
        refers to the row they think it does.  The hint costs nothing: it is
        attached to the result the model is already reading and is shown once.
        """
        return self._note(
            "Before you submit, read some rows of `%s` themselves (%s): nothing has yet confirmed how "
            "each column really stores its values or how the rows are ordered. If the question turns on "
            "'last', 'first' or 'highest', also read the rows that decide that ordering, in both "
            "directions." % (self.facts_table, self._probe_statement())
        )

    # ---------------------------------------------------------------- helpers
    def _new_parsed(self, hook_input, calls) -> ParsedResponse:
        return ParsedResponse(
            original_response=hook_input.original_response or "",
            tool_calls=list(calls),
            sub_agent_calls=[],
            batch_agent_calls=[],
            model_response=hook_input.model_response,
        )

    @staticmethod
    def _signature(query: str) -> str:
        return _WS_RE.sub(" ", (query or "").strip().lower())

    def _observe(self, query: str, value: str, list_shaped: bool) -> None:
        sig = self._signature(query)
        if sig:
            self.executed[sig] = self.executed.get(sig, 0) + 1
        if _DML_RE.match(query or ""):
            self.executed_dml = True
        if not _READ_RE.match(query or ""):
            return
        if not list_shaped:
            return
        self.ever_read = True
        rows = [row for row in _rows_from_result(value or "") if any(cell.strip() for cell in row)]
        if rows:
            self.saw_rows = True
            self.last_rows = rows
            if _is_probe(query or ""):
                self.probe_query = query or ""
                self.probe_rows = rows
            else:
                self.answer_reads.append((query or "", rows))
                if len(self.answer_reads) > 12:
                    del self.answer_reads[0]

    def _forced_answers(self):
        rows = self.last_rows or []
        if self._records_question() or self._group_question():
            for _query, read_rows in reversed(self.answer_reads):
                if max(len(row) for row in read_rows) < 2:
                    continue
                cells = [str(cell).strip() for row in read_rows for cell in row]
                if cells:
                    return cells[:60]
        out = []
        seen = set()
        for row in rows:
            if not row:
                continue
            value = row[0].strip()
            if not value or value in seen:
                continue
            seen.add(value)
            out.append(value)
            if len(out) >= _MAX_ANSWERS:
                break
        if out:
            return out
        flat = []
        for row in rows:
            for cell in row:
                cell = (cell or "").strip()
                if cell and cell not in flat:
                    flat.append(cell)
                if len(flat) >= _MAX_ANSWERS:
                    break
            if len(flat) >= _MAX_ANSWERS:
                break
        return flat or None

    def _verification_select(self, query: str):
        """Build the SELECT that looks for the NEW values a DML wrote."""
        table = self._table_name()
        if not table:
            return None
        flat = _WS_RE.sub(" ", query or "")
        inserted = _INSERT_VALUES_RE.search(flat)
        if inserted:
            cols = _INSERT_COLS_RE.search(flat)
            values = self._split_columns(inserted.group("vals"))
            names = self._split_columns(cols.group("cols")) if cols else []
            if names and values:
                return "SELECT * FROM `%s` WHERE `%s` = %s LIMIT 5;" % (
                    table,
                    names[0].strip().strip("`"),
                    values[0].strip(),
                )
            return None
        updated = _SET_RE.search(flat)
        if updated:
            return "SELECT * FROM `%s` WHERE `%s` = %s LIMIT 5;" % (
                table,
                updated.group("col").strip().strip("`"),
                updated.group("val"),
            )
        return None

    def _probe_call(self, call):
        """A read that has not been tried yet, used to break a loop."""
        table = self._table_name()
        candidates = []
        if table:
            candidates.append("SELECT * FROM `%s` LIMIT 5;" % table)
            candidates.append("SHOW COLUMNS FROM `%s`;" % table)
        candidates.append("SHOW TABLES;")
        for candidate in candidates:
            sig = self._signature(candidate)
            if sig in self.attempted or sig in self.probes_sent:
                continue
            self.probes_sent.add(sig)
            return ToolCall(
                tool_name=_SQL_TOOL,
                parameters={"query": candidate},
                tool_call_id=call.tool_call_id,
            )
        return None

    def _commit_call(self, call, answers):
        return ToolCall(
            tool_name=_COMMIT_TOOL,
            parameters={"answers": [_unescape(str(item)) for item in answers]},
            tool_call_id=call.tool_call_id,
        )

    def _clean_commit(self, hook_input, call):
        """Repair a submission without changing what it says.

        Three deterministic repairs are applied to the committed values:

        * the raw output is a Python repr, so a value copied out of a result can
          still carry an escape rendering (``Soviet Union\\xa0(URS)``); the real
          character is what the database stores and what the answer is compared
          against, so it is written back;
        * a per-group result submitted as every label and then every number is
          put back into row order (see ``_row_major``);
        * a submission that holds only part of the result it was drawn from -
          one column of a multi-column read, or the labels without their numbers
          while the question asks for a value per group - is completed with the
          whole result (see ``_expand_answers``).

        Returns a HookResult that re-issues the corrected commit, or None when
        the arguments are already exactly right.
        """
        params = call.parameters if isinstance(call.parameters, dict) else {}
        answers = params.get("answers") if params else None
        if not isinstance(answers, list) or not answers:
            return None
        cleaned = [_unescape(str(item)) for item in answers]
        rebuilt = _row_major(cleaned, self.last_rows)
        if rebuilt is not None:
            cleaned = rebuilt
        expanded = self._expand_answers(cleaned)
        if expanded is not None:
            cleaned = expanded
        if cleaned == [str(item) for item in answers]:
            return None
        return HookResult(
            parsed_response=self._new_parsed(hook_input, [self._commit_call(call, cleaned)])
        )

    @staticmethod
    def _drop_blocked_messages(messages, blocked_ids):
        if not blocked_ids:
            return list(messages)
        try:
            rebuilt = []
            for message in messages:
                blocks = list(message.content or [])
                keep = []
                for block in blocks:
                    if isinstance(block, ToolUseBlock) and block.id in blocked_ids:
                        continue
                    keep.append(block)
                if len(keep) == len(blocks):
                    rebuilt.append(message)
                elif keep:
                    rebuilt.append(
                        Message(role=message.role, content=keep, metadata=dict(message.metadata or {}))
                    )
            return rebuilt
        except Exception:
            return None

    def _intervention(self, hook_input, call):
        """Decide whether this SQL call must be replaced.

        Returns a HookResult or None.  A statement is never executed a third
        time, a modification is never executed twice, and once the episode has
        burned most of its budget a read is turned into a submission.
        """
        params = call.parameters if isinstance(call.parameters, dict) else None
        query = params.get("query") if params else None
        if not isinstance(query, str) or not query.strip():
            return None
        repaired = self._repair(query)
        sig = self._signature(repaired)
        ran = self.executed.get(sig, 0)
        is_dml = bool(_DML_RE.match(repaired))
        is_read = bool(_READ_RE.match(repaired))
        late = self.sql_calls >= _FORCE_AT
        if ran < 1 and not (late and is_read):
            return None

        answers = self._forced_answers()
        if is_dml:
            # The identical modification already ran once: never run it again.
            if ran < 1:
                # The budget alone must not block a first DML.
                return None
            replacement = self._verification_select(repaired)
            if replacement:
                self.pending_note = self._note(
                    "That exact modification already ran once, so it was NOT executed again - "
                    "repeating an INSERT creates a duplicate row. The SELECT above shows the new state: "
                    "if your row/values are there, submit with commit_final_answer now."
                )
                return HookResult(
                    parsed_response=self._new_parsed(
                        hook_input,
                        [
                            ToolCall(
                                tool_name=_SQL_TOOL,
                                parameters={"query": replacement},
                                tool_call_id=call.tool_call_id,
                            )
                        ],
                    )
                )
            if answers:
                self.pending_note = self._note(
                    "That exact modification already ran once, so it was NOT executed again. "
                    "Submitting the values already collected: " + json.dumps(answers) + "."
                )
                return HookResult(
                    parsed_response=self._new_parsed(hook_input, [self._commit_call(call, answers)])
                )
            if self.sql_calls >= _LAST_CHANCE and not self.committed:
                self.pending_note = self._note(
                    "That exact modification already ran once, so it was NOT executed again. The "
                    "action budget is exhausted, so the episode was ended with a submission - for a "
                    "task that only modifies rows the answer text is not what is scored."
                )
                self.forced = True
                return HookResult(
                    parsed_response=self._new_parsed(
                        hook_input, [self._commit_call(call, ["done"])]
                    )
                )
            return None

        if ran >= 2 or late:
            if answers:
                self.pending_note = self._note(
                    "The statement that was sent again cannot produce new information, so the episode "
                    "was ended by submitting the values already collected: " + json.dumps(answers) + "."
                )
                self.forced = True
                return HookResult(
                    parsed_response=self._new_parsed(hook_input, [self._commit_call(call, answers)])
                )
            if self.sql_calls >= _LAST_CHANCE and not self.committed:
                # Last chance: the environment cuts the episode off after about
                # _BUDGET actions, and an episode that never submits scores zero.
                submission = answers or ["0"]
                self.pending_note = self._note(
                    "The action budget is exhausted, so the episode was ended with a submission: "
                    + json.dumps(submission)
                    + "."
                )
                self.forced = True
                return HookResult(
                    parsed_response=self._new_parsed(
                        hook_input, [self._commit_call(call, submission)]
                    )
                )
            probe = self._probe_call(call)
            if probe is not None:
                self.pending_note = self._note(
                    "The same kind of statement was sent again without progress, so it was replaced by "
                    "a fresh inspection of the table (result above). Read the real values, then either "
                    "fix the condition once or submit."
                )
                return HookResult(parsed_response=self._new_parsed(hook_input, [probe]))
            if ran >= 2 and not self.committed:
                submission = answers or ["0"]
                self.pending_note = self._note(
                    "That statement has already been executed %d times, so it cannot produce new "
                    "information and the episode was ended with a submission: " % (ran + 1)
                    + json.dumps(submission)
                    + "."
                )
                self.forced = True
                return HookResult(
                    parsed_response=self._new_parsed(
                        hook_input, [self._commit_call(call, submission)]
                    )
                )
        return None

    def _shape_note(self, answers):
        """Refuse to submit a column of a statement that cannot be the answer.

        Two shapes were observed in failed episodes:

        * the values are exactly one column of the rows a ``SELECT * ... LIMIT n``
          inspection returned - that statement only sampled arbitrary rows and
          was never built for the question, so its column cannot be the answer;
        * the last read returned several columns per row and the model submitted
          exactly one of them, while the question asks for whole records or for a
          value for *each* group - dropping the other column can never match.

        The note is a bounded nudge (once per episode) and only conditions the
        feedback the model sees; nothing is executed here.
        """
        if self.shape_gate_used or self.executed_dml:
            return None
        values = [str(item).strip() for item in (answers or [])]
        if not values:
            return None
        if self._expand_answers(values) is not None:
            # the commit path completes this submission itself
            return None
        sampled = self.probe_rows
        if sampled is not None and self.last_rows is sampled:
            for column in _columns_of(sampled):
                if column and column == values:
                    return self._note(
                        "The values you submitted are one column of the rows `%s` returned. That "
                        "statement only inspected arbitrary sample rows - it was not built for this "
                        "question - so its columns are not an answer. Run ONE SELECT that returns "
                        "exactly the value(s) the question asks for (name the column(s) the question "
                        "refers to) and submit the values that query returns, one element per value."
                        % (self.probe_query or self._probe_statement())
                    )
        rows = self.last_rows
        if not rows:
            return None
        columns = _columns_of(rows)
        if len(columns) < 2:
            return None
        if not any(column and column == values for column in columns):
            return None
        question = self.question or ""
        if _RECORDS_Q_RE.search(question) and len(columns) >= 3:
            return self._note(
                "Your last SELECT returned %d values per row and you are submitting only one of "
                "them. The question asks for whole records/rows, so an answer needs every column of "
                "each matching row: re-send one SELECT that returns those columns and submit its "
                "cells in the returned row order, one element per value, dropping nothing."
                % len(columns)
            )
        if _EACH_Q_RE.search(question) and _COUNT_Q_RE.search(question):
            return self._note(
                "Your last SELECT returned %d values per row and you are submitting only one of "
                "them. This question asks for a value for EACH group, so each group needs BOTH its "
                "own label and its value: re-send one SELECT that returns the group column and the "
                "value, then submit every returned value, one element per value, group by group, in "
                "the order the query returned them."
                % len(columns)
            )
        return None

    def _records_question(self) -> bool:
        """True for "find/list all ..." questions (not for per-group counts)."""
        question = self.question or ""
        return bool(_ALL_Q_RE.search(question)) and not bool(_GROUP_Q_RE.search(question))

    def _group_question(self) -> bool:
        """True for questions that ask for a value for each group."""
        return bool(_GROUP_Q_RE.search(self.question or ""))

    def _expand_answers(self, values: list):
        """Complete a submission that holds only part of the rows it came from.

        Every answer is compared value by value with the cells of the query the
        question implies.  For "find all ..." questions that query returns whole
        rows and for "for each X, how many Y" questions it returns a label and a
        number per group, so a submission that is only a projection of that
        result - the one column the model filtered on, or the labels without
        their numbers - can never match.  The newest read whose cells contain
        the submitted values in order is used, and the whole result is submitted
        instead.  Only questions of those two shapes are touched, and only when
        the submitted values really are a proper part of that read.
        """
        if self.executed_dml or not values:
            return None
        if not (self._records_question() or self._group_question()):
            return None
        wanted = [_unescape(str(item)).strip() for item in values]
        for _query, rows in reversed(self.answer_reads):
            if max(len(row) for row in rows) < 2:
                continue
            cells = [_unescape(str(cell)).strip() for row in rows for cell in row]
            if len(cells) <= len(wanted) or len(cells) > _MAX_ANSWER_CELLS:
                continue
            if _subsequence(wanted, cells):
                return [cell for row in rows for cell in row]
        return None

    def _widen_records_select(self, sql: str) -> str:
        """Make a "list of all ..." read return whole rows.

        A list-of-records answer needs every column of each matching row, so a
        projection that follows the wording of the question ("include its name,
        the city, and the years ...") silently drops the columns the answer is
        compared on.  Only a plain single-table SELECT without an aggregate,
        DISTINCT, GROUP BY, HAVING, UNION or JOIN is widened, and the condition,
        ordering and limit are left exactly as they were.
        """
        if not self._records_question():
            return sql
        match = _PLAIN_SELECT_RE.match(sql or "")
        if not match:
            return sql
        proj = match.group("proj").strip()
        if proj == "*" or "(" in proj or "`(" in proj:
            return sql
        lowered = " " + _WS_RE.sub(" ", sql or "").lower() + " "
        if proj.lower().startswith("distinct") or any(
            token in lowered for token in _WIDE_BLOCKED
        ):
            return sql
        return "SELECT * FROM " + match.group("rest")

    def _listed_note(self) -> str:
        table = self._table_name() or "<table>"
        return self._note(
            "`ORDER BY` re-sorts the rows, so `ORDER BY ... LIMIT 1` returns the row with the "
            "largest/smallest value of a column - not the row the table lists first/last. This "
            "question asks about the row as the table lists it: read the whole table with no "
            "ORDER BY (`SELECT * FROM %s;`) and use the LAST tuple it returns (if the table "
            "numbers its rows with a column such as `No.` or `Rank`, the highest number is the "
            "last listed row)." % table
        )

    def _commit_gate(self, hook_input, call):
        """Never submit from zero evidence (bounded: at most once)."""
        if self.nudges >= self.max_nudges:
            return None
        note = None
        answers = None
        params = call.parameters if isinstance(call.parameters, dict) else {}
        candidate = params.get("answers") if params else None
        if isinstance(candidate, list):
            answers = candidate
        if self.sql_calls == 0:
            note = self._need_query_note()
        elif self.ever_read and not self.saw_rows and not self.gate_used:
            self.gate_used = True
            note = self._no_rows_note()
        elif answers is not None and not self.shape_gate_used:
            note = self._shape_note(answers)
            if note:
                self.shape_gate_used = True
        if not note:
            return None
        cleaned = self._drop_blocked_messages(hook_input.messages, {call.tool_call_id})
        if cleaned is None:
            return None
        self.nudges += 1
        return HookResult(
            parsed_response=self._new_parsed(hook_input, []),
            messages=list(cleaned) + [Message.user(note)],
            force_continue=True,
        )

    # ------------------------------------------------------------------ hooks
    def before_model(self, hook_input):
        self._scan_facts(hook_input.messages)
        return HookResult.no_changes()

    def before_tool(self, hook_input):
        if hook_input.tool_name != _SQL_TOOL:
            return HookResult.no_changes()
        params = hook_input.tool_input
        if not isinstance(params, dict):
            return HookResult.no_changes()
        query = params.get("query")
        if not isinstance(query, str) or not query.strip():
            return HookResult.no_changes()
        self.sql_calls += 1
        fixed = self._repair(query)
        self.attempted.add(self._signature(fixed))
        if fixed != query:
            self.repairs += 1
            updated = dict(params)
            updated["query"] = fixed
            return HookResult.with_modifications(tool_input=updated)
        return HookResult.no_changes()

    def after_tool(self, hook_input):
        if hook_input.tool_name != _SQL_TOOL:
            return HookResult.no_changes()
        payload = hook_input.tool_output
        if not isinstance(payload, dict):
            return HookResult.no_changes()
        value = payload.get("result")
        if not isinstance(value, str):
            return HookResult.no_changes()
        params = hook_input.tool_input
        query = ""
        if isinstance(params, dict):
            candidate = params.get("query")
            if isinstance(candidate, str):
                query = candidate
        stripped = value.strip()
        list_shaped = stripped.startswith("[")
        self._observe(query, value, list_shaped)
        note = self.pending_note or ""
        self.pending_note = ""
        if _ERROR_RE.search(value) or (not list_shaped and _READ_RE.match(query)):
            note = self._error_note(query, value)
        elif _DML_RE.match(query):
            note = self._dml_note(query)
        elif _READ_RE.match(query):
            if list_shaped and _rows_from_result(value):
                note = note  # a successful read needs no note; keep a pending one
            else:
                note = self._empty_note(query)
        if (
            not self.ground_noted
            and not self.executed_dml
            and self.probe_rows is None
            and self.facts_table
            and list_shaped
            and not _is_probe(query)
            and _rows_from_result(value)
        ):
            self.ground_noted = True
            note += self._ground_note()
        if (
            not self.listed_noted
            and _LISTED_Q_RE.search(self.question or "")
            and _ORDER_LIMIT1_RE.search(query or "")
        ):
            self.listed_noted = True
            note += self._listed_note()
        if note or self._budget_line():
            updated = dict(payload)
            updated["result"] = value + note + self._budget_line()
            return HookResult.with_modifications(tool_output=updated)
        return HookResult.no_changes()

    def after_model(self, hook_input):
        self._scan_facts(hook_input.messages)
        parsed = hook_input.parsed_response
        calls = list(parsed.tool_calls) if (parsed is not None and parsed.tool_calls) else []
        if not calls:
            recovered = self._recover_call(hook_input.original_response or "")
            if recovered is not None:
                return HookResult(parsed_response=self._new_parsed(hook_input, [recovered]))
            if self.nudges >= self.max_nudges:
                return HookResult.no_changes()
            self.nudges += 1
            return HookResult(
                messages=list(hook_input.messages) + [Message.user(self._need_action_note())],
                force_continue=True,
            )
        if len(calls) == 1:
            call = calls[0]
            if call.tool_name == _SQL_TOOL:
                result = self._intervention(hook_input, call)
                if result is not None:
                    return result
            elif call.tool_name == _COMMIT_TOOL:
                result = self._commit_gate(hook_input, call)
                if result is not None:
                    return result
                result = self._clean_commit(hook_input, call)
                self.committed = True
                if result is not None:
                    return result
        return HookResult.no_changes()
