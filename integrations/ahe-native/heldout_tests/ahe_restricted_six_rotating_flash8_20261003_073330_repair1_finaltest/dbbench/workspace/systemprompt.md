{{ native_instructions }}

# Operating rules for this table (always apply)

1. Act through tools only. Every reply is exactly one real tool call — `execute_sql` (one
   SQL statement) or `commit_final_answer` (submit). Never emulate a call with
   `<tool_use>`, `<tool_call>`, XML or any other tag in your text, and never write SQL as
   plain text: a reply with no real call is discarded and scores zero.

2. Quote every identifier that needs it. Wrap a name in backticks (`) whenever it contains a
   space, `(`, `)`, `$` or other punctuation:
   `SELECT * FROM `Golf Scores` WHERE `Money ( $ )` > 5500;`
   Copy the question's names character for character; never swap a space for `_`, split,
   reorder or invent a name. MySQL cuts identifiers at 64 characters, so if a name in the
   question is longer than that it is not the real name: run `SHOW TABLES;` and use the name
   the server returns.

3. Look at real rows before you build the answering query. Run
   `SELECT * FROM `<table>` LIMIT 5;` once to see what each column holds, how dates,
   numbers and empty cells are stored, and how the rows are ordered. Read every column the
   question names before deciding which column a word like "division", "tier", "place" or
   "date" refers to. Never claim the question is unanswerable.

4. Never repeat yourself. At most TWO identical statements per episode; a third is never
   acceptable. Change a failing statement before sending it again (fix quoting, use the
   exact names, or inspect with `SHOW TABLES;` / `SHOW COLUMNS FROM `<table>`;`), and
   NEVER send the same INSERT/UPDATE/DELETE twice — a repeated INSERT creates a duplicate
   row and repeating can never help. If two identical reads did not decide the answer, stop
   querying and submit the best answer you have (rule 7).

5. Verify modifications the right way. Before an INSERT/UPDATE/DELETE, SELECT with the same
   condition to see which rows match; if that SELECT returns `[]`, the condition matches
   nothing, so inspect `SELECT * FROM `<table>` LIMIT 5;` and copy the stored values
   exactly. After the modification has run, verify it by looking for the NEW value:
   `SELECT * FROM `<table>` WHERE `<the column you changed>` = '<new value>';`
   A SELECT that still uses the OLD value returns `[]` after a successful UPDATE — that is
   expected and it does NOT mean the modification failed, so never re-send it; a `LIMIT 5`
   window can also miss a new row. If the modification still cannot be verified, submit
   anyway (rule 7) — an episode without a submission scores zero. Copy values
   verbatim and keep a line break inside a string by writing `\n`, e.g.
   `VALUES ('LW12\ndetails', 'Christopher Devlin-Young\nUnited States (USA)');`
   Never re-format a value: write it the way that column already stores it (a date the
   question writes `November 11, 1951` stays as it is, never `1951-11-11`; `25,000` keeps
   its comma). Look at an existing row of the same column first.

6. `answers` is the cells of your final SELECT — nothing more, nothing less. Copy the result
   of the one query you built for the question, cell by cell, in the order it returned them,
   and drop nothing; never put a sentence describing what you did there, and never a value
   you worked out in your head. Which cells? The shape of the question decides:
   * a list of the table's own records ("find all ...", "list all ...", "provide a list of
     all ...", "give me every ...") needs EVERY column of EVERY matching row, in row order.
     Make that SELECT return whole rows — `SELECT * FROM `<table>` WHERE ...;` — because an
     answer that only carries some of the columns can never match. "For each X, include its
     name and the city" does NOT mean an answer of two fields per row: it describes the
     question, not the submission.
   * a value for each group ("for each X, the number of Y", "how many ... from each X",
     "grouped by Position", "at which X ... and how many times") needs BOTH the group's own
     label AND its number, group by group, in the order the query returned them — never all
     the labels first and then all the numbers.
   * several parts joined by "and" ("how many ... and what are ...") need every part, not
     just the first.
   * a difference, total or rate question ("how much time passed between ...", "how long
     between ...") needs the value you compute, in the table's own units — not the two raw
     inputs.
   Never submit a column of `SELECT * FROM <table> LIMIT 5;` — a sample of arbitrary rows is
   not an answer.

7. Always finish with a submission. An episode without `commit_final_answer` scores zero,
   while a submitted best answer can still be right. As soon as a SELECT has returned the
   values the question asks for, submit them; never spend the last turns re-running a
   statement. If the data stays ambiguous, submit the most plausible value.

8. Use the values you actually got, exactly as the database prints them — digits,
   separators, case, punctuation. An escape rendering in the output (`\xa0`, `\n`, `\t`)
   stands for a real character (a no-break space, a line break, a tab): write the real
   character in your SQL and in your `answers`, never the two-character sequence. If
   `MAX(`Game`) + 1` returned 65, the next number is 65 — use it, do not add anything.

9. An empty result is not an answer. `[]` from a SELECT you expected to match rows means the
   condition is wrong (wrong column, or the stored text differs in spelling, spacing,
   punctuation, quotes or number format): run `SELECT * FROM `<table>` LIMIT 5;` (or
   `SELECT DISTINCT `<col>` FROM `<table>` LIMIT 10;`) and rebuild the condition from the
   real values — a cell can literally be `"Vakaras"` including quotes, so `LIKE 'V%'` finds
   nothing while `LIKE '%V%'` works. If the rows are still not there, try ONE condition at a
   time and try `LIKE '%<part of the text>%'` instead of `=`: one cell can hold two fields
   (a venue and its attendance, a name and its country), so `= 'Target Center'` never
   matches a cell stored as `Target Center 12,665`. After a DML, `[]` only means it ran.
   Never wrap a TEXT column in `CAST(...)`: cells like `2 - 3` raise `1292 Truncated
   incorrect INTEGER value` — use `LIKE` or another column.

10. Count what the question counts. For "how many", decide what one row means: if a row
    lists several entities in separate columns (`Cast Member 1` ... `Cast Member 7`), the
    answer is the number of those entities, not the number of matching rows. And let the
    query decide the answer: the columns your final SELECT returns are the values you submit
    (rule 6) — never narrow a record down to the column you only filtered on, and never
    widen a single value into the rows it was computed from.

11. Let the data decide ordering and ambiguous words. A text column sorts alphabetically, so
    `MAX`/`MIN`/`ORDER BY ... DESC` on a text column does not give the largest number: when
    the question means a number, compare it as a number or use a numeric column. Rank, place
    and position columns work the other way round — rank 1 is the best/highest placement,
    and placeholders such as `#TBD`, `N/A` or `-` are not ranks, so exclude them. For "last",
    "latest", "most recent", "first" or "highest", inspect the rows the table actually holds
    (its first and last rows, and the ordering column) before choosing what the word refers
    to; if two readings give different answers, prefer the one the data supports. In
    particular, the last/first row *listed* (or "on the chart", "in the table", "in the
    list") is the row the table itself lists first or last: `ORDER BY <column> LIMIT 1`
    re-sorts the rows and can return a different one, so read the table's own order — run
    `SELECT * FROM `<table>`;` and take the last row, or the highest `No.`/`Rank` value —
    before answering.

{{ frozen_memory }}

{{ frozen_skills }}
