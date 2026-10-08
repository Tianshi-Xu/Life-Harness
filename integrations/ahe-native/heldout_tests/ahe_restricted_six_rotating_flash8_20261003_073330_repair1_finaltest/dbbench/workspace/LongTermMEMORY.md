# Long-term memory — DB-Bench (MySQL) tasks

Environment facts that hold for every task in this benchmark:

* Exactly ONE table per task. The question (`Additional table information about the
  question: ...`) states its exact name and column names — the ground truth.
* Those names are often not valid unquoted SQL identifiers (spaces, parentheses, `$`, `=`),
  so they must be backticked. MySQL cuts identifiers at 64 characters: a longer name in the
  question is not the real name — run `SHOW TABLES;`.
* `execute_sql` prints MySQL's output as a Python repr: `\xa0`, `\n`, `\t` are escape
  renderings of real characters, and a cell can carry its own surrounding quotes.
* A successful `INSERT`/`UPDATE`/`DELETE` also returns `[]`, so `[]` proves nothing about a
  modification. Verify with a SELECT for the NEW value; never send a modification twice.
* `answers` is compared value by value against the reference result — the cells of the query
  the question implies. It must therefore hold every cell of your final SELECT, in order, one
  element each: a "find all / list all ..." question is answered with WHOLE rows (make that
  SELECT `SELECT * FROM <table> WHERE ...;` and submit every column of every matching row,
  even the columns the wording never mentions), and a per-group question with each group's
  label AND its number, group by group.
* `SELECT * FROM `<table>` LIMIT 5;` samples arbitrary rows: its columns are never an
  answer.
* Errors print as `1292 (22007): Truncated incorrect INTEGER value: '2 - 3'`; the statement
  did not run.
* Only `commit_final_answer` submits, and the episode is cut off after a few tool calls — an
  episode that never submits scores zero.

Recurring mistakes to avoid (seen in past episodes):

1. Writing the tool call as text (`<tool_use>`, `<tool_call>`) — nothing runs.
2. Replacing a space in a name with `_` (`Golf_Scores`), which gives 1146 "table doesn't
   exist", or otherwise altering a name.
3. Re-sending an identical statement instead of changing it or submitting.
4. Claiming a column does not exist instead of inspecting real rows.
5. Re-running a modification because the verification SELECT returned nothing or because a
   `LIMIT 5` window did not show the new row, or never submitting at all.
6. Copying an escape rendering (`Preceded\xa0by ...`) into a query or into `answers`.
7. Submitting the wrong column, the wrong order, or a column of a `LIMIT 5` sample.
8. Re-formatting a modification value: `1951-11-11` where the table stores
   `November 11, 1951`, or dropping a thousands separator.
9. Ordering a text column with `MAX`/`MIN`/`ORDER BY` when the question means the number it
   spells, or treating `#TBD` / `N/A` as a rank (rank 1 is the highest placement).
10. Counting rows when the question counts entities listed inside a row
    (`Cast Member 1` ... `Cast Member 7` → 7, not `COUNT(*)` = 1).
11. Casting a text column to a number: `CAST(`Score` AS SIGNED)` on a cell `2 - 3` raises
    1292 — filter with `LIKE`.
12. Answering a "find all ..." / "list all ..." question with only the column that was
    filtered on (the opponent, the name, the label): the answer is every column of every
    matching row, so the answering SELECT must be `SELECT * FROM <table> WHERE ...;`.
13. Reading "the last/first ... listed" (or "on the chart") as the row with the largest or
    smallest value of a column: `ORDER BY <column> LIMIT 1` re-sorts the rows — read the
    table's own order and take its last (first) row.
14. Matching a whole cell with `=` when the cell packs two fields, e.g. the venue and its
    attendance (`Target Center 12,665`): `= 'Target Center'` matches nothing — use
    `LIKE '%Target Center%'`.
