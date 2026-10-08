# Skill: DB-Bench SQL playbook

Apply this whenever a task gives you one table and asks a question about it. The question
lists the table name and the header names — write them exactly, and backtick every name that
contains anything other than letters, digits or `_`.

| Question says | Write |
| --- | --- |
| table `Golf Scores`, headers `To par`, `Money ( $ )` | ``SELECT `To par`, `Money ( $ )` FROM `Golf Scores`;`` |
| table `Rugby team performance`, header `Points for ( Tests )` | ``SELECT `Points for ( Tests )` FROM `Rugby team performance`;`` |
| table `Brazil Football Matches`, header `Brazil scorers` | ``INSERT INTO `Brazil Football Matches` (`Date`, `Brazil scorers`) VALUES (...);`` |

## Workflow

1. `SELECT * FROM `<table>` LIMIT 5;` — see what each column holds, how dates, numbers and
   empty cells are formatted, and how the rows are ordered.
2. Build ONE query for the question, using the column whose contents actually match the
   words in the question. If several columns could match (`Division` vs `Tier`, `Gold` vs
   `Total`), read them all before deciding. If the question asks for a list of the table's
   records ("find all ...", "list all ..."), write it as `SELECT * FROM <table> WHERE ...;` —
   the reference answer is the whole rows, however few fields the wording names.
3. Read the rows the query returned: count them, and check every cell the question asks
   about. If the result looks empty or short, the condition does not match the stored text —
   see "Text matching" below.
4. `commit_final_answer` with the cells of that query as `answers`, one element per cell, in
   the order the query returned them — every cell, including the columns the wording of the
   question never mentions.

## Answer shape — what `answers` must hold

`answers` is always the cell list of your final SELECT. What that SELECT has to return depends
on the question:

| Question wording | Final SELECT must return | `answers` must contain |
| --- | --- | --- |
| "find all ...", "list all ...", "provide a list of all ...", "give me every ..." | whole rows: `SELECT * FROM <table> WHERE ...;` | every cell of every matching row, in row order — including columns the question never mentions |
| "for each X, the number of Y", "grouped by ...", "how many ... per X" | the group column plus its number | each group's label AND its number, alternating in the order the query returned them |
| "how many ... and what are ..." | both parts | every part, not just the first |
| "what is the `<col>` of the ...", "which `<col>` ..." | exactly that column | that column's values, for the row the question means |
| "how many ..." | `SELECT COUNT(...)` | the single number |
| "how much time passed / how long between X and Y" | the difference | the value you compute, in the table's own units |

* A question that names a few fields ("for each team, include the team name, the city, and the
  years") does not shrink the answer: if it asks for the table's records, submit whole rows —
  a submission that carries only some of the columns cannot match the reference result.
* Keep the order the query returned (or the order the question lists, e.g.
  Freshman, Sophomore, Junior, Senior).
* Formatting matters: copy each value the way the database printed it (case, separators,
  punctuation). `\xa0` / `\n` / `\t` in the output are escape renderings of real characters —
  write the real character, never the two-character sequence.

## Counting questions

* Decide first what one row represents. If a row lists repeated columns (`Cast Member 1` ...
  `Cast Member 7`, `2007`, `2008`, `2009`), the answer is how many of those cells are
  filled, not `COUNT(*)`. `COUNT(*)` is only the same thing when one row is one counted
  entity.
* "The value of X when Y is Z" wants the value of column X, not the column you filtered on.
* For counting/grouping questions use `GROUP BY` on the exact column and read the result
  rows; do not summarise from memory.

## Text matching

* `LIKE '%word%'` matches substrings, not words: `LIKE '%Back%'` also matches `Linebacker`
  and `Quarterback`. When the question says 'contains the word "Back"', look at
  `SELECT DISTINCT `<col>` FROM `<table>` LIMIT 10;` first and decide whether a
  word-boundary match (`LIKE '% Back'` / `LIKE 'Back%'`) is right.
* Cells can carry surrounding quotes, extra spaces or different number formats, so a prefix
  pattern can miss every row (`LIKE 'D%'` finds nothing when the cell starts with `"`).
  Copy the stored spelling instead of retyping it from the question.
* A condition that returns `[]` is wrong, not empty: try ONE condition at a time, and try
  `LIKE '%<part of the text>%'` instead of `=`. One cell often packs two fields together —
  a venue and its attendance (`Target Center 12,665`), a name and its country — so
  `WHERE `Location Attendance` = 'Target Center'` never matches anything.

## Symptom → fix

* "the last/first ... listed", "on the chart", "in the table" → the row the table itself
  lists first or last. Read `SELECT * FROM `<table>`;` and take the last (first) row, or
  the highest (lowest) `No.`/`Rank`; `ORDER BY <column> LIMIT 1` re-sorts the rows and can
  return a different row, which is why the answer looked right but was not.
* `1064 ... syntax to use near` / `1146 ... doesn't exist` → a name with a space was not
  quoted, or a name was altered. Copy the name from the question and backtick it.
* `1054 Unknown column` → wrong column name; `SHOW COLUMNS FROM `<table>`;` and use the
  real one.
* `1292 Truncated incorrect INTEGER value` → the cell holds text such as `2 - 3`; never cast
  a text column — filter with `LIKE` or another column.
* Empty result where you expected rows → the condition is too strict or the value format
  differs. It is never itself the answer: fix the condition, then answer.
* Same error twice → stop, inspect the schema, rewrite the statement from scratch, then
  submit. A third identical statement is never allowed.
