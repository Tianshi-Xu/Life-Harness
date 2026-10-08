"""H5: cold-start answer contract for a DBBench task whose public table is bound; it names the answering column,
the quantity-column reading, the exact-cell answer format and the confirmation step for a modification."""

TARGET_LAYER = 'h5'

SKILL = ("Answer contract for this task: (1) name the single column that answers the question and read the value "
         "from that column; when the question asks how many people watched, attended or visited and the table has "
         "a column of such counts, the answer is that column's value for the matching row, not COUNT(*) of the "
         "matching rows; (2) commit the cell text exactly as the tool printed it, one answer string per returned "
         "row, with no added words such as 'from', 'in' or 'the answer is'; (3) a modifying statement returns no "
         "rows, so confirm its effect with one SELECT over the same filter and never run an identical modifying "
         "statement twice.")


class Harness:
    context = None

    def bind_task_context(self, context):
        self.context = context

    def h5(self, messages):
        if not isinstance(messages, list) or not messages:
            return []
        if not isinstance(self.context, dict):
            return []
        if not isinstance(self.context.get('table'), dict):
            return []
        return [SKILL]
