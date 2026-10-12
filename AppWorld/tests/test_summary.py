import json
from pathlib import Path
from tempfile import TemporaryDirectory

from life_harness_appworld.analysis.summarize_experiment import summarize


def test_final_dev_scores_and_unknown_usage_are_preserved():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        logs = root / "tasks/task_one/logs"
        logs.mkdir(parents=True)
        (root / "evaluations").mkdir()
        final = {"accuracy": 0.5, "split": "dev"}
        (root / "evaluations/dev.json").write_text(json.dumps(final))
        calls = [{"output": {"usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}}},
                 {"output": {"error": "timeout"}}]
        (logs / "lm_calls.jsonl").write_text("\n".join(json.dumps(call) for call in calls))
        report = summarize(root, "dev")
        assert report["evaluation"] == final
        usage = report["model_token_usage"]
        assert usage["calls"] == 2
        assert usage["known_total_tokens"] == 13
        assert usage["calls_missing_usage"] == 1
        assert not usage["accounting_complete"]


def test_total_derived_only_from_two_measured_fields():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        logs = root / "tasks/task_one/logs"
        logs.mkdir(parents=True)
        (logs / "lm_calls.jsonl").write_text(json.dumps({"output": {
            "usage": {"prompt_tokens": 5, "completion_tokens": 2}}}))
        usage = summarize(root)["model_token_usage"]
        assert usage["known_total_tokens"] == 7
        assert usage["total_tokens_derived_calls"] == 1
        assert usage["accounting_complete"]


def test_task_with_no_call_log_cannot_claim_complete_accounting():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "tasks/task_one/logs").mkdir(parents=True)
        usage = summarize(root)["model_token_usage"]
        assert usage["tasks_missing_call_logs"] == 1
        assert not usage["accounting_complete"]
