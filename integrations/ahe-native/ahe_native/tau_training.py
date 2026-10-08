"""Tau schedule validation only; AHE search/selection remain upstream."""
TAU_DOMAINS = ("airline", "retail", "telecom")


def make_tau_schedule(historical):
    return {"schema_version": 1, "historical_manifest_sha256": historical["sha256"],
        "split": "train", "iterations": 6, "batch_policy": "frozen_tau_native_batches",
        "domains": {d: {"trials": historical["domains"][d]["train_trials"],
            "batches": [{"iteration": i + 1, "task_ids": row["task_ids"]}
                        for i, row in enumerate(historical["domains"][d]["batches"])]}
            for d in TAU_DOMAINS}}


def scheduled_tau_batch(request, schedule, historical):
    domain = request["domain"]
    if domain not in TAU_DOMAINS or schedule["iterations"] != 6:
        raise ValueError("six native Tau TRAIN batches required")
    rows = schedule["domains"][domain]["batches"]
    frozen = historical["domains"][domain]["batches"]
    if len(rows) != 6 or schedule["domains"][domain]["trials"] != historical["domains"][domain]["train_trials"]:
        raise ValueError("Tau trial/batch budget changed")
    if request["k"] != schedule["domains"][domain]["trials"]:
        raise ValueError("Tau repeat count differs from frozen protocol")
    for number, (row, old) in enumerate(zip(rows, frozen), 1):
        ids = row["task_ids"]
        if (row["iteration"] != number or ids != old["task_ids"] or
                not ids or any(not isinstance(task, str) for task in ids) or len(ids) != len(set(ids))):
            raise ValueError("Tau tasks differ from frozen native TRAIN batch")
    iteration = request["iteration"]
    if type(iteration) is not int or not 1 <= iteration <= 6:
        raise ValueError("Tau iteration outside frozen budget")
    return list(rows[iteration - 1]["task_ids"])
