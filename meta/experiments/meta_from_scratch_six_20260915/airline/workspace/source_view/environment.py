# Copyright Sierra
from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.tools import AirlineTools
from tau2.domains.airline.utils import (
    AIRLINE_DB_PATH,
    AIRLINE_POLICY_PATH,
    AIRLINE_TASK_SET_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file


def get_environment(
    db: Optional[FlightDB] = None,
    solo_mode: bool = False,
    harness_enabled: bool = False,
    harness_h3: bool = False,
    harness_h4: bool = False,
) -> Environment:
    if solo_mode:
        raise ValueError("Airline domain does not support solo mode")
    if db is None:
        db = FlightDB.load(AIRLINE_DB_PATH)
    if harness_enabled and harness_h3 and harness_h4:
        from tau2.harness.airline import H3H4HarnessedAirlineTools

        tools = H3H4HarnessedAirlineTools(db)
    elif harness_enabled and harness_h4:
        from tau2.harness.airline import H4HarnessedAirlineTools

        tools = H4HarnessedAirlineTools(db)
    elif harness_h4:
        from tau2.harness.airline import H4AirlineTools

        tools = H4AirlineTools(db)
    elif harness_enabled and harness_h3:
        from tau2.harness.airline import H3HarnessedAirlineTools

        tools = H3HarnessedAirlineTools(db)
    elif harness_enabled:
        from tau2.harness.airline import HarnessedAirlineTools

        tools = HarnessedAirlineTools(db)
    elif harness_h3:
        from tau2.harness.airline import H3AirlineTools

        tools = H3AirlineTools(db)
    else:
        tools = AirlineTools(db)
    with open(AIRLINE_POLICY_PATH, "r") as fp:
        policy = fp.read()
    return Environment(
        domain_name="airline",
        policy=policy,
        tools=tools,
    )


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    tasks = load_file(AIRLINE_TASK_SET_PATH)
    tasks = [Task.model_validate(task) for task in tasks]
    if task_split_name is None:
        return tasks
    task_splits = get_tasks_split()
    if task_split_name not in task_splits:
        raise ValueError(
            f"Invalid task split name: {task_split_name}. Valid splits are: {task_splits.keys()}"
        )
    return [task for task in tasks if task.id in task_splits[task_split_name]]


def get_tasks_split() -> dict[str, list[str]]:
    split_file = (
        Path(AIRLINE_TASK_SET_PATH).parent
        / f"split_{Path(AIRLINE_TASK_SET_PATH).stem}.json"
    )
    return load_file(split_file)
