from datetime import datetime

from tau2.utils.utils import DATA_DIR

AIRLINE_DATA_DIR = DATA_DIR / "tau2" / "domains" / "airline"
AIRLINE_DB_PATH = AIRLINE_DATA_DIR / "db.json"
AIRLINE_POLICY_PATH = AIRLINE_DATA_DIR / "policy.md"
AIRLINE_TASK_SET_PATH = AIRLINE_DATA_DIR / "tasks.json"

# Simulated "current time" as stated in policy.md: 2024-05-15 15:00:00 EST.
# All time-based harness rules must use this constant instead of datetime.now().
SIMULATION_TIME: datetime = datetime(2024, 5, 15, 15, 0, 0)
