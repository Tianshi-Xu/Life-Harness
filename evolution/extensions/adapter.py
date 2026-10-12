"""The trusted host adapter owns the solver, environment and native grader."""
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

RequestRecorder = Callable[[str, Callable[[], dict]], dict]


@dataclass
class EpisodeSetup:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    public_context: dict[str, Any] | None = None
    max_turns: int = 20


@dataclass
class ModelResponse:
    message: dict[str, Any]
    usage: dict[str, Any] | None = None
    # Only a deterministic, non-API solver should use local_no_api.
    usage_source: str = "provider"


@dataclass
class StepResult:
    messages: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False


@runtime_checkable
class BenchmarkEpisode(Protocol):
    setup: EpisodeSetup

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        """One solver request; return the unmodified assistant message and usage."""
        ...

    def execute(self, message: dict) -> StepResult:
        """Dispatch the assistant action through the original environment."""
        ...

    def grade(self) -> bool:
        """Native binary success; grader state is never supplied to the hooks."""
        ...

    def close(self) -> None: ...


@runtime_checkable
class BenchmarkAdapter(Protocol):
    name: str
    public_context_fields: Sequence[str]

    def task_ids(self, split: str) -> Sequence[str]:
        """Stable, unique IDs; TRAIN and TEST must identify disjoint tasks."""
        ...

    def open_episode(self, task_id: str, split: str) -> BenchmarkEpisode:
        """Create a fresh native episode and return its public setup."""
        ...

    def fingerprint(self) -> dict:
        """Stable JSON: data/version, solver configuration and native budgets; no keys."""
        ...
