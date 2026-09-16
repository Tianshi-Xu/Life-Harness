import pytest
import tau2.utils.llm_utils as llm_utils

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool, as_tool
from tau2.utils.llm_utils import generate


def test_user_simulator_pacing_does_not_throttle_agent_calls(monkeypatch):
    monkeypatch.setenv("TAU_USER_LLM_MIN_INTERVAL_SECONDS", "3.75")
    times = iter([10.0, 10.0])
    sleeps = []
    monkeypatch.setattr(llm_utils.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(llm_utils.time, "sleep", sleeps.append)
    monkeypatch.setattr(llm_utils, "_user_llm_next_request_at", 0.0)

    llm_utils._pace_user_simulator("user_simulator_response")
    llm_utils._pace_user_simulator("agent_response")
    llm_utils._pace_user_simulator("user_simulator_response")

    assert sleeps == [3.75]
    assert llm_utils._user_llm_next_request_at == 17.5


@pytest.fixture
def model() -> str:
    return "gpt-4o-mini"


@pytest.fixture
def messages() -> list[Message]:
    messages = [
        SystemMessage(role="system", content="You are a helpful assistant."),
        UserMessage(role="user", content="What is the capital of the moon?"),
    ]
    return messages


@pytest.fixture
def tool() -> Tool:
    def calculate_square(x: int) -> int:
        """Calculate the square of a number.
            Args:
            x (int): The number to calculate the square of.
        Returns:
            int: The square of the number.
        """
        return x * x

    return as_tool(calculate_square)


@pytest.fixture
def tool_call_messages() -> list[Message]:
    messages = [
        SystemMessage(role="system", content="You are a helpful assistant."),
        UserMessage(
            role="user",
            content="What is the square of 5? Just give me the number, no explanation.",
        ),
    ]
    return messages


def test_generate_no_tool_call(model: str, messages: list[Message]):
    response = generate(model, messages)
    assert isinstance(response, AssistantMessage)
    assert response.content is not None


def test_generate_tool_call(model: str, tool_call_messages: list[Message], tool: Tool):
    response = generate(model, tool_call_messages, tools=[tool])
    assert isinstance(response, AssistantMessage)
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "calculate_square"
    assert response.tool_calls[0].arguments == {"x": 5}
    follow_up_messages = [
        response,
        ToolMessage(role="tool", id=response.tool_calls[0].id, content="25"),
    ]
    response = generate(
        model,
        tool_call_messages + follow_up_messages,
        tools=[tool],
    )
    assert isinstance(response, AssistantMessage)
    assert response.tool_calls is None
    assert response.content == "25"
