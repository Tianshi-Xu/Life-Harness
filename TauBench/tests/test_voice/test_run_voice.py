from copy import deepcopy

import pytest

from tau2.config import (
    DEFAULT_LLM_AGENT,
    DEFAULT_LLM_ARGS_AGENT,
    DEFAULT_LLM_ARGS_USER,
    DEFAULT_LLM_USER,
)
from tau2.data_model.simulation import TextRunConfig
from tau2.data_model.tasks import Task, make_task
from tau2.data_model.voice import SynthesisConfig, TranscriptionConfig, VoiceSettings
from tau2.run import EvaluationType, run_task, run_tasks


@pytest.fixture
def transcription_voice_settings() -> VoiceSettings:
    return VoiceSettings(
        synthesis_config=None,
        transcription_config=TranscriptionConfig(),
    )


@pytest.fixture
def synthesis_voice_settings() -> VoiceSettings:
    return VoiceSettings(
        synthesis_config=SynthesisConfig(),
        transcription_config=None,
    )


@pytest.fixture
def run_config_voice(
    transcription_voice_settings: VoiceSettings, synthesis_voice_settings: VoiceSettings
) -> TextRunConfig:
    """Legacy text-agent configuration with voice settings.

    Voice settings are not part of TextRunConfig; full voice simulations
    should use VoiceRunConfig instead.
    """
    return TextRunConfig(
        domain="mock",
        agent="llm_agent",
        user="user_simulator",
        task_ids=["create_task_1"],
        llm_agent="gpt-3.5-turbo",
        llm_args_agent={},
        llm_user="gpt-3.5-turbo",
        llm_args_user={},
        num_trials=3,
        max_steps=20,
        max_errors=10,
        save_to=None,
        max_concurrency=3,
    )


@pytest.mark.skip(
    reason="Voice agent API not yet integrated into run_tasks/run_task. Use audio_native_config instead."
)
def test_simplified_voice_run(
    domain_name: str,
    transcription_voice_settings: VoiceSettings,
    synthesis_voice_settings: VoiceSettings,
):
    """Test that we can run a task with the mock domain and voice settings"""

    def run_simple_task(user_instruction: str, domain_name: str):
        task = make_task(
            user_instructions=user_instruction,
            eval_criteria=None,
            initialization_data=None,
            message_history=None,
        )
        simulation = run_tasks(
            domain=domain_name,
            tasks=[task],
            agent="voice_llm_agent",
            user="voice_user_simulator",
            llm_agent=DEFAULT_LLM_AGENT,
            llm_args_agent=deepcopy(DEFAULT_LLM_ARGS_AGENT),
            llm_user=DEFAULT_LLM_USER,
            llm_args_user=deepcopy(DEFAULT_LLM_ARGS_USER),
            max_steps=5,
            max_errors=5,
            evaluation_type=EvaluationType.ENV,
            console_display=False,
            max_concurrency=1,
            agent_voice_settings=transcription_voice_settings,
            user_voice_settings=synthesis_voice_settings,
        )
        return simulation

    simulation = run_simple_task(
        user_instruction="create a task called 'test' for user_1",
        domain_name=domain_name,
    )
    assert simulation is not None


@pytest.mark.skip(
    reason="Voice agent API not yet integrated into run_tasks/run_task. Use audio_native_config instead."
)
def test_run_tasks_base_voice(
    domain_name: str,
    base_task: Task,
    transcription_voice_settings: VoiceSettings,
    synthesis_voice_settings: VoiceSettings,
):
    """Test running a task with the mock domain and voice settings"""
    results = run_tasks(
        domain=domain_name,
        tasks=[base_task],
        agent="voice_llm_agent",
        user="voice_user_simulator",
        llm_agent="gpt-3.5-turbo",
        llm_args_agent={},
        llm_user="gpt-3.5-turbo",
        llm_args_user={},
        max_concurrency=1,
        agent_voice_settings=transcription_voice_settings,
        user_voice_settings=synthesis_voice_settings,
    )
    # Check that simulation ran and has expected structure
    assert len(results.simulations) == 1
    simulation = results.simulations[0]
    assert simulation.messages is not None
    assert len(simulation.messages) > 0
    assert simulation.start_time is not None
    assert simulation.end_time is not None
    assert simulation.reward_info.reward is not None


@pytest.mark.skip(
    reason="Voice agent API not yet integrated into run_tasks/run_task. Use audio_native_config instead."
)
def test_run_task_base_voice(
    domain_name: str,
    base_task: Task,
    transcription_voice_settings: VoiceSettings,
    synthesis_voice_settings: VoiceSettings,
):
    """Test running a task with the mock domain"""
    simulation = run_task(
        domain=domain_name,
        task=base_task,
        agent="voice_llm_agent",
        user="voice_user_simulator",
        llm_agent="gpt-3.5-turbo",
        llm_args_agent={},
        llm_user="gpt-3.5-turbo",
        llm_args_user={},
        evaluation_type=EvaluationType.ENV,
        agent_voice_settings=transcription_voice_settings,
        user_voice_settings=synthesis_voice_settings,
    )
    # Check that simulation ran and has expected structure
    assert simulation.messages is not None
    assert len(simulation.messages) > 0
    assert simulation.start_time is not None
    assert simulation.end_time is not None
    assert simulation.reward_info.reward is not None
