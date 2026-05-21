import logging
import os
import traceback
from copy import deepcopy
from typing import Dict, Any, List, Optional
import json

from agentrl.worker.task import Task, Session
from agentrl.worker.typings import (AgentCancelledException,
                                    RewardHistoryItem,
                                    SampleStatus,
                                    TaskOutput,
                                    TaskSampleExecutionResult)
from openai.types.chat import (ChatCompletionSystemMessageParam,
                               ChatCompletionToolMessageParam,
                               ChatCompletionUserMessageParam)

from .environment import AlfworldEnvWrapper
from .utils import *
from src.server.harness import (
    ALFWorldHarnessConfig,
    ALFWorldHarnessRuntime,
    patch_take_action_tool_description,
)


class ALFWorld(Task):

    def __init__(self,
                 data_path: Optional[str],
                 config_path: Optional[str],
                 prompts_path: Optional[str],
                 split: str = 'dev',
                 max_step: int = 20,
                 tools: Optional[List[Dict[str, Any]]] = None,
                 **kwargs):
        enabled = kwargs.pop("enabled", False)
        h2 = kwargs.pop("h2", True)
        h3 = kwargs.pop("h3", True)
        h4 = kwargs.pop("h4", True)
        h5 = kwargs.pop("h5", True)
        h5_top_k = kwargs.pop("h5_top_k", 1)
        self.harness_config = ALFWorldHarnessConfig(
            enabled=bool(enabled),
            h2_enabled=bool(h2),
            h3_enabled=bool(h3),
            h4_enabled=bool(h4),
            h5_enabled=bool(h5),
            h5_top_k=int(h5_top_k),
        )
        if self.harness_config.enabled and self.harness_config.h3_enabled:
            # H3 hint is task-type-aware; a generic runtime is used here since
            # per-episode TaskContext is not yet available at class init time.
            # The per-episode runtime will apply a task-specific hint via init_task.
            runtime = ALFWorldHarnessRuntime(self.harness_config)
            tools = patch_take_action_tool_description(tools, runtime.build_h3_hint())
        super().__init__(tools=tools, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.tools = tools

        # load data_path
        self.data_path = data_path
        if self.data_path is None:
            raise Exception("missing parameter data_path")
        os.environ["ALFWORLD_DATA"] = self.data_path

        # load config for alfworld benchmark
        self.config_path = config_path
        if self.config_path is None:
            raise Exception("missing parameter config_path")
        self.config = load_config(self.config_path)

        # load prompts
        self.prompts_path = prompts_path
        if self.prompts_path is None:
            raise Exception("missing parameter prompts_path")
        self.prompts = load_prompts(self.prompts_path)

        # prepare data_files
        self.data_files = []
        self.split = split
        data_path = os.path.join("data/alfworld", f"{self.split}.json")
        with open(data_path, "r") as f:
            content = json.loads(f.read())
        for _, v in content.items():
            self.data_files.extend(v)
        self.data_files = [os.path.join(self.data_path, file) for file in self.data_files]

        # 支持 Deterministic Shuffle (设定种子)
        shuffle_seed = kwargs.get("shuffle_seed", None)
        if shuffle_seed is not None:
            import random
            random.Random(shuffle_seed).shuffle(self.data_files)
        
        # 支持分片取出部分任务
        start_idx = kwargs.get("start", 0)
        end_idx = kwargs.get("end", len(self.data_files))
        self.data_files = self.data_files[start_idx:end_idx]

        # 支持抽样运行固定数量任务
        sample_size = kwargs.get("sample_size", None)
        if sample_size is not None:
            self.data_files = self.data_files[:sample_size]

        self.logger.info(f"successfully loaded {len(self.data_files)} games")
        if len(self.data_files) > 0:
            self.logger.debug(f"{self.data_files[0]=}")

        # other configs
        self.max_step = max_step
        self.prefixes = {
            'pick_and_place': 'put',
            'pick_clean_then_place': 'clean',
            'pick_heat_then_place': 'heat',
            'pick_cool_then_place': 'cool',
            'look_at_obj': 'examine',
            'pick_two_obj': 'puttwo'
        }

        self.env = AlfworldEnvWrapper(self.config)

    def get_indices(self) -> List[Any]:
        return list(range(len(self.data_files)))

    def calculate_overall(self, results: List[TaskOutput]) -> Dict[str, Any]:
        """
            TaskOutput.result 0/1
        """
        def is_pass(config: TaskOutput) -> bool:
            if not config or not isinstance(config.result, dict):
                return False
            # Legacy path: explicit success marker.
            if "result" in config.result:
                return int(config.result.get("result", 0) == 1) == 1
            # New controller protocol path: reward/score carries success.
            reward = config.result.get("reward", None)
            if reward is not None:
                try:
                    return float(reward) >= 1.0
                except Exception:
                    return False
            metrics = config.result.get("metrics", {})
            score = metrics.get("score", None) if isinstance(metrics, dict) else None
            if score is not None:
                try:
                    return float(score) >= 1.0
                except Exception:
                    return False
            return False

        overall = {
            "total": len([config for config in results if config]),
            "pass": len([config for config in results if is_pass(config)]),
        }
        overall["wrong"] = overall["total"] - overall["pass"]
        overall["success_rate"] = overall["pass"] / overall["total"] if overall["total"] else 0
        return {
            "overall": overall,
        }

    def sync_start_sample(self, index, session: Session) -> TaskSampleExecutionResult:
        data_item = self.data_files[index]
        env = self.env.create_env(data_item)
        try:
            result, log_info, finish_reason = self.alfworld_run(session, env)
        except AgentCancelledException:
            return TaskSampleExecutionResult(status=SampleStatus.CANCELLED)
        except Exception:
            traceback.print_exc()
            return TaskSampleExecutionResult(status=SampleStatus.TASK_ERROR)
        finally:
            self.env.close_env(env)
        log_info.update({"result": result})
        return TaskSampleExecutionResult(status=finish_reason, result=log_info)

    @staticmethod
    def get_task_instruction():
        return """Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete) the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. A tool will be provided for you to use to submit the action you want to take. This tool is the only tool you should and must take in order to operate any action in the environment. The way you perform action is to place the action chosen by you in the arguments field of your tool call. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. The action you would like to take should be offered in this format: "the name of your next action", and you should fill it in the argument field of your tool call. Note that you should always call a tool to operate an action from the given choices. After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the environment output "Nothing happened", that means the previous action is invalid and you should try more options.
 Reminder:
1. the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal.
2. Always call the tool to hand in your next action and think when necessary."""

    def get_prompt(self, filename: str):
        # return []
        for k, v in self.prefixes.items():
            if filename.startswith(k):
                example = self.prompts[v]
                return deepcopy(example)
        raise Exception(f"unsupported name: {filename}")
        # return self.prompts["naive_example"]

    @staticmethod
    def get_available_actions(actions):
        actions = "\n".join(actions)
        return " AVAILABLE ACTIONS: " + actions + "\n"

    def alfworld_run(self, session: Session, env):
        finish_reason = SampleStatus.COMPLETED
        # env init
        ob, info = self.env.reset_env(env)
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        log_info = {
            "log": [],
            "harness_trace": {"h2": [], "h3": [], "h4": [], "h5": []},
        }
        harness_runtime = None
        if self.harness_config.enabled:
            harness_runtime = ALFWorldHarnessRuntime(self.harness_config)
        initial_admissible = info.get('admissible_commands', [[]])[0]
        init_prompt = "Here is your task. " + ob + self.get_available_actions(initial_admissible)
        log_info["init_prompt"] = init_prompt
        if harness_runtime:
            # H0: parse task context and seed world model
            harness_runtime.init_task(init_prompt, initial_admissible)

        # H5 cold-start: gather per-task tips before sending the first system prompt.
        cold_skills: list = []
        if harness_runtime and self.harness_config.h5_enabled:
            cold_skills = harness_runtime.cold_start_skill_hints()
            for item in cold_skills:
                log_info["harness_trace"]["h5"].append(item)

        system_prompt = self.get_task_instruction()
        if cold_skills:
            skill_lines = "\n".join(f"- {item['text']}" for item in cold_skills)
            system_prompt += f"\n\nSome tips that may help for this task:\n{skill_lines}"

        session.inject(ChatCompletionSystemMessageParam(
            role='system',
            content=system_prompt
        ))
        session.inject(ChatCompletionUserMessageParam(
            role='user',
            content=init_prompt
        ))

        # interact
        # Cache admissible commands from the previous env step so they are
        # available even on turns where the agent makes no tool call.
        _last_admissible: List[str] = initial_admissible
        # Count consecutive turns with no tool call so we can escalate hints.
        _no_tool_consecutive: int = 0

        for i in range(0, self.max_step):
            output = session.sync_action()

            tool_calls = []
            for message in output.messages:
                tool_calls.extend(message.get('tool_calls', []) or [])

            if not tool_calls:
                _no_tool_consecutive += 1
                # Do not set finish_reason here — this is a mid-episode event,
                # not a terminal state. The episode continues via `continue`.
                no_exec_msg = (
                    'You MUST call the take_action tool — '
                    'do NOT output plain text without a tool call.'
                )
                # After 2+ consecutive no-tool turns, inject a directed step hint
                # so the agent has a concrete action to take rather than narrating.
                if (
                    _no_tool_consecutive >= 2
                    and harness_runtime
                    and self.harness_config.h4_enabled
                ):
                    forced_hint = harness_runtime.step_guidance(
                        current_round=i + 1,
                        max_step=self.max_step,
                        admissible=_last_admissible,
                    )
                    if forced_hint:
                        no_exec_msg = no_exec_msg + '\n' + forced_hint
                session.inject(ChatCompletionUserMessageParam(
                    role='user',
                    content=no_exec_msg,
                ))
                session.inject(RewardHistoryItem(reward=0, score=0))
                continue

            _no_tool_consecutive = 0

            try:
                tool_call = tool_calls[0]
                arguments = tool_call["function"]["arguments"]
                arguments = json.loads(arguments)
                arguments = list(arguments.values())
                call_id = tool_call["id"]
                # process action
                admissible_commands = info.get('admissible_commands', [[]])[0]
                output = arguments[0]
                if harness_runtime and self.harness_config.h2_enabled:
                    h2 = harness_runtime.pre_validate_action(output, admissible_commands)
                    action = h2["action"]
                    log_info["harness_trace"]["h2"].append(
                        {
                            "round": i + 1,
                            "canonicalized": h2["canonicalized"],
                            "blocked": h2["blocked"],
                            "reason": h2["reason"],
                            "raw_action": h2["raw_action"],
                            "final_action": action,
                        }
                    )
                    if h2["blocked"]:
                        # Do not set finish_reason — mid-episode block, episode continues.
                        session.inject(ChatCompletionUserMessageParam(
                            role='user',
                            content=(
                                "Harness blocked an invalid action sequence. "
                                "Please choose a valid action from AVAILABLE ACTIONS."
                            ),
                        ))
                        session.inject(RewardHistoryItem(reward=0, score=0))
                        continue
                else:
                    action = process_action(output, admissible_commands)
            except:
                # Do not set finish_reason — mid-episode exception, episode continues.
                session.inject(ChatCompletionUserMessageParam(
                    role='user',
                    content='No valid tool calls found. Please call a tool instead.'
                ))
                session.inject(RewardHistoryItem(reward=0, score=0))
                continue

            observation, reward, done, info = self.env.step_env(env, action)
            observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
            _last_admissible = info.get('admissible_commands', [[]])[0]
            session.inject(ChatCompletionToolMessageParam(
                role='tool',
                tool_call_id=call_id,
                content=observation + self.get_available_actions(_last_admissible)
            ))
            round_reward = reward
            if "Nothing happens" in observation:
                round_reward = 0
            session.inject(RewardHistoryItem(reward=round_reward, score=reward))

            # save
            payload = {
                "round": i + 1,
                "output": output,
                "action": action,
                "admissible_commands": admissible_commands,
                "observation": observation,
                "done": done,
            }
            log_info["log"].append(payload)

            post_admissible = info.get('admissible_commands', [[]])[0]
            if harness_runtime and self.harness_config.h4_enabled:
                h4 = harness_runtime.post_step_monitor(
                    raw_output=output,
                    final_action=action,
                    observation=observation,
                    admissible=post_admissible,
                )
                log_info["harness_trace"]["h4"].append({"round": i + 1, **h4})
                if h4.get("recovery_prompt"):
                    session.inject(ChatCompletionUserMessageParam(
                        role='user',
                        content=h4["recovery_prompt"],
                    ))
                # H4-D: step-budget management
                budget = harness_runtime.budget_check(
                    remaining_steps=self.max_step - i - 1,
                    admissible=post_admissible,
                )
                if budget.get("force_action"):
                    harness_runtime.force_next_action = budget["force_action"]
                if budget.get("hint"):
                    session.inject(ChatCompletionUserMessageParam(
                        role='user',
                        content=budget["hint"],
                    ))
                if budget.get("force_action") or budget.get("hint"):
                    log_info["harness_trace"]["h4"].append({
                        "round": i + 1,
                        "sub": "h4d_budget",
                        "forced": budget.get("force_action"),
                        "hint": budget.get("hint"),
                    })
                # H4-E: state-driven per-step guidance (WorldModel + SubgoalSM)
                step_hint = harness_runtime.step_guidance(
                    current_round=i + 1,
                    max_step=self.max_step,
                    admissible=post_admissible,
                )
                if step_hint:
                    session.inject(ChatCompletionUserMessageParam(
                        role='user',
                        content=step_hint,
                    ))
                    log_info["harness_trace"]["h4"].append({
                        "round": i + 1,
                        "sub": "h4e_step_guidance",
                        "text": step_hint,
                        "token_cost": str(len(step_hint.split())),
                    })
            # failure test
            if len(log_info["log"]) > 3:
                pre_logs = log_info["log"][-3:]
                pre_acts = [pre_log["output"] for pre_log in pre_logs]
                if len(list(set(pre_acts))) == 1:
                    self.logger.info("repeat actions for 3 times: failure")
                    return 0, log_info, SampleStatus.AGENT_INVALID_ACTION

            if done:
                return reward, log_info, finish_reason
        else:
            finish_reason = SampleStatus.TASK_LIMIT_REACHED
            final_reward = 0
            reward_history = RewardHistoryItem(reward=final_reward, score=0)
            session.inject(reward_history)

        return 0, log_info, finish_reason
