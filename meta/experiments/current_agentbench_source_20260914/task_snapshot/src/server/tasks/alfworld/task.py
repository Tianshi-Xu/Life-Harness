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
from types import SimpleNamespace
from src.server.harness.alfworld import Harness
from src.server.harness.session import FourHookSession



class ALFWorld(Task):

    def __init__(self,
                 data_path: Optional[str],
                 config_path: Optional[str],
                 prompts_path: Optional[str],
                 split: str = 'dev',
                 max_step: int = 20,
                 tools: Optional[List[Dict[str, Any]]] = None,
                 **kwargs):
        enabled = kwargs.pop("enabled", True)
        self.harness_config = SimpleNamespace(
            enabled=bool(enabled),
            **{f"{layer}_enabled": bool(kwargs.pop(layer, True)) for layer in ("h2", "h3", "h4", "h5")},
        )
        kwargs.pop("h5_top_k", None)  # The published candidate owns its skill policy.
        self.base_tools = deepcopy(tools)
        if self.harness_config.enabled and self.harness_config.h3_enabled:
            tools = Harness().h3(deepcopy(tools))
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

        # Support deterministic shuffle with a fixed seed
        shuffle_seed = kwargs.get("shuffle_seed", None)
        if shuffle_seed is not None:
            import random
            random.Random(shuffle_seed).shuffle(self.data_files)
        
        # Support selecting a task slice
        start_idx = kwargs.get("start", 0)
        end_idx = kwargs.get("end", len(self.data_files))
        self.data_files = self.data_files[start_idx:end_idx]

        # Support running a fixed-size sample
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
        ob, info = self.env.reset_env(env)
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        log_info = {'log': [], 'harness_trace': {'h2': [], 'h3': [], 'h4': [], 'h5': []}}
        if self.harness_config.enabled:
            session = FourHookSession(session, Harness(), self.base_tools, self.max_step, self.harness_config)
            log_info['harness_trace'] = session.trace
        initial_admissible = info.get('admissible_commands', [[]])[0]
        init_prompt = 'Here is your task. ' + ob + self.get_available_actions(initial_admissible)
        log_info['init_prompt'] = init_prompt
        system_prompt = self.get_task_instruction()
        session.inject(ChatCompletionSystemMessageParam(role='system', content=system_prompt))
        session.inject(ChatCompletionUserMessageParam(role='user', content=init_prompt))
        _last_admissible: List[str] = initial_admissible
        _no_tool_consecutive: int = 0
        for i in range(0, self.max_step):
            output = session.sync_action()
            tool_calls = []
            for message in output.messages:
                tool_calls.extend(message.get('tool_calls', []) or [])
            if not tool_calls:
                _no_tool_consecutive += 1
                no_exec_msg = 'You MUST call the take_action tool — do NOT output plain text without a tool call.'
                session.inject(ChatCompletionUserMessageParam(role='user', content=no_exec_msg))
                session.inject(RewardHistoryItem(reward=0, score=0))
                continue
            _no_tool_consecutive = 0
            try:
                tool_call = tool_calls[0]
                arguments = tool_call['function']['arguments']
                arguments = json.loads(arguments)
                arguments = list(arguments.values())
                call_id = tool_call['id']
                admissible_commands = info.get('admissible_commands', [[]])[0]
                output = arguments[0]
                # H2 has already performed the legacy validator's exact-match /
                # canonicalization decision.  Running process_action a second
                # time can turn a deliberately allowed invalid command into a
                # different admissible command (for example desk 1 -> desk 2),
                # changing both the environment state and the evaluated policy.
                if getattr(session, 'h2_prevalidated_actions', False):
                    action = output
                else:
                    action = process_action(output, admissible_commands)
            except:
                session.inject(ChatCompletionUserMessageParam(role='user', content='No valid tool calls found. Please call a tool instead.'))
                session.inject(RewardHistoryItem(reward=0, score=0))
                continue
            observation, reward, done, info = self.env.step_env(env, action)
            observation, reward, done = (process_ob(observation[0]), info['won'][0], done[0])
            _last_admissible = info.get('admissible_commands', [[]])[0]
            session.inject(ChatCompletionToolMessageParam(role='tool', tool_call_id=call_id, content=observation + self.get_available_actions(_last_admissible)))
            round_reward = reward
            if 'Nothing happens' in observation:
                round_reward = 0
            session.inject(RewardHistoryItem(reward=round_reward, score=reward))
            payload = {'round': i + 1, 'output': output, 'action': action, 'admissible_commands': admissible_commands, 'observation': observation, 'done': done}
            log_info['log'].append(payload)
            if len(log_info['log']) > 3:
                pre_logs = log_info['log'][-3:]
                pre_acts = [pre_log['output'] for pre_log in pre_logs]
                if len(list(set(pre_acts))) == 1:
                    self.logger.info('repeat actions for 3 times: failure')
                    return (0, log_info, SampleStatus.AGENT_INVALID_ACTION)
            if done:
                return (reward, log_info, finish_reason)
        else:
            finish_reason = SampleStatus.TASK_LIMIT_REACHED
            final_reward = 0
            reward_history = RewardHistoryItem(reward=final_reward, score=0)
            session.inject(reward_history)
        return (0, log_info, finish_reason)
