import json
import logging
import re
from typing import Dict, List, Any
from uuid import uuid4

from agentrl.worker.task import Task, Session
from agentrl.worker.typings import (AgentCancelledException,
                                    RewardHistoryItem,
                                    SampleStatus,
                                    TaskOutput,
                                    TaskSampleExecutionResult)
from openai.types.chat import (ChatCompletionSystemMessageParam,
                               ChatCompletionToolMessageParam,
                               ChatCompletionUserMessageParam)
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

from src.server.harness import (
    WebShopHarnessConfig,
    WebShopHarnessRuntime,
    patch_webshop_tool_descriptions,
)

prompt_with_max_turn = """You are a web shopping agent. Follow the task instruction to find and buy the correct product.

CRITICAL RULES:
1. You MUST call a tool EVERY turn. NEVER respond with only text — always call search_action or click_action.
2. On search results: read product titles carefully. Click the product whose title best matches ALL key terms in the instruction (brand name, product type, specific features).
3. On a product page: select ALL required attributes (color, size, etc.) THEN click 'buy now' immediately.
4. After selecting attributes, click 'buy now' right away. Do NOT hesitate or deliberate — just buy.
5. If the Hint says "All checked" or "All attributes selected", your ONLY correct action is click 'buy now'.
6. Keywords in search should be the product name and key features, NOT prices or filler words.
7. The click value MUST be exactly one of the available clickable values.
"""


def _extract_instruction(observation: str) -> str:
    """Parse the task instruction out of the initial WebShop observation.

    WebShop text-mode format: "WebShop [SEP] Instruction: [SEP] <text> [SEP] Search"
    """
    # Primary: [SEP]-delimited format used by WebShop text env
    m = re.search(r"Instruction:\s*\[SEP\]\s*(.+?)\s*\[SEP\]", observation, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: newline-delimited format
    m = re.search(r"Instruction:\s*\n(.+?)(?:\n\n|\[|$)", observation, re.DOTALL)
    if m:
        return m.group(1).strip()
    return observation[:300]


def _parse_available_actions(available_actions) -> tuple:
    """Return (has_search_bar: bool, clickables: list[str]) from env response."""
    if isinstance(available_actions, dict):
        return (
            bool(available_actions.get("has_search_bar", False)),
            list(available_actions.get("clickables", [])),
        )
    return False, []


class WebShop(Task):
    def __init__(self, tools=None, **configs):
        # Extract harness config params before passing configs to super()
        enabled = configs.pop("enabled", False)
        h2 = configs.pop("h2", True)
        h3 = configs.pop("h3", True)
        h4 = configs.pop("h4", True)
        h5 = configs.pop("h5", True)
        h5_top_k = configs.pop("h5_top_k", 2)
        h5_score_threshold = configs.pop("h5_score_threshold", 0.0)
        self.harness_config = WebShopHarnessConfig(
            enabled=bool(enabled),
            h2_enabled=bool(h2),
            h3_enabled=bool(h3),
            h4_enabled=bool(h4),
            h5_enabled=bool(h5),
            h5_top_k=int(h5_top_k),
            h5_score_threshold=float(h5_score_threshold),
        )
        if self.harness_config.enabled and self.harness_config.h3_enabled:
            tools = patch_webshop_tool_descriptions(tools)

        super().__init__(**configs)
        self.logger = logging.getLogger(__name__)
        self.ranging = (configs.pop("start", 0), configs.pop("end", 500))
        self.shuffle_seed = configs.pop("shuffle_seed", None)
        self.sample_size = configs.pop("sample_size", None)
        print(
            f"[MOUNT_CHECK] WebShop mapped source active: range={self.ranging}, sample_size={self.sample_size}",
            flush=True,
        )
        self.logger.warning(
            "[MOUNT_CHECK] WebShop mapped source loaded: start=%s end=%s sample_size=%s",
            self.ranging[0],
            self.ranging[1],
            self.sample_size,
        )
        self.logger.info('Initializing WebShop environment...')
        self.server = WebAgentTextEnv(observation_mode="text", human_goals=True).server
        self.tools = tools
        self.max_rounds = configs.get('round', 20)

    def get_indices(self) -> List[Any]:
        indices = list(range(*self.ranging))
        if self.shuffle_seed is not None:
            import random
            random.Random(self.shuffle_seed).shuffle(indices)
        if self.sample_size is not None:
            indices = indices[:self.sample_size]
        return indices

    def sync_start_sample(self, index: int, session: Session) -> TaskSampleExecutionResult:
        print(f"[MOUNT_CHECK][SAMPLE_START] webshop index={index}", flush=True)
        self.logger.warning("[MOUNT_CHECK][SAMPLE_START] webshop index=%s", index)
        history = []

        env = WebAgentTextEnv(
            observation_mode="text",
            server=self.server,
            human_goals=True,
            session_prefix=str(uuid4()) + '-'
        )
        try:
            env.reset(index)
            # Harness: initialise per-episode runtime
            harness_runtime = None
            cold_skills = []
            if self.harness_config.enabled:
                harness_runtime = WebShopHarnessRuntime(config=self.harness_config)
                harness_runtime.init_task(_extract_instruction(env.observation))

            # H5 cold-start: gather per-task tips before the first system message.
            if harness_runtime and self.harness_config.h5_enabled:
                cold_skills = harness_runtime.cold_start_skill_hints()

            system_prompt = prompt_with_max_turn
            if cold_skills:
                skill_lines = [f"- {item['text']}" for item in cold_skills]
                system_prompt += (
                    "\n\nSome tips that may help for this task:\n"
                    + "\n".join(skill_lines)
                )
            session.inject(ChatCompletionSystemMessageParam(
                role='system',
                content=system_prompt
            ))

            action = None
            observation = env.observation
            reward = 0
            call_id = None
            _no_tool_consecutive = 0

            for j in range(self.max_rounds):
                available_actions = env.get_available_actions()
                has_search_bar, clickables = _parse_available_actions(available_actions)

                if j == 0:
                    session.inject(ChatCompletionUserMessageParam(
                        role='user',
                        content=f'The initial observation:\n{observation}\n\nAvailable Actions:\n{available_actions}'
                    ))
                    # H4-E: initial search hint on step 0
                    if harness_runtime:
                        first_hint = harness_runtime.step_guidance(
                            step_num=0,
                            max_steps=self.max_rounds,
                            observation=observation,
                            has_search_bar=has_search_bar,
                            clickables=clickables,
                        )
                        if first_hint:
                            session.inject(ChatCompletionUserMessageParam(
                                role='user',
                                content=first_hint,
                            ))
                else:
                    if action is None:
                        session.inject(ChatCompletionUserMessageParam(
                            role='user',
                            content=f'Observation:\n{observation}\n\nAvailable Actions:\n{available_actions}'
                        ))
                    else:
                        session.inject(ChatCompletionToolMessageParam(
                            role='tool',
                            content=f'Action: {action}\n\nObservation:\n{observation}\n\nAvailable Actions:\n{available_actions}',
                            tool_call_id=call_id
                        ))

                response = session.sync_action()

                tool_calls = []
                for message in response.messages:
                    tool_calls.extend(message.get('tool_calls', []) or [])

                finish_reason = SampleStatus.COMPLETED
                if not tool_calls:
                    _no_tool_consecutive += 1
                    action = None

                    # Aggressive force: on product page with buy-now available,
                    # force buy-now after just 1 text-only turn (saves 265+ wasted turns).
                    buy_now_available = "buy now" in clickables
                    if harness_runtime and buy_now_available and _no_tool_consecutive >= 1:
                        # Force buy now immediately — the agent is deliberating uselessly
                        harness_runtime.force_next_action = "click[buy now]"
                        no_exec_msg = (
                            "You must call a tool! Since 'buy now' is available, "
                            "buying now. Call click_action with 'buy now' next turn."
                        )
                    elif harness_runtime and _no_tool_consecutive >= 2 and "back to search" in clickables:
                        harness_runtime.force_next_action = "click[back to search]"
                        no_exec_msg = (
                            "You must call a tool! Forcing back to search."
                        )
                    elif harness_runtime and _no_tool_consecutive >= 1:
                        # Immediate directive message
                        if has_search_bar:
                            no_exec_msg = (
                                "You must call a tool! Use search_action to search for the product."
                            )
                        elif "back to search" in clickables:
                            no_exec_msg = (
                                "You must call a tool! Click 'back to search' or click a product."
                            )
                        else:
                            no_exec_msg = (
                                "You must call a tool! Click one of the available options."
                            )
                    else:
                        no_exec_msg = "You must call a tool! NEVER respond with only text."
                    observation = no_exec_msg
                else:
                    _no_tool_consecutive = 0
                    action = None
                    try:
                        tool_call = tool_calls[0]
                        func_name = tool_call["function"]["name"]
                        arguments = tool_call["function"]["arguments"]
                        arguments = json.loads(arguments)
                        arguments = list(arguments.values())
                        call_id = tool_call["id"]

                        if harness_runtime and self.harness_config.h2_enabled:
                            raw_value = arguments[0]
                            h2_result = harness_runtime.pre_validate_action(
                                tool_name=func_name,
                                raw_value=raw_value,
                                has_search_bar=has_search_bar,
                                clickables=clickables,
                            )
                            if h2_result["blocked"]:
                                reason = h2_result["reason"]
                                # Use custom block_message if available (e.g. buy-now pre-check)
                                if "block_message" in h2_result:
                                    block_msg = h2_result["block_message"]
                                elif "search_not_available" in reason:
                                    block_msg = "Search is not available on this page. Click one of the available options."
                                elif "repeat_click" in reason:
                                    block_msg = f"You already clicked '{raw_value}'. Choose a different action."
                                else:
                                    block_msg = f"Invalid action: {reason}. Please choose a valid action from available options."
                                session.inject(ChatCompletionUserMessageParam(
                                    role='user',
                                    content=block_msg,
                                ))
                                session.inject(RewardHistoryItem(reward=0, score=0))
                                continue
                            action = h2_result["action"]
                        else:
                            if func_name == "search_action":
                                action = f"search[{arguments[0]}]"
                            elif func_name == "click_action":
                                action = f"click[{arguments[0]}]"
                    except:
                        self.logger.warning(f'Error processing tool call. {tool_calls=}', exc_info=True)
                        session.inject(ChatCompletionUserMessageParam(
                            role='user',
                            content=f"No valid tool call found from agent."
                        ))
                        session.inject(RewardHistoryItem(reward=0, score=0))
                        continue

                history.append(
                    {
                        "observation": observation,
                        "available_actions": available_actions,
                        "response": response,
                        "action": action,
                    }
                )

                if not action:
                    reward = 0
                    done = False
                    round_reward = 0
                else:
                    observation, reward, done, info = env.step(action)
                    round_reward = reward

                    if harness_runtime:
                        # Get post-step available actions for harness checks
                        post_available = env.get_available_actions()
                        post_has_sb, post_clickables = _parse_available_actions(post_available)

                        # H1: update page state
                        harness_runtime.update_state(action, observation, post_has_sb, post_clickables)

                        # H4: shopping monitor
                        h4_result = harness_runtime.post_step_monitor(
                            action, observation, post_has_sb, post_clickables
                        )
                        if h4_result.get("recovery_prompt"):
                            session.inject(ChatCompletionUserMessageParam(
                                role='user',
                                content=h4_result["recovery_prompt"],
                            ))

                        # H4-E: state-driven per-step guidance (page state + requirements)
                        h4e_hint = harness_runtime.step_guidance(
                            step_num=j + 1,
                            max_steps=self.max_rounds,
                            observation=observation,
                            has_search_bar=post_has_sb,
                            clickables=post_clickables,
                        )
                        if h4e_hint:
                            session.inject(ChatCompletionUserMessageParam(
                                role='user',
                                content=h4e_hint,
                            ))

                        # H4-D: step-budget management
                        budget_result = harness_runtime.budget_check(
                            remaining_steps=self.max_rounds - j - 1,
                            clickables=post_clickables,
                        )
                        if budget_result.get("force_action"):
                            harness_runtime.force_next_action = budget_result["force_action"]
                        if budget_result.get("hint"):
                            session.inject(ChatCompletionUserMessageParam(
                                role='user',
                                content=budget_result["hint"],
                            ))

                history[-1]["reward"] = reward
                history[-1]["done"] = done
                rewardhistory = RewardHistoryItem(reward=round_reward, score=round_reward)
                session.inject(rewardhistory)
                if done:
                    break
            else:
                finish_reason = SampleStatus.TASK_LIMIT_REACHED
                rewardhistory = RewardHistoryItem(reward=0, score=0)
                session.inject(rewardhistory)
                session.inject(ChatCompletionToolMessageParam(
                    role='tool',
                    content='Task limit reached.',
                    tool_call_id=call_id
                ))

            return TaskSampleExecutionResult(
                status=finish_reason,
                result={
                    "reward": reward,
                    "history": history,
                },
            )
        except AgentCancelledException:
            session.inject(RewardHistoryItem(reward=0, score=0))
            return TaskSampleExecutionResult(
                status=SampleStatus.CANCELLED,
                result={
                    "reward": 0,
                    "history": history,
                },
            )
        except:
            self.logger.exception(f'Error during sample execution')
            return TaskSampleExecutionResult(
                status=SampleStatus.TASK_ERROR,
                result={
                    "reward": 0,
                    "history": history,
                },
            )
        finally:
            try:
                env.close()
            except:
                pass

    def calculate_overall(self, results: List[TaskOutput]) -> Dict:
        result_payloads = [x.result for x in results if x and isinstance(x.result, dict)]
        rewards = [x.get("reward") for x in result_payloads if x.get("reward") is not None]
        total = len(results)
        completed = len([x for x in results if x and x.status == SampleStatus.COMPLETED])
        success_at_1 = len([r for r in rewards if float(r) >= 1.0])
        average_reward = sum(rewards) / len(rewards) if rewards else 0
        usages = [x.get("token_usage", {}) for x in result_payloads]
        n_ep = len(usages) or 1
        total_prompt = sum(u.get("prompt_tokens", 0) for u in usages if u)
        total_completion = sum(u.get("completion_tokens", 0) for u in usages if u)
        total_tokens = sum(u.get("total_tokens", 0) for u in usages if u)
        return {
            "overall": {
                "total": total,
                "completed": completed,
                "completed_rate": completed / total if total else 0,
                "success_at_1": success_at_1,
                "success_at_1_rate": success_at_1 / total if total else 0,
                "average_reward": average_reward,
            },
            "token_usage": {
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "total_tokens": total_tokens,
                "avg_prompt_tokens_per_episode": round(total_prompt / n_ep),
                "avg_completion_tokens_per_episode": round(total_completion / n_ep),
                "avg_total_tokens_per_episode": round(total_tokens / n_ep),
            },
        }
