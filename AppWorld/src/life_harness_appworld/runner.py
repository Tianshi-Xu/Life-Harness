from __future__ import annotations

import argparse
import os
from copy import deepcopy
from pathlib import Path

from appworld import load_task_ids, update_root
from appworld.cli import extract_runner_config

from life_harness_appworld.agent import LifeHarnessFunctionCallingAgent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--appworld-root", type=Path, required=True)
    parser.add_argument("--policy-directory", type=Path, required=True)
    parser.add_argument("--dataset", default="train")
    parser.add_argument("--task-id")
    parser.add_argument(
        "--task-ids-file",
        type=Path,
        help="newline-delimited task IDs for a targeted regression subset",
    )
    parser.add_argument(
        "--base-experiment",
        default="simplified_function_calling_agent/local/qwen3-4b/train",
    )
    parser.add_argument("--model-name", default="Qwen3-4B")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--api-predictor-temperature",
        type=float,
        default=None,
        help=(
            "optional routing-only temperature; defaults to the main-agent "
            "temperature when omitted"
        ),
    )
    parser.add_argument(
        "--experiment-name", default="life_harness/qwen3-4b/v005/train"
    )
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--process-index", type=int, default=0)
    parser.add_argument("--baseline", action="store_true",
                        help="Disable all harness layers for baseline comparison")
    parser.add_argument("--h5-top-k", type=int, default=1)
    parser.add_argument(
        "--api-predictor-max-completion-tokens",
        type=int,
        default=3000,
        help=(
            "routing-only predictor budget; set to 0 to omit it. The main agent "
            "always uses vLLM's remaining context without a completion cap."
        ),
    )
    parser.add_argument(
        "--qwen-disable-thinking",
        action="store_true",
        help=(
            "run an explicit Qwen3 non-thinking ablation through vLLM's template switch. "
            "The default preserves Qwen's native thinking behavior and never adds an output cap."
        ),
    )
    parser.add_argument("--disable-h2", action="store_true")
    parser.add_argument("--disable-h3", action="store_true")
    parser.add_argument("--disable-h4", action="store_true")
    parser.add_argument("--disable-h5", action="store_true")
    args = parser.parse_args()

    root = str(args.appworld_root.resolve())
    os.environ["APPWORLD_ROOT"] = root
    os.environ.setdefault("MODEL_SERVER_URL", "http://127.0.0.1:8000")
    os.environ.setdefault("NO_API_KEY", "EMPTY")
    os.environ.setdefault("OPENAI_API_KEY", os.environ["NO_API_KEY"])
    update_root(root)

    runner_config = extract_runner_config(args.base_experiment)
    agent_config = deepcopy(runner_config["agent"])
    agent_config.pop("type", None)
    main_model_config = agent_config.get("model_config", {})
    predictor_model_config = agent_config.get("api_predictor_config", {}).get(
        "model_config", {}
    )
    main_model_config["name"] = args.model_name
    main_model_config["temperature"] = args.temperature
    predictor_model_config["name"] = args.model_name
    predictor_model_config["temperature"] = (
        args.temperature
        if args.api_predictor_temperature is None
        else args.api_predictor_temperature
    )
    # The upstream local Qwen config chooses 3,000 solely as a speed guard.
    # Do not send a per-request completion cap: vLLM will use the remaining
    # max-model-len after prompt tokens, which avoids artificial context errors.
    agent_config.get("model_config", {}).pop("max_completion_tokens", None)
    agent_config.get("model_config", {}).pop("max_tokens", None)
    agent_config.get("api_predictor_config", {}).get("model_config", {}).pop(
        "max_completion_tokens", None
    )
    agent_config.get("api_predictor_config", {}).get("model_config", {}).pop(
        "max_tokens", None
    )
    if args.api_predictor_max_completion_tokens > 0:
        predictor_model_config["max_completion_tokens"] = (
            args.api_predictor_max_completion_tokens
        )
    if args.qwen_disable_thinking:
        # Optional H5 ablation for Qwen3's model-native template switch. It is
        # deliberately opt-in after regression showed that default-on disabling
        # can suppress useful task planning. It is never a max_tokens limit.
        for model_config in (
            main_model_config,
            predictor_model_config,
        ):
            existing_extra_body = model_config.get("extra_body", {})
            extra_body = (
                deepcopy(existing_extra_body)
                if isinstance(existing_extra_body, dict)
                else {}
            )
            existing_template_kwargs = extra_body.get("chat_template_kwargs", {})
            template_kwargs = (
                deepcopy(existing_template_kwargs)
                if isinstance(existing_template_kwargs, dict)
                else {}
            )
            template_kwargs["enable_thinking"] = False
            extra_body["chat_template_kwargs"] = template_kwargs
            model_config["extra_body"] = extra_body
    agent = LifeHarnessFunctionCallingAgent(
        harness_policy_directory=str(args.policy_directory.resolve()),
        harness_h2=not args.disable_h2 and not args.baseline,
        harness_h3=not args.disable_h3 and not args.baseline,
        harness_h4=not args.disable_h4 and not args.baseline,
        harness_h5=not args.disable_h5 and not args.baseline,
        harness_h5_top_k=args.h5_top_k,
        baseline=args.baseline,
        **agent_config,
    )
    if args.task_id:
        task_ids = [args.task_id]
    elif args.task_ids_file:
        task_ids = [
            line.strip()
            for line in args.task_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        task_ids = load_task_ids(args.dataset)
    agent.solve_tasks(
        task_ids=task_ids,
        experiment_name=args.experiment_name,
        num_processes=args.num_processes,
        process_index=args.process_index,
    )


if __name__ == "__main__":
    main()
