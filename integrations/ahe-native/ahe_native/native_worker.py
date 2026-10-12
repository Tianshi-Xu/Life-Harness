"""Unmodified native Task/environment execution in each benchmark's interpreter."""
from pathlib import Path
from types import SimpleNamespace
import argparse, asyncio, copy, json, os, sys, time, traceback

ROOT = Path(__file__).resolve().parents[3]
AB = ROOT / "AgentBench"
HISTORICAL = ROOT / "evolution/reproduction/reference"
TASK = None
DOMAIN = None
SPLIT = None

def initialize(domain, split="train"):
 global TASK, DOMAIN, SPLIT
 DOMAIN=domain;SPLIT=split;os.chdir(AB)
 if domain=='os':
  sys.path.insert(0,str(ROOT))
  from evolution.os_comparison.native import task
  TASK,_=task(split)
  assert not TASK.harness_config.enabled
  return
 if domain=='webshop':
  ws=AB/'.native/webshop-src';sys.path.insert(0,str(ws))
 snap=HISTORICAL/'task_snapshot';sys.path.insert(0,str(AB));sys.path.insert(0,str(snap))
 import yaml
 cfg=yaml.safe_load((snap/f'configs/tasks/{domain}.yaml').read_text())['default']['parameters'];cfg.update(name=f'{domain}-ahe-{split}',concurrency=1)
 if domain=='alfworld':
  from src.server.tasks.alfworld.task import ALFWorld
  cfg.update(data_path=str(AB/'data/alfworld'),config_path=str(snap/'src/server/tasks/alfworld/configs/base_config.yaml'),prompts_path=str(snap/'src/server/tasks/alfworld/prompts/alfworld_multiturn_plan_first.json'),split='train_valid' if split=='train' else 'new_std',enabled=False,h2=False,h3=False,h4=False,h5=False)
  TASK=ALFWorld(**cfg)
 elif domain=='dbbench':
  from src.server.tasks.dbbench.task import DBBenchTask
  cfg.update(data_file=str(AB/f"data/dbbench/{'db_out_new.jsonl' if split=='train' else 'standard.jsonl'}"),env_driver='native_mysql',env_options={'host':'127.0.0.1','port':13306},db_password='',harness={'enabled':False,'h2':False,'h3':False,'h4':False,'h5':False})
  TASK=DBBenchTask(**cfg)
 else:
  from src.server.tasks.webshop.task import WebShop
  cfg.update(start=0,end=1021,enabled=False,h2=False,h3=False,h4=False,h5=False)
  TASK=WebShop(**cfg)
 assert not TASK.harness_config.enabled


def describe(domain):
    if domain in ("dbbench", "webshop", "alfworld", "os"):
        initialize(domain)
        return {"domain": domain, "tools": TASK.tools}
    from tau2.runner.build import build_environment
    env = build_environment(domain)
    return {"domain": domain, "tools": [tool.openai_schema for tool in env.get_tools()]}


def run_agentbench(request):
    from ahe_native.session import NativeSession
    initialize(request["domain"], request["split"])
    session = NativeSession(request["workspace"], request["trace_dir"], TASK.tools,
                            request.get("agent_config", "code_agent.yaml"),
                            capability_policy=request.get("capability_policy"))
    start = time.monotonic()
    try:
        index = int(request["task_id"])
        if DOMAIN in {"dbbench", "os"}:
            async def run():
                try:
                    return await TASK.start_sample(index, session)
                finally:
                    if DOMAIN == "os":
                        from evolution.os_comparison.native import close_controller
                        await close_controller(TASK)
                    elif TASK.env_controller_background_task:
                        TASK.env_controller_background_task.cancel()
                        TASK.env_controller_background_task = None
            result = asyncio.run(run())
        else:
            result = TASK.sync_start_sample(index, session)
        status = result.status.value
        private = result.result or {}
        if DOMAIN == "dbbench":
            success = bool(private.get("is_correct", False))
        elif DOMAIN in {"alfworld", "os"}:
            success = private.get("result") == 1
        else:
            success = float(private.get("reward", 0) or 0) >= 1
        infra = status.replace(" ", "_") in ("task_error", "cancelled", "unknown", "running")
        row = {"success": int(success and not infra), "status": status,
               "infrastructure_error": infra}
    except Exception as exc:
        traceback.print_exc()
        row = {"success": 0, "status": type(exc).__name__, "infrastructure_error": True}
    finally:
        session.close()
    row.update(task_id=str(request["task_id"]), trial=request["trial"],
               messages=session.history or [{"role": "user", "content": "Episode initialization failed."}],
               wall_seconds=time.monotonic() - start)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--describe", choices=["dbbench","webshop","alfworld","os","airline","retail","telecom"])
    parser.add_argument("--request", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from ahe_native.records import write_json
    if args.describe:
        value = describe(args.describe)
    else:
        request = json.loads(args.request.read_text())
        if request["domain"] in ("dbbench","webshop","alfworld","os"):
            value = run_agentbench(request)
        else:
            from ahe_native.tau_bridge import run_tau
            value = run_tau(request)
    write_json(args.out, value)


if __name__ == "__main__":
    main()
