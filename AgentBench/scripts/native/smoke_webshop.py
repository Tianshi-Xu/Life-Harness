"""Load the exact 100k WebShop environment and execute one real search."""

import json

from web_agent_site.engine import engine

# Keep the smoke output small while preserving the environment's load path.
engine.tqdm = lambda values, total=None: values

from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv


env = WebAgentTextEnv(
    observation_mode="text",
    human_goals=True,
    num_products=100_000,
)
try:
    env.reset(0)
    initial = env.observation
    goal = env.server.goals[0]
    observation, reward, done, _ = env.step(f"search[{goal['query']}]")
    actions = env.get_available_actions()
    assert len(env.server.all_products) == 100_000
    assert len(env.server.goals) == 1_021
    assert initial and observation
    assert isinstance(actions, dict) and actions.get("clickables")
    print(
        json.dumps(
            {
                "products": len(env.server.all_products),
                "goals": len(env.server.goals),
                "initial_observation_chars": len(initial),
                "search_observation_chars": len(observation),
                "search_clickables": len(actions["clickables"]),
                "reward": reward,
                "done": done,
            },
            indent=2,
        )
    )
finally:
    env.close()
