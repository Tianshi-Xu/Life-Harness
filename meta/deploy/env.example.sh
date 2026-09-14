# Copy to env.sh, fill in local values, then source it before an iteration.
# env.sh is ignored by git.
export AGENT_API_BASE=http://127.0.0.1:8400/v1
export AGENT_API_KEY=EMPTY
export USER_API_BASE=https://your-user-simulator.example.com/v1
export OPENAI_API_KEY=replace-me

# Pass the corresponding public model names to the loop commands.
export LIFE_AGENT_MODEL=openai/qwen3-4b
export LIFE_USER_MODEL=openai/your-user-simulator
