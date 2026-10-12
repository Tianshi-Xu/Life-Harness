# Copy to env.sh, fill in local values, then source it before an iteration.
# env.sh is ignored by git.
export AGENT_API_BASE=http://127.0.0.1:8400/v1
export AGENT_API_KEY=EMPTY
export USER_API_BASE=https://your-user-simulator.example.com/v1
export OPENAI_API_KEY=replace-me

# Pass the corresponding public model names to the loop commands.
export LIFE_AGENT_MODEL=openai/qwen3-4b
export LIFE_USER_MODEL=openai/your-user-simulator

# Local solver tokenizer assets for META/native comparisons and OS.
export LIFE_SOLVER_TOKENIZER=/absolute/path/to/Qwen3-4B
# Optional override for the verified Flash tokenizer cache.
# export LIFE_FLASH_TOKENIZER=/absolute/path/to/DeepSeek-V4-Flash-tokenizer

# Native optimizer gateway (host prefix before /protocol/openai/v1).
export MR_HOST=https://your-optimizer-gateway.example.com
export MR_API_KEY=replace-me
export DASHSCOPE_API_KEY=replace-me
