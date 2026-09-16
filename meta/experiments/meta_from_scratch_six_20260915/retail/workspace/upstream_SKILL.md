---
name: meta-harness-terminal-bench-2
description: Run one iteration of AgentHarness evolution for Terminal-Bench 2.
---

# Meta-Harness (Terminal-Bench 2)

Run ONE iteration of agent scaffold evolution.

**You do NOT run benchmarks.** You analyze results + failed trajectories, propose agent variants, and implement them. The outer loop (`meta_harness.py`) handles benchmarking.

## CRITICAL CONSTRAINTS

- You MUST produce 1 new agent variant every iteration
- Do NOT write "the frontier is optimal" or "stop iterating", or abort early.

### Anti-overfitting rules

- **No task-specific hints.** Do not hardcode knowledge about specific tasks. Agents must be general-purpose.
- **Never mention task names** in agent code, prompts, or comments. No references like "if task contains 'async'" or "for polyglot tasks." If your improvement only helps one task, it's too specific.
- **General guidance is OK.** Rules like "back up files before opening them with tools that modify on read" are fine -- they happen to help specific tasks but apply broadly. The test: would this advice be useful to a human developer working on MANY unfamiliar tasks?
- **If in doubt, make it more general.** "Always read eval scripts before submitting" > "Read the grading script for DNA assembly tasks."

## CONTEXT

You are evolving the **AgentHarness** agent scaffold for Terminal-Bench 2. It is located in `agents/baseline_kira.py`.

**The search space is arbitrary Python code.** You can override any method, call any library, make raw API calls, add new tools, change how the LLM is called, rewrite command execution, intercept and transform observations -- anything that's expressible in Python is fair game. The only constraint is that the agent must subclass `harbor.agents.terminus_2.terminus_2.Terminus2` in the same way as `baseline_kira.py` does (for compatibility with the eval harness).

**Model: Claude Opus 4.6.** Evaluation uses the full official 89-task TB2 dataset. `meta_harness.py` chooses the trial count; the default path uses 2 trials per task.

**Key files to read:**

- `agents/baseline_kira.py` - the full baseline implementation. Read to understand overridable methods.

You should copy over one of the agents in `agents/` as a starting point. **You can rewrite or override ANY method.** Never import from other candidate agents. Copy any code you want to reuse.

Key methods:

- `_call_llm_with_tools` - makes the litellm API call. Override to change tools, add parameters, adjust retries.
- `_parse_tool_calls` - converts raw tool call dicts to commands. Override to add new tools.
- `_execute_commands` - runs commands on tmux. Override to change execution behavior.
- `_run_agent_loop` - main episode loop. Override for structural changes.
- `_get_completion_confirmation_message` - what to ask on the "are you sure?" step.
- `_get_prompt_template_path` - path to system prompt. Override to use a custom prompt.
- `_summarize_context` - summarizes history on context overflow. Override to change summarization.
- `_execute_image_read` - handles image_read tool. Override to change multimodal behavior.

## CANDIDATE DESIGN

Each candidate is a **single Python file** at `agents/<name>.py` containing the full agent class. No subdirectories needed.

The agent class must be named `AgentHarness` and subclass `harbor.agents.terminus_2.terminus_2.Terminus2`.
The agent will be loaded and evaluated through Harbor via `--agent-import-path "agents.<name>:AgentHarness"`.

### What you can and cannot modify

- **CAN**: edit your new `agents/<name>.py` file freely.
- **CAN**: create a new prompt template at `prompt-templates/<name>.txt` and point `_get_prompt_template_path` to it.
- **CANNOT**: modify any existing agent file, `meta_harness.py`, or `claude_wrapper.py`.

### Design principles

- Your primary goal is to improve the agent's performance (pass rate) on the full 89-task Terminal-Bench-2 dataset.
- One mechanism per candidate. Each candidate tests exactly one hypothesis. If you're tempted to add "and also..." -- that's a second candidate.
- Mechanism-first. Identify a specific failure mode or hypothesis from trajectories, then design changes that target it. Never add changes speculatively.

## WORKFLOW

### Step 1: Analyze (1 subagent)

Launch ONE Agent subagent (subagent_type: "general-purpose"). It should:

1. Read state files:
   - `frontier_val.json` - current best agent per task (path given in task prompt)
   - `evolution_summary.jsonl` - what's been tried, what worked/didn't, plus `rollout_metrics` (path given in task prompt)
2. **Deep-read failed AND successful trajectories.** Most important step.
   Use the current run outputs under `jobs/` and `logs/` as the primary source of truth.

3. Read agent implementations in `agents/*.py`
4. Return:

```
STATE: <5-line summary: current scores, what's been tried, avg tokens/turns/cost>

HYPOTHESIS: "<falsifiable claim about what will improve scores>"
CANDIDATE: name=<snake_case>, changes="<specific changes>", prediction="<expected pass rate improvement AND expected token/turn impact>"
```

### Step 2: Implement (1 subagent)

Launch **1 Agent subagent** (subagent_type: "general-purpose"). The prompt must include the candidate name, class name, specific changes from Step 1, and the working directory.

The subagent should:

1. Copy over one of the agents in `agents/` as a starting point: `agents/<snake_case_name>.py`. Your final agent should be a subclass of `harbor.agents.terminus_2.terminus_2.Terminus2`.
2. Make targeted changes according to the agent instructions.
3. **Smoke test**: validate import (`uv run python -c "from agents.<name> import *; print('OK')"`)
4. Return: file path, validation status

### Step 3: Write pending_eval.json

Write `pending_eval.json` to the path specified in the task prompt:

```json
{
  "iteration": <N>,
  "candidates": [
    {
      "name": "<name>",
      "import_path": "agents.<name>:AgentHarness",
      "hypothesis": "<falsifiable claim>",
      "changes": "<what was changed>",
      "expected_efficiency": "<expected token/turn impact>"
    }
  ]
}
```

Output: `CANDIDATES: <name1>`

## IMPORTANT NOTES

- **Always name the class `AgentHarness`** in candidate files. The import path is always `agents.<name>:AgentHarness`.
- **ALL methods in `AgentHarness` are async.** Use `await` when calling super() methods.
- `AgentHarness` calls `litellm.acompletion` directly -- it does NOT use harbor's `Chat` class for LLM calls. The `Chat` object is passed in but only used for message history / token counting.
- The `TOOLS` constant defines the native tool schema. You can extend it with new tools by overriding `_call_llm_with_tools` to pass `tools=TOOLS + [new_tool]` and `_parse_tool_calls` to handle the new tool.
- The prompt template uses `{instruction}` and `{terminal_state}` placeholders (standard Python `.format()`).
- `_limit_output_length` truncates at 30KB (KIRA's limit, vs 10KB in Terminus2).
