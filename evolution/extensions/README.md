# Extend Life-Harness to another benchmark

This package provides a runnable extension interface: implement a native benchmark adapter, then use the shared runner to evolve H2–H5 from empty hooks and evaluate the frozen result. The runner handles hook execution, public TRAIN evidence, model proposals, sandboxed validation, candidate selection, final TEST and accuracy/token records.

## Run the complete offline example

From the repository root:

```bash
python -m evolution.extensions demo \
  --output evolution/experiments/extension_demo --rounds 2
```

This uses only Python's standard library on Linux. It runs the complete workflow: the first scripted repair raises toy TRAIN accuracy from 0 to 1 and is accepted; the second ties and is rejected; the frozen harness then scores 2/2 on TEST. Both the solver and proposer are local scripts, with no API calls. These scores demonstrate integration, not benchmark performance or LLM learning.

[examples.py](examples.py) contains a complete adapter, episode and offline proposer. [adapter.py](adapter.py) defines the public types, exported by `evolution.extensions`:

```python
from evolution.extensions import (
    BenchmarkAdapter, BenchmarkEpisode, EpisodeSetup,
    ModelResponse, StepResult, ExtensionRunner, evaluate,
)
```

## Implement the native adapter

An adapter is a class or zero-argument factory in an importable Python module. Inheritance is optional; implement the following protocol. The native benchmark keeps ownership of its model calls, environment and grading.

| Adapter member | Contract |
| --- | --- |
| `name` | Unique lowercase identifier, such as `my_benchmark`; do not reuse a built-in domain name |
| `public_context_fields` | Names of public fields hooks may see; use `()` if none |
| `task_ids(split)` | Stable nonempty list of unique string task IDs for `train` or `test`; the two sets must be disjoint |
| `open_episode(task_id, split)` | Create a fresh native episode, including fresh environment and solver state |
| `fingerprint()` | Stable JSON configuration covering dataset/rubric hashes, SDK versions, solver/model settings and native budgets; omit credentials |

Each episode implements:

| Episode member | Contract |
| --- | --- |
| `setup` | `EpisodeSetup(messages, tools, public_context=None, max_turns=20)` |
| `complete(messages, tools)` | Make one solver request and return `ModelResponse(assistant_message, usage=provider_usage)` |
| `execute(message)` | Execute the repaired assistant action in the native environment; return `StepResult(public_feedback_messages, done)` |
| `grade()` | Return native binary success (`bool`, or integer 0/1) |
| `close()` | Release native resources, including after failures |

Messages use `system`, `user`, `assistant` and `tool` roles; tools and assistant tool calls use OpenAI-style function schemas. Preserve native tool-call IDs and tool feedback. Wrap your current solver request, dispatch and grader in these methods; the runnable toy episode shows each boundary. Invalid actions should return ordinary native feedback. Exceptions represent infrastructure failures, are recorded separately, and stop evolution before selection.

The runner automatically registers declared public context fields and restores their schema in validation workers; no edits to core files are required. Keep answers, private grader state and hidden observations out of messages and `public_context`. A fresh H2–H5 instance is created for each task. Vetoes and retries consume the declared solver-turn budget; hooks cannot execute native tools themselves.

The adapter source file is hashed automatically. Include all external episode modules, data, grading dependencies and relevant configuration in `fingerprint()` so changes invalidate the run. Class definitions must live in a source file rather than an interactive notebook. Use globally unique IDs when a benchmark has split-local numbering.

## Start real evolution

Install the shared environment and pinned optimizer tokenizer following the root [Quick Start](../../README.md#quick-start). Set `MR_HOST` and `MR_API_KEY` for the optimizer, and configure your adapter's solver and native services separately. Put `my_benchmark.py` on `PYTHONPATH`, exporting `MyAdapter`:

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
TauBench/.venv/bin/python -m evolution.extensions evolve \
  --adapter my_benchmark:MyAdapter \
  --output evolution/experiments/my_benchmark --rounds 6
```

This command makes model calls. It starts from empty H2–H5, evaluates the incumbent and candidate on the same complete TRAIN task set each round, accepts only strict improvement, seals TRAIN selection, then evaluates the frozen harness once on TEST. Arbitrary nonempty binary task sets are supported. This extension runner repeats the full TRAIN set; benchmark-specific paper reproduction drivers retain their own batch schedules. The extension API currently runs Life-Harness; use existing reproduction commands for Meta-Harness/AHE comparisons.

Every invocation requires a fresh output directory. Failed or interrupted model work is preserved and is never automatically replayed. The runner checks configuration, task identities and artifact hashes during evaluation. Infrastructure failures remain in the task denominator and are flagged; incomplete evaluations cannot select a candidate. Missing optimizer usage stops promotion.

Python entry point:

```python
from evolution.extensions import ExtensionRunner
from my_benchmark import MyAdapter

ExtensionRunner(MyAdapter(), "evolution/experiments/my_benchmark", rounds=6).run()
```

## Evaluate a frozen result

The evolution command already performs final TEST. Use this separate command for another fixed-harness evaluation, without further optimization:

```bash
TauBench/.venv/bin/python -m evolution.extensions evaluate \
  --adapter my_benchmark:MyAdapter \
  --manifest evolution/experiments/my_benchmark/my_benchmark/runtime/final/manifest.json \
  --output evolution/experiments/my_benchmark_test --split test
```

The second `my_benchmark` in the manifest path is the adapter's `name`. This command calls the native solver and grader and records any infrastructure failures before returning an error. The Python `evaluate(adapter, manifest, output, split="test")` function returns the summary, including its `infra_errors` field.

## Inspect accuracy and token consumption

| Output | Contents |
| --- | --- |
| `protocol.json` | Adapter/configuration fingerprint, splits, initialization and code hashes |
| `TRAINING_RESULTS.json` | Per-round incumbent/candidate accuracy and acceptance decisions |
| `FINAL_RESULTS.json` | Frozen final TEST results and manifest; explicitly excluded from selection |
| `TOKEN_USAGE.json` | Separate optimizer, solver and optional native model-role totals |
| `requests.jsonl` | Durable request-start/finish events, raw provider usage, failures and pending calls |
| `ADAPTER_NAME/evals/PHASE/` | Full public requests, responses, tool feedback, hook traces, native outcomes and usage summaries |
| `training_seal.json`, `ADAPTER_NAME/state.json` | Selected artifact snapshot, hashes and immutable TRAIN selection seal |
| `optimizer_inputs/`, `ADAPTER_NAME/proposals/`, `ADAPTER_NAME/qualified/` | TRAIN-only evidence, original proposals and validated bundles |

Standalone evaluation writes `config.json`, `summary.json`, episode records and `requests.jsonl` in its output directory. Accuracy always reports successes and the full task denominator. Token summaries distinguish measured usage, unknown usage, pending requests and explicit local non-API calls; missing usage is never counted as measured zero. Return the provider's usage dictionary, including cached-token details, in `ModelResponse`; totals require nonnegative integer `prompt_tokens` and `completion_tokens`. For providers with other field names, add these two canonical fields from their measured input/output counts while retaining the original fields. Do not mark a remote request `local_no_api`.

If native simulation or grading also calls models, the episode may implement the optional `bind_request_recorder(record)` method. The runner calls it before reading `setup`. Wrap **each** extra request, including retries:

```python
def bind_request_recorder(self, record):
    self.record = record

def grade(self):
    response = self.record("judge", lambda: self.native_judge_request())
    return bool(response["success"])
```

The callable returns the native response dictionary with its original `usage`. The recorder passes that response back to the adapter and logs usage only, keeping private grading content out of public trajectories. Use roles such as `judge` or `simulator`; solver and optimizer requests are already metered. Never hide retries inside `complete()` or inside a single recorded request.
