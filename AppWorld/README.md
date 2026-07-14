# Life-Harness for AppWorld

This adapter follows Life-Harness's four lifecycle layers while leaving the
model, AppWorld transition function, tasks, and evaluator unchanged.

## Strict layer boundaries

| Layer | Runs | Input | May output | Must not do |
|---|---|---|---|---|
| H5 Procedural Skill | once, at episode start | task instruction + training-derived generic skills | at most one matched skill plus a common conditional submission convention, sent to the main agent and API predictor | force/add/remove APIs, edit tool docs, issue actions, consume runtime state |
| H3 Environment Contract | once, after upstream API prediction | selected tool descriptions | cloned descriptions with stable calling conventions | add/remove tools, prescribe workflows, inspect trajectory |
| H2 Action Realization | after model output, before execution | current raw call + complete environment action schemas | same action, representation-only repair (integer strings and an irrelevant `access_token` on schemas that do not declare it), or deterministic block | choose another action, expose/add tools to the model, invent credentials/IDs, use H4 state, block valid repeats |
| H4 Trajectory Regulation | after execution/no-action turn, before next turn | executed action/observation sequence + step count | bounded advisory feedback; discard only oversized private text from a no-action turn | rewrite environment observations or successful action history, block actions, communicate state to H2/H3/H5 |

The dataflow is one-way:

```text
task --H5--> initial messages
selected tools --H3--> calibrated tool descriptions
model call + environment action contract --H2--> same executable call or local validation feedback
environment result --H4--> optional next-turn trajectory feedback
```

Version `v005` removes the cross-layer and semantic interventions used by
earlier experimental policies: forced credential/action rescues, pagination
takeover, task-specific recovery, H4-to-H2 action quarantine, observation
projection, successful-action history rewriting, and H5 API forcing. The only
terminal-call gate is H2-local and deterministic: a submission in the same
model batch as another executable action waits until that action's environment
result can be observed.

## Run

Targeted subset:

```bash
cd /mnt/workspace/xts/others/Life-Harness/AppWorld
TASK_IDS_FILE=artifacts/iterations/v005_boundary_subset.txt \
EXPERIMENT_NAME=life_harness/qwen3-4b/v005_boundary/train \
NUM_PROCESSES=8 \
PREDICTOR_TEMPERATURE=0.0 \
bash scripts/run_subset.sh
```

`TEMPERATURE` controls the main acting model and remains `1.0` by default.
`PREDICTOR_TEMPERATURE` controls only the one-time API routing call; when
omitted it inherits `TEMPERATURE`. Setting it to `0.0` is an optional routing
stability ablation and does not remove sampling from the acting model.

Full train evaluation (16 processes):

```bash
cd /mnt/workspace/xts/others/Life-Harness/AppWorld
bash scripts/run_train_16.sh
```

Each layer can be ablated with `--disable-h2`, `--disable-h3`, `--disable-h4`,
or `--disable-h5` when invoking `python -m life_harness_appworld.runner`.
