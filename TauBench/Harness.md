# General Harness Design: A Four-Layer Intervention Framework (Method Section)

---

## 3.1 Design Principles

Task-environment determinism is the central leverage point of the harness: **the more precise the environment signals are, the more reliable the trigger conditions become at every intervention layer**. Action-validity checks can be performed at verb-level granularity; stagnation detection can be triggered by binary signals rather than probabilistic estimates; and answer normalization can target exact formatting deviations. In contrast, in open-domain settings or environments with high perceptual noise, the same intervention logic would produce many false triggers.

Baseline failure-mode analysis reveals a systematic pattern: **most failures are not caused by insufficient model reasoning capability, but by execution-level errors that can be mechanically identified and intervened on**. Root-cause analysis over failed samples from the four environments suggests three major failure modes, corresponding to the design motivations of H2, H3, and H4.

### Failure Type 1: Action-Interface Errors (H2)

Errors may occur before the agent output reaches the environment execution layer. These errors can be detected before execution with deterministic rules: some can be repaired, while others should be blocked.

The shared property of this failure type is that **it can be judged before execution using prior information**. H2 does not need to infer the agent's intent; it only checks the proposed action against these references. Repairable errors are handled by rescue rules that synthesize a valid call, while unsafe or invalid errors are intercepted by gates that inject corrective feedback.

### Failure Type 2: Tool-Usage Mismatch (H3)

The agent's tool call may be syntactically valid but semantically wrong or inefficient in the current environment. For example, the agent may choose the wrong tool, pass arguments in an incompatible format, or follow calling conventions that do not match the environment. Such mismatches already exist at the beginning of an episode and are unlikely to be self-corrected by the agent.

The root cause is a mismatch between pretrained knowledge and the tool semantics of a specific environment. H3 augments tool descriptions (`tool.description`) by embedding environment constraints and correct usage patterns directly into the content the agent reads whenever it considers calling a tool. This calibrates the tool contract at episode initialization. **H3 does not fix reasoning errors; it fixes misunderstandings of the tool contract**.

### Failure Type 3: Self-Reinforcing Trajectory Failure (H4)

The agent may fall into a behavior pattern that does not advance the task. Because the agent lacks reliable self-awareness of this pattern, it can reinforce itself until the step budget is exhausted.

This failure type is detectable through **statistical properties of the behavior sequence rather than content semantics**. One does not need to understand the meaning of a SQL query to know that it has been repeated three times, nor does one need to understand navigation semantics to detect an A<->B loop. All detection conditions are based on counts over action and observation sequences, independent of the specific content, which makes them broadly generalizable.

---

The harness is designed to **systematically eliminate these three types of execution-level failures without changing model weights or modifying the evaluation datasets**, while providing precise guidance for the remaining reasoning-level errors. H5 serves as a general knowledge layer: through task-type-aware skill retrieval and step-wise suggestions aligned with real-time state, it provides an additional buffer against all three failure types. The four intervention layers act at different points in the agent reasoning and execution process:

| Layer | Trigger Timing | Target Failure Type | Core Function |
|---|---|---|---|
| **H2** Action Gate | After agent output, before tool execution | Action-interface errors | Format repair, validity checking, forced action execution |
| **H3** Tool-Description Embedding | At episode initialization, once | Tool-usage mismatch | Tool-description augmentation and calibration of tool-contract understanding |
| **H4** Post-Execution Monitor | After tool execution, before the next turn | Self-reinforcing trajectory failure | Error recovery, stagnation detection, turn-budget management |
| **H5** Goal-Directed Guidance | At episode start | General improvement | Skill retrieval and injection |

The four layers have clear responsibilities and operate together: H3 sets up defenses at the beginning of an episode, H2 guards each pre-execution step, H4 diagnoses the trajectory after execution, and H5 injects task-relevant experience.
