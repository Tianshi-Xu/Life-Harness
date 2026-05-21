# Life-harness

This repository contains the harness code for the paper. The two benchmark
families are kept in separate top-level folders because they use different
runtime environments.

## Repository Layout

```text
Life-harness/
  AgentBench/      # Docker-based AgentBench-style tasks
  TauBench/    # uv-based tau-bench-style tasks
```

Use the README in each subfolder for setup and evaluation commands:

- `AgentBench/README.md`: ALFWorld, DBBench, OS, and WebShop.
- `TauBench/README.md`: Airline, Retail, and Telecom.
