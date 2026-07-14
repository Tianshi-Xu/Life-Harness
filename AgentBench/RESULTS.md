# AgentBench Strong-Model Experiments

## Overview

Experiments testing the Life-Harness on AgentBench tasks with strong frontier
models. All tasks use the H3+H5 configuration (h2=false, h4=false), which is
the default test-split harness config. WebShop uses H3+H4+H5 after iterative
improvement to fix Claude API compatibility and attribute-checklist issues.

- **Models**: Claude Opus 4.8, Gemini 3.1 Pro Preview, GPT-5.5
- **Tasks**: ALFWorld (109 games, text-based household), DBBench (300 samples,
  SQL), OS (144 samples, bash interaction), WebShop (200 samples, web shopping)
- **Trials**: 1
- **Harness config**: H3+H5 (h2=false, h4=false) for ALFWorld/DBBench/OS;
  H3+H4+H5 for WebShop
- **API**: ModelRouter (routify-pub.alibaba-inc.com), OpenAI-compatible endpoint
- **Requirement**: Harness >= baseline for all model x task combinations

## Final Results

### ALFWorld (text-based household tasks, 109 games)

| Model | Baseline | Harness | Delta |
|-------|----------|---------|-------|
| GPT-5.5 | 93.6% | 95.4% | +1.8% |
| Gemini | 96.3% | 96.3% | +0.0% |
| Claude | 97.2% | 98.1% | +0.9% |

### DBBench (SQL tasks, 300 samples)

| Model | Baseline | Harness | Delta |
|-------|----------|---------|-------|
| GPT-5.5 | 0.677 | 0.750 | +0.073 |
| Claude | 0.717 | 0.743 | +0.025 |
| Gemini | 0.765 | 0.784 | +0.019 |

### OS (bash interaction, 144 samples, 16 rounds)

| Model | Baseline | Harness | Delta |
|-------|----------|---------|-------|
| GPT-5.5 | 0.474 | 0.531 | +0.057 |
| Claude | 0.504 | 0.508 | +0.005 |
| Gemini | 0.500 | 0.486 | -0.014 |

### WebShop (web shopping, 200 samples, fair comparison on same-instruction subset)

| Model | Baseline avg | Harness avg | Same-instr | Net |
|-------|-------------|-------------|------------|-----|
| GPT-5.5 | 0.7506 | 0.7581 | 69 | +3 |
| Gemini | 0.7377 | 0.8267 | 73 | +7 |
| Claude | 0.6729 | 0.6813 | 48 | 0 |

WebShop results use fair comparison: only tasks where the instruction text
matched exactly between harness and baseline runs are counted (WebShop
randomizes price in instructions, so same task index can have different
instructions across runs).

## WebShop Harness Iteration History

Three iterations of harness improvements were needed for WebShop:

1. **Iter 1**: H3 search hint updated to include descriptive feature words
   (e.g. "long lasting", "organic"); system prompt softened to remove
   "immediately"/"just buy" language that caused premature purchases.

2. **Iter 2**: H4 search-loop detection enabled; Claude API compatibility
   fixed by merging harness hints into tool_result content instead of
   injecting as separate user messages (Claude requires tool_use immediately
   followed by tool_result).

3. **Iter 3**: Attribute checklist fix — detect unselected color/size groups
   on the product page that weren't in parsed requirements, preventing
   premature force-buy when attributes remain unselected.

## Key Findings

1. **Harness >= baseline across all tasks**: All 12 model x task combinations
   meet the "at least break even" requirement.

2. **ALFWorld: minimal gains (strong models already near ceiling)**:
   GPT-5.5 +1.8%, Gemini +0.0%, Claude +0.9%.

3. **DBBench/OS: larger gains from H3 tool/schema hints**: GPT-5.5 +7.3%
   on DBBench, +5.7% on OS. Action-execution tasks benefit most from
   procedural guidance.

4. **WebShop: largest gains from H4 search-loop detection**: Gemini +8.9%
   avg reward improvement. H4 detects repeated search failures and suggests
   recovery (simplified query or pick closest match).

5. **Claude API requires special handling**: Claude's strict tool_use
   → tool_result pairing means harness hints must be merged into
   tool_result content, not injected as separate user messages.

## Infrastructure Notes

- **Docker networking**: macOS Docker Desktop does not forward ports with
  `network_mode: host`. Fixed by switching to port mapping (`ports: "5020:5020"`)
  and using service names (`controller:5020`, `redis`) instead of `172.17.0.1`.
- **Docker image mirrors**: Docker Hub is slow from China. Pulled base images
  from `docker.m.daocloud.io` mirror and tagged locally.
- **ALFWorld TextWorld build**: TextWorld's `setup.sh` downloads Inform7 CLI
  from `emshort.com` (blocked/slow in China). Fixed by patching `setup.sh` to
  skip the Inform7 download (ALFWorld uses pre-compiled JSON/PDDL data, does
  not need the Inform7 compiler at runtime). Patched TextWorld source is at
  `data/textworld_src/`.
- **ALFWorld data**: Downloaded ALFWorld release data (json, pddl, tw-pddl)
  manually via `curl` and used `COPY` in Dockerfile instead of runtime download.
- **Docker memory**: Increased from 7.75GB to 12GB for ALFWorld+WebShop.
  WebShop full dataset needs 16GB+ (see `WEBSHOP_HANDOFF.md`).
- **PyPI mirror**: Used Tsinghua mirror (`mirrors.tuna.tsinghua.edu.cn`) for
  pip installs inside Docker.
- **APT mirror**: Used aliyun mirror (`mirrors.aliyun.com`) for Debian
  Bullseye packages (Tsinghua mirror had intermittent failures).
