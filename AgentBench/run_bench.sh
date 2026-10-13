#!/usr/bin/env bash
# ============================================================
# AgentBench command reference
# Run commands manually and manage concurrency as needed.
# First: cd AgentBench && source .venv/bin/activate
# ============================================================

# ── claude-opus-4-8 ──────────────────────────────────────────

# alfworld
docker compose -f extra/docker-compose.yml up -d --force-recreate redis controller alfworld-std
python -m src.assigner --config configs/assignments/alfworld_claude-opus-4-8.yaml

# dbbench
docker compose -f extra/docker-compose.yml up -d --force-recreate redis controller dbbench-std
python -m src.assigner --config configs/assignments/dbbench_claude-opus-4-8.yaml

# os
docker compose -f extra/docker-compose.yml up -d --force-recreate redis controller os-std
python -m src.assigner --config configs/assignments/os_claude-opus-4-8.yaml

# webshop (allow about 60 seconds for the service to become ready)
docker compose -f extra/docker-compose.yml up -d --force-recreate redis controller webshop-std
python -m src.assigner --config configs/assignments/webshop_claude-opus-4-8.yaml

# ── gemini-3.1-pro-preview ───────────────────────────────────

# alfworld
docker compose -f extra/docker-compose.yml up -d --force-recreate redis controller alfworld-std
python -m src.assigner --config configs/assignments/alfworld_gemini-3.1-pro-preview.yaml

# dbbench
docker compose -f extra/docker-compose.yml up -d --force-recreate redis controller dbbench-std
python -m src.assigner --config configs/assignments/dbbench_gemini-3.1-pro-preview.yaml

# os
docker compose -f extra/docker-compose.yml up -d --force-recreate redis controller os-std
python -m src.assigner --config configs/assignments/os_gemini-3.1-pro-preview.yaml

# webshop
docker compose -f extra/docker-compose.yml up -d --force-recreate redis controller webshop-std
python -m src.assigner --config configs/assignments/webshop_gemini-3.1-pro-preview.yaml

# ── gpt-5.5 ──────────────────────────────────────────────────

# alfworld
docker compose -f extra/docker-compose.yml up -d --force-recreate redis controller alfworld-std
python -m src.assigner --config configs/assignments/alfworld_gpt-5.5.yaml

# dbbench
docker compose -f extra/docker-compose.yml up -d --force-recreate redis controller dbbench-std
python -m src.assigner --config configs/assignments/dbbench_gpt-5.5.yaml

# os
docker compose -f extra/docker-compose.yml up -d --force-recreate redis controller os-std
python -m src.assigner --config configs/assignments/os_gpt-5.5.yaml

# webshop
docker compose -f extra/docker-compose.yml up -d --force-recreate redis controller webshop-std
python -m src.assigner --config configs/assignments/webshop_gpt-5.5.yaml

# ── Stop all services ────────────────────────────────────────
# docker compose -f extra/docker-compose.yml down
