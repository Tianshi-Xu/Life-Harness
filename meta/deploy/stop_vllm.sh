#!/usr/bin/env bash
# Stop the proxy and all vLLM instances started by start_vllm.sh.
# Bracket trick: keeps pkill/pgrep patterns from matching this script's own
# command line.
pkill -f "meta/deploy/proxy[.]py" 2>/dev/null
pkill -f "vllm serve.*qwen3-4[b]" 2>/dev/null
sleep 2
pgrep -af "vllm serve.*qwen3-4[b]|proxy[.]py" || echo "all stopped"
