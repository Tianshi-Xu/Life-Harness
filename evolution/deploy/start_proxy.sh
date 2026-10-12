#!/usr/bin/env bash
# Start the round-robin proxy on 127.0.0.1:8400 (see proxy.py).
cd "$(dirname "$0")"
nohup python3 proxy.py > logs/proxy.log 2>&1 &
sleep 4
curl -sf http://127.0.0.1:8400/health && echo
