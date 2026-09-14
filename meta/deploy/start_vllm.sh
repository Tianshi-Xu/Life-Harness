#!/usr/bin/env bash
# Launch 8 vLLM instances of Qwen3-4B: GPUs 2-5, 2 instances per GPU.
# Ports: GPU2 -> 8401,8402 ; GPU3 -> 8403,8404 ; GPU4 -> 8405,8406 ; GPU5 -> 8407,8408
# Single entry point for clients: the round-robin proxy on 127.0.0.1:8400 (proxy.py).
#
# Two instances share each GPU. This vLLM fork requires
# util x total <= FREE memory at startup, so slot2 (started after slot1 has
# allocated ~43.5 GiB of the 96 GiB card) must fit in the remaining ~53 GiB.
set -u

MODEL=${MODEL:-/mnt/data2/xts/models/Qwen3-4B}
LOG_DIR=$(dirname "$0")/logs
mkdir -p "$LOG_DIR"

wait_healthy() {  # $1=port
  for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$1/health" > /dev/null 2>&1; then
      echo "  $1 healthy"
      return 0
    fi
    sleep 5
  done
  echo "  $1 FAILED to become healthy, see $LOG_DIR/vllm_$1.log"
  return 1
}

launch() {  # $1=gpu $2=port $3=util
  echo "starting qwen3-4b on gpu $1 port $2 (util $3)"
  CUDA_VISIBLE_DEVICES=$1 nohup vllm serve "$MODEL" \
    --served-model-name qwen3-4b \
    --port "$2" \
    --gpu-memory-utilization "$3" \
    --max-model-len 40960 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    > "$LOG_DIR/vllm_$2.log" 2>&1 &
}

for gpu in 2 3 4 5; do
  p1=$((8400 + (gpu - 2) * 2 + 1))
  p2=$((8400 + (gpu - 2) * 2 + 2))
  launch "$gpu" "$p1" 0.44
  wait_healthy "$p1" || exit 1
  launch "$gpu" "$p2" 0.50
done

echo "waiting for second-slot instances..."
rc=0
for port in 8402 8404 8406 8408; do
  wait_healthy "$port" || rc=1
done
[ "$rc" = 0 ] || exit 1

echo "starting proxy on 8400"
nohup python3 "$(dirname "$0")/proxy.py" > "$LOG_DIR/proxy.log" 2>&1 &
sleep 3
curl -sf http://127.0.0.1:8400/health && echo
echo "done. endpoint: http://127.0.0.1:8400/v1 (model: qwen3-4b)"
