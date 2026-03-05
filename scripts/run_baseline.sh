#!/usr/bin/env bash
# Baseline one-shot FL: alpha=0.1, seed=42, 15 clients, epoch=1.
# Usage: from project root (with venv activated or PYTHON set):
#   ./scripts/run_baseline.sh   or   bash scripts/run_baseline.sh

set -e
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then PYTHON=python3; fi

ALPHA=0.1
SEED=42
SPLIT_PATH="outputs/splits/cifar10_dirichlet_a${ALPHA}_seed${SEED}.npy"

# 1) Generate split if missing
if [[ ! -f "$SPLIT_PATH" ]]; then
  echo "Generating split alpha=$ALPHA seed=$SEED ..."
  "$PYTHON" -m src.data.check_split --alpha "$ALPHA" --seeds "$SEED"
fi

# 2) Start server (background)
echo "Starting server..."
"$PYTHON" -m src.main_server &
SERVER_PID=$!
sleep 5

# 3) Start 15 clients
for cid in $(seq 0 14); do
  "$PYTHON" -m src.main_client --cid "$cid" --split_path "$SPLIT_PATH" --seed "$SEED" &
done
wait

kill $SERVER_PID 2>/dev/null || true
echo "Done. Check outputs/client_protos_a${ALPHA}_s${SEED}/ and outputs/metrics/"
