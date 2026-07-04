#!/bin/bash
# Run all 9 self_evolverec tasks sequentially on GPU
# Usage: nohup bash run_all.sh > run_all.log 2>&1 &

set -e

cd /Users/zhuzhuwentao/Desktop/autoresearch/self_evolverec
PYTHON=/usr/local/bin/python3.12

TASKS=(
  "hstu_beauty"
  "hstu_baby"
  "hstu_pet"
  "sasrec_beauty"
  "sasrec_baby"
  "sasrec_pet"
  "random_beauty"
  "random_baby"
  "random_pet"
)

echo "=============================================="
echo "Starting all 9 tasks at $(date)"
echo "=============================================="

for task in "${TASKS[@]}"; do
  echo ""
  echo ">>> Running: problem=$task ($(date))"
  LOGFILE="logs/${task}_$(date +%Y%m%d_%H%M%S).log"
  mkdir -p logs

  $PYTHON self_evolverec.py problem=$task > "$LOGFILE" 2>&1

  EXIT_CODE=$?
  echo "<<< Finished: problem=$task (exit=$EXIT_CODE) at $(date)"

  # Sleep briefly between tasks
  sleep 5
done

echo ""
echo "=============================================="
echo "All tasks completed at $(date)"
echo "=============================================="