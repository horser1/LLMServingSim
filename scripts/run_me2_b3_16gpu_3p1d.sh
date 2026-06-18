#!/bin/bash

cd /app/LLMServingSim/feat_perinstance_config/LLMServingSim/

set -e

LOG_FILE="logs/me2_b3_16gpu_3p1d.log"
RUN_ID="me2_b3_16gpu_3p1d"

# Clear old log
> "$LOG_FILE"

echo "========================================" >> "$LOG_FILE"
echo "Config: me2_b3_16gpu_3p1d (B3: P/D-only 3P:1D)" >> "$LOG_FILE"
echo "Dataset: workloads/me2/workload_me2_01_mixed.jsonl" >> "$LOG_FILE"
echo "Output: outputs/me2/me2_b3_16gpu_3p1d.csv" >> "$LOG_FILE"
echo "Start time: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

python -m serving \
  --cluster-config 'configs/cluster/me2/me2_b3_16gpu_3p1d.json' \
  --dtype bfloat16 \
  --dataset 'workloads/me2/workload_me2_01_mixed.jsonl' \
  --output 'outputs/me2/me2_b3_16gpu_3p1d.csv' \
  --run-id "$RUN_ID" \
  --log-interval 0.5 \
  --log-level INFO \
  --no-enable-prefix-caching \
  >> "$LOG_FILE" 2>&1

echo "Finished time: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
