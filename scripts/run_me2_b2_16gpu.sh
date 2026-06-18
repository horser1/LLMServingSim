#!/bin/bash

cd /app/LLMServingSim/feat_perinstance_config/LLMServingSim/

set -e

LOG_FILE="logs/me2_b2_16gpu.log"
RUN_ID="me2_b2_16gpu"

# Clear old log
> "$LOG_FILE"

echo "========================================" >> "$LOG_FILE"
echo "Config: me2_b2_16gpu (B2: Context-only)" >> "$LOG_FILE"
echo "Dataset: workloads/me2/workload_me2_01_mixed.jsonl" >> "$LOG_FILE"
echo "Output: outputs/me2/me2_b2_16gpu.csv" >> "$LOG_FILE"
echo "Start time: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

python -m serving \
  --cluster-config 'configs/cluster/me2/me2_b2_16gpu.json' \
  --dtype bfloat16 \
  --dataset 'workloads/me2/workload_me2_01_mixed.jsonl' \
  --output 'outputs/me2/me2_b2_16gpu.csv' \
  --run-id "$RUN_ID" \
  --log-interval 0.5 \
  --log-level INFO \
  --no-enable-prefix-caching \
  >> "$LOG_FILE" 2>&1

echo "Finished time: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
