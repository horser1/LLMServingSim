#!/bin/bash

cd /app/LLMServingSim/feat_perinstance_config/LLMServingSim/

set -e

LOG_FILE="me1_analyse_8tp1.log"

# 清空旧 log
> "$LOG_FILE"

datasets=(
  "workloads/me1/workload_prefill_heavy_10rps.jsonl"
  "workloads/me1/workload_decode_heavy_10rps.jsonl"
  "workloads/me1/workload_balanced_10rps.jsonl"
)

outputs=(
  "outputs/me1/me1_prefill_heavy_10rps_8tp1.csv"
  "outputs/me1/me1_decode_heavy_10rps_8tp1.csv"
  "outputs/me1/me1_balanced_10rps_8tp1.csv"
)

for i in "${!datasets[@]}"; do
  echo "========================================" >> "$LOG_FILE"
  echo "Running dataset: ${datasets[$i]}" >> "$LOG_FILE"
  echo "Output file: ${outputs[$i]}" >> "$LOG_FILE"
  echo "Start time: $(date)" >> "$LOG_FILE"
  echo "========================================" >> "$LOG_FILE"

  python -m serving \
    --cluster-config 'configs/cluster/me1_instance_8tp1.json' \
    --dtype bfloat16 \
    --dataset "${datasets[$i]}" \
    --output "${outputs[$i]}" \
    --num-reqs 60 \
    --no-enable-prefix-caching \
    >> "$LOG_FILE" 2>&1

  echo "Finished time: $(date)" >> "$LOG_FILE"
  echo "" >> "$LOG_FILE"
done