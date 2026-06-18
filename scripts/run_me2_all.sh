#!/bin/bash

# Master script to run all 7 me2 simulation experiments sequentially.
# Errors in one experiment do not stop the next one.
# Usage:  screen -S me2_all -L -Logfile me2_all_screen.log bash scripts/run_me2_all.sh

set +e  # Do NOT stop on errors — continue to next experiment

ROOT_DIR="/app/LLMServingSim/feat_perinstance_config/LLMServingSim"
cd "$ROOT_DIR" || exit 1

# Ensure output directory exists
mkdir -p outputs/me2 logs

# Define the 7 experiments in order
# Format: "config_name|log_file|output_file|description"
EXPERIMENTS=(
  "configs/cluster/me2/me2_b1_16gpu.json|logs/me2_b1_16gpu.log|outputs/me2/me2_b1_16gpu.csv|B1: Homogeneous"
  "configs/cluster/me2/me2_b2_16gpu.json|logs/me2_b2_16gpu.log|outputs/me2/me2_b2_16gpu.csv|B2: Context-only"
  "configs/cluster/me2/me2_b3_16gpu_1p3d.json|logs/me2_b3_16gpu_1p3d.log|outputs/me2/me2_b3_16gpu_1p3d.csv|B3: P/D-only 1P:3D"
  "configs/cluster/me2/me2_b3_16gpu_2p2d.json|logs/me2_b3_16gpu_2p2d.log|outputs/me2/me2_b3_16gpu_2p2d.csv|B3: P/D-only 2P:2D"
  "configs/cluster/me2/me2_b3_16gpu_3p1d.json|logs/me2_b3_16gpu_3p1d.log|outputs/me2/me2_b3_16gpu_3p1d.csv|B3: P/D-only 3P:1D"
  "configs/cluster/me2/me2_b4_16gpu.json|logs/me2_b4_16gpu.log|outputs/me2/me2_b4_16gpu.csv|B4: Decode-biased-only"
  "configs/cluster/me2/me2_b5_16gpu.json|logs/me2_b5_16gpu.log|outputs/me2/me2_b5_16gpu.csv|B5: Coexisting"
)

TOTAL=${#EXPERIMENTS[@]}
PASSED=0
FAILED=0
FAILED_LIST=()

echo "=============================================="
echo "  me2 All Experiments Runner"
echo "  Total: $TOTAL experiments"
echo "  Start time: $(date)"
echo "=============================================="
echo ""

for i in "${!EXPERIMENTS[@]}"; do
  IFS='|' read -r CONFIG LOG OUTPUT DESC <<< "${EXPERIMENTS[$i]}"
  IDX=$((i + 1))

  echo "=============================================="
  echo "[$IDX/$TOTAL] $DESC"
  echo "  Config: $CONFIG"
  echo "  Log:    $LOG"
  echo "  Output: $OUTPUT"
  echo "  Start:  $(date)"
  echo "=============================================="

  # Clear old log
  > "$LOG"

  echo "========================================" >> "$LOG"
  echo "Experiment $IDX/$TOTAL: $DESC" >> "$LOG"
  echo "Config: $CONFIG" >> "$LOG"
  echo "Dataset: workloads/me2/workload_me2_01_mixed.jsonl" >> "$LOG"
  echo "Output: $OUTPUT" >> "$LOG"
  echo "Start time: $(date)" >> "$LOG"
  echo "========================================" >> "$LOG"

  START_TS=$(date +%s)

  python -m serving \
    --cluster-config "$CONFIG" \
    --dtype bfloat16 \
    --dataset 'workloads/me2/workload_me2_01_mixed.jsonl' \
    --output "$OUTPUT" \
    --num-reqs 120 \
    --no-enable-prefix-caching \
    >> "$LOG" 2>&1

  EXIT_CODE=$?
  END_TS=$(date +%s)
  ELAPSED=$((END_TS - START_TS))

  echo "Finished time: $(date)" >> "$LOG"
  echo "Exit code: $EXIT_CODE" >> "$LOG"
  echo "Elapsed: ${ELAPSED}s" >> "$LOG"
  echo "" >> "$LOG"

  if [ $EXIT_CODE -eq 0 ]; then
    echo "[$IDX/$TOTAL] $DESC — PASSED (${ELAPSED}s)"
    PASSED=$((PASSED + 1))
  else
    echo "[$IDX/$TOTAL] $DESC — FAILED (exit code $EXIT_CODE, ${ELAPSED}s)"
    FAILED=$((FAILED + 1))
    FAILED_LIST+=("[$IDX] $DESC (exit $EXIT_CODE)")
  fi

  echo ""
done

echo "=============================================="
echo "  me2 All Experiments — Summary"
echo "  Finished: $(date)"
echo "  Total:  $TOTAL"
echo "  Passed: $PASSED"
echo "  Failed: $FAILED"
if [ $FAILED -gt 0 ]; then
  echo ""
  echo "  Failures:"
  for f in "${FAILED_LIST[@]}"; do
    echo "    $f"
  done
fi
echo "=============================================="

# Exit with 0 so screen doesn't complain — check the summary output for actual results
exit 0
