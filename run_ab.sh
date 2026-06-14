#!/bin/bash
set -u
cd "$(dirname "$0")"
export YAZ_THREADS="${YAZ_THREADS:-4}"
export YAZ_STEPS="${YAZ_STEPS:-3000}"
echo "########## A/B generalization run (steps=$YAZ_STEPS threads=$YAZ_THREADS) ##########"
for cfg in baseline_1phrasing treatment_8phrasing; do
  echo ""; echo "===== $cfg  (t=$(date -u +%H:%M:%S)Z) ====="
  python3 scripts/train_gen.py "configs/${cfg}.json" || echo "FAILED $cfg, continuing"
done
echo ""; echo "########## A/B DONE (t=$(date -u +%H:%M:%S)Z) ##########"
