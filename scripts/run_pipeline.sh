#!/usr/bin/env bash
# End-to-end pipeline driver.
#
# Paths are configured at the top of each Python script; edit them to match your
# data layout (see docs/reproduction.md) before running.
#
# Usage:
#   bash scripts/run_pipeline.sh [stage]
#     stage = sij | bme | axspa | all   (default: all)

set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${1:-all}"
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

run_sij() {
  echo "== Stage 1a: SIJ localization =="
  python src/stage1_sij_localization/prepare_dataset.py
  python src/stage1_sij_localization/train.py
  python src/stage1_sij_localization/evaluate.py
}

run_bme() {
  echo "== Stage 1b: BME classification =="
  python src/stage1_bme_classification/train.py
  python src/stage1_bme_classification/evaluate.py
}

run_axspa() {
  echo "== Stage 2: patient-level axSpA classification =="
  python src/stage2_axspa_classification/train.py
  python src/stage2_axspa_classification/evaluate.py
}

case "$STAGE" in
  sij)   run_sij ;;
  bme)   run_bme ;;
  axspa) run_axspa ;;
  all)   run_sij; run_bme; run_axspa ;;
  *) echo "unknown stage: $STAGE (expected sij | bme | axspa | all)" >&2; exit 1 ;;
esac

echo "done."
