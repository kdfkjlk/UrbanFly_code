#!/usr/bin/env bash
set -e

# ==================================================
# Edit these variables before running
# ==================================================
MAP_NAME="ModernCityEnvironment"
FLYTYPE="spiral"        # spiral or zigzag
MODE="test"             # test / train / val_unseen

# ==================================================
# Automatically locate paths
# ==================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${BASELINE_DIR}/../.." && pwd)"

DATA_DIR="${PROJECT_ROOT}/DATA"
OUTPUT_DIR="${BASELINE_DIR}/logs/agent_2D"

cd "${BASELINE_DIR}"

echo "=================================================="
echo "Running Heuristic 2D"
echo "BASELINE_DIR: ${BASELINE_DIR}"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "DATA_DIR:     ${DATA_DIR}"
echo "MAP_NAME:     ${MAP_NAME}"
echo "FLYTYPE:      ${FLYTYPE}"
echo "MODE:         ${MODE}"
echo "OUTPUT_DIR:   ${OUTPUT_DIR}"
echo "=================================================="

python src/agent_2D.py \
    --map_name "${MAP_NAME}" \
    --flytype "${FLYTYPE}" \
    --mode "${MODE}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}"