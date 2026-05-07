#!/usr/bin/env bash
set -e

# ==================================================
# Edit these two variables before running
# ==================================================
METHOD="3D"          # 2D or 3D
FLYTYPE="zigzag"    # spiral or zigzag

# ==================================================
# Automatically locate paths
# ==================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ "${METHOD}" = "2D" ]; then
    RESULT_DIR="${BASELINE_DIR}/logs/agent_2D"
elif [ "${METHOD}" = "3D" ]; then
    RESULT_DIR="${BASELINE_DIR}/logs/agent_3D"
else
    echo "Error: METHOD must be 2D or 3D."
    exit 1
fi

OUTPUT_CSV="${RESULT_DIR}/summary_${METHOD}_${FLYTYPE}.csv"

cd "${BASELINE_DIR}"

echo "=================================================="
echo "Calculating Heuristic Metrics"
echo "METHOD:     ${METHOD}"
echo "FLYTYPE:    ${FLYTYPE}"
echo "RESULT_DIR: ${RESULT_DIR}"
echo "OUTPUT_CSV: ${OUTPUT_CSV}"
echo "=================================================="

python src/result_analysis_metrics.py \
    --result_dir "${RESULT_DIR}" \
    --flytype "${FLYTYPE}" \
    --method "${METHOD}" \
    --output_csv "${OUTPUT_CSV}"