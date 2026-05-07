

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TEST_MODE="test" 
FLY_MODE="3D"  # 2D, 3D
PY_FILE="$PROJ_ROOT/src/eval_airsim_E2E_3D.py"

MAP_NAME="UrbanDistrict"
CLIENT_PORT=41451


cd "$PROJ_ROOT"


MODEL_PATH="$PROJ_ROOT/src/weights/best_model_3D.pth"
WORKSPACE_ROOT="$(cd "$PROJ_ROOT/../.." && pwd)"
CONTENT_DIR="$WORKSPACE_ROOT/DATA/${TEST_MODE}"
RESULTS_DIR="$PROJ_ROOT/logs/E2E_${FLY_MODE}/${TEST_MODE}"



# 2D--------------------------------------------------------------------------------------------------------
python "$PY_FILE" \
  --client_port "$CLIENT_PORT" \
  --train_mode $TEST_MODE \
  --model_path "$MODEL_PATH" \
  --content_dir "$CONTENT_DIR" \
  --map_name "$MAP_NAME" \
  --results_dir "$RESULTS_DIR" \


