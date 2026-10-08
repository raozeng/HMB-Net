#!/bin/bash

# ==================================================
# MambaRNA Execution Launcher
# ==================================================
# This script mirrors the robust nohup execution architecture 
# from PhyNet-DTA for safe remote server execution.

# 1. Read positional parameters (with defaults)
DATASET=${1:-Atlas_h_m5C}
MODEL_TYPE=${2:-hybrid}
EPOCHS=${3:-30}
BATCH_SIZE=${4:-64}
LR=${5:-1e-3}
EXTRA_ARGS="${@:6}"

# Build exact paths to the data directory (assuming script run from project root)
if [ -f "data/train/benchmark/${DATASET}.fa" ]; then
    TRAIN_DATA="data/train/benchmark/${DATASET}.fa"
    TEST_DATA="data/test/independent/${DATASET}_Test.fa"
elif [ -f "data/train/benchmark/${DATASET}_all.fa" ]; then
    TRAIN_DATA="data/train/benchmark/${DATASET}_all.fa"
    TEST_DATA="data/test/independent/${DATASET}_Test.fa"
else
    echo "Error: Could not find ${DATASET}.fa or ${DATASET}_all.fa in data/train/benchmark!"
    exit 1
fi

# Unique log file name based on dataset and model
LOG_FILE="mambarna_${DATASET}_${MODEL_TYPE}_nohup.log"

echo "=================================================="
echo "   MambaRNA Target Execution Launcher"
echo "   Dataset:    $DATASET"
echo "   Model:      $MODEL_TYPE"
echo "   Epochs:     $EPOCHS"
echo "   Batch Size: $BATCH_SIZE"
echo "   Learning R: $LR"
echo "   Train Path: $TRAIN_DATA"
echo "   Test Path:  $TEST_DATA"
echo "   Extra Args: $EXTRA_ARGS"
echo "   Log File:   $LOG_FILE"
echo "=================================================="

# 2. Run the process in the background using nohup
nohup bash -c '
# Source virtual environment if exists
source .venv/bin/activate 2>/dev/null || true

echo "==================================="
echo "Starting Training & Testing Pipeline"
echo "Model: '"$MODEL_TYPE"'"
echo "==================================="

# Note: Using python3 run.py assuming execution from repository root
# If code throws "not found", ensure you are running this from the directory containing run.py.
python3 run.py \
    --mode all \
    --model_type '"$MODEL_TYPE"' \
    --train_data '"$TRAIN_DATA"' \
    --test_data '"$TEST_DATA"' \
    --epochs '"$EPOCHS"' \
    --k_folds 5 \
    --lr '"$LR"' \
    --batch_size '"$BATCH_SIZE"' '"$EXTRA_ARGS"'

echo ""
echo "Pipeline finished! Check results/ directory for predictions."
' > "$LOG_FILE" 2>&1 &

PID=$!
echo "Process started in background. PID: $PID"
echo "To watch progress, run:"
echo "  tail -f $LOG_FILE"
echo "To stop process, run:"
echo "  kill $PID"
echo "=================================================="
