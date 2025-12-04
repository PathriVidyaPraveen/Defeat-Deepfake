#!/bin/bash
set -e

# --- CONFIGURATION ---
# Change these paths to match your actual folders on Param Seva
DATA_ROOT="../../data/Celeb_df" 
PROCESSED_DIR="../../data/Celeb_df_processed"

echo "=========================================="
echo "Pipeline Started at $(date)"
echo "=========================================="

# 1. Check if we need to preprocess
if [ -d "$PROCESSED_DIR" ] && [ "$(ls -A $PROCESSED_DIR)" ]; then
    echo "[INFO] Processed data found at $PROCESSED_DIR."
else
    echo "[INFO] Processed data NOT found. Running preprocessing..."
    echo "Reading from: $DATA_ROOT"
    echo "Saving to:    $PROCESSED_DIR"
    
    python3 preprocess.py --data_path "$DATA_ROOT" --output_path "$PROCESSED_DIR"
fi

# 2. Run Training
echo "=========================================="
echo "[INFO] Starting Training..."
python3 train.py --data_path "$PROCESSED_DIR" --epochs 20 --batch_size 32

echo "=========================================="
echo "Pipeline Finished at $(date)"