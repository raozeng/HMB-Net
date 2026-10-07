#!/bin/bash

# ==================================================
# MambaRNA Batch Execution Script
# 顺序执行多个数据集 (依次等待上一个完成再执行下一个)
# ==================================================

# 总体进度日志文件
BATCH_LOG="batch_total_progress.log"

echo "========================================" > $BATCH_LOG
echo "  MambaRNA Batch Execution Started" >> $BATCH_LOG
echo "  Time: $(date)" >> $BATCH_LOG
echo "========================================" >> $BATCH_LOG

# 需要遍历的数据集数组
DATASETS=("h_b" "h_k" "h_l" "m_b" "m_h" "m_k")

# 统一参数
MODEL="hybrid"
EPOCHS=60
BATCH_SIZE=32
LR="1e-4"

echo "Start batch processing ${#DATASETS[@]} datasets. Check '$BATCH_LOG' for overall progress."

for DATASET in "${DATASETS[@]}"; do
    echo "[$(date)] >> Starting dataset: $DATASET" | tee -a $BATCH_LOG
    
    # 构建训练和测试数据路径 (与单例脚本逻辑保持一致)
    if [ -f "data/train/benchmark/${DATASET}.fa" ]; then
        TRAIN_DATA="data/train/benchmark/${DATASET}.fa"
        TEST_DATA="data/test/independent/${DATASET}_Test.fa"
    elif [ -f "data/train/benchmark/${DATASET}_all.fa" ]; then
        TRAIN_DATA="data/train/benchmark/${DATASET}_all.fa"
        TEST_DATA="data/test/independent/${DATASET}_Test.fa"
    else
        echo "Error: Could not find ${DATASET}.fa or ${DATASET}_all.fa in data/train/benchmark! Skipping..." | tee -a $BATCH_LOG
        continue
    fi
    
    # 每个数据集独立的日志文件
    LOG_FILE="mambarna_${DATASET}_${MODEL}_nohup.log"
    echo "  -- Training Log will be saved to: $LOG_FILE" | tee -a $BATCH_LOG
    
    # 激活虚拟环境
    source .venv/bin/activate 2>/dev/null || true
    
    # 执行具体的 Python 命令，不放入后台，保证严格的顺序执行（挨个跑）
    # 将标准输出和错误都重定向到专属日志文件中
    python3 run.py \
        --mode all \
        --model_type "$MODEL" \
        --train_data "$TRAIN_DATA" \
        --test_data "$TEST_DATA" \
        --epochs "$EPOCHS" \
        --k_folds 5 \
        --lr "$LR" \
        --batch_size "$BATCH_SIZE" > "$LOG_FILE" 2>&1
        
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] << Finished dataset: $DATASET successfully." | tee -a $BATCH_LOG
    else
        echo "[$(date)] << Failed dataset: $DATASET with exit code $EXIT_CODE. See $LOG_FILE for details." | tee -a $BATCH_LOG
    fi
    
    echo "----------------------------------------" >> $BATCH_LOG
done

echo "========================================" >> $BATCH_LOG
echo "All datasets processed!" | tee -a $BATCH_LOG
echo "Time: $(date)" >> $BATCH_LOG
echo "========================================" >> $BATCH_LOG
