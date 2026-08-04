#!/usr/bin/env bash
set -e  # 遇到错误立即停止

# ================= 配置区域 =================
# 1. 数据位置
DATA_ROOT="data/NYC_KFolds"

# 2. 输出位置
CACHE_ROOT="cache/nyc_kfold"
CKPT_ROOT="checkpoints/nyc_kfold"

# 3. 训练参数
EPOCHS_PRETRAIN=20
EPOCHS_FINETUNE=30
BATCH_SIZE=256
GPU_ID=2  # 指定使用的GPU ID

# 创建输出目录
mkdir -p ${CACHE_ROOT}
mkdir -p ${CKPT_ROOT}

# 准备一个文件来记录最终结果
RESULT_FILE="kfold_final_results.txt"
# 如果文件不存在，写入表头
if [ ! -f "$RESULT_FILE" ]; then
    echo "K-Fold Cross Validation Results" > ${RESULT_FILE}
    echo "=============================" >> ${RESULT_FILE}
fi

# ================= 开始循环 5 折 =================
# 这里的 {0..4} 对应 fold_0 到 fold_4
for i in {0..4}
do
    echo " "
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "Running FOLD $i / 4"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

    # 定义当前 Fold 的路径变量
    FOLD_DIR="${DATA_ROOT}/fold_$i"
    FOLD_CACHE="${CACHE_ROOT}/nyc_fold_$i.pt"
    PRETRAIN_NAME="pretrain_fold_$i"
    FINETUNE_NAME="finetune_fold_$i"
    
    # [修改点]：修正了模型权重的路径，直接指向 .pt 文件
    PRETRAIN_CKPT="${CKPT_ROOT}/${PRETRAIN_NAME}.pt"
    FINETUNE_CKPT="${CKPT_ROOT}/${FINETUNE_NAME}.pt"

    # ------------------------------------------------
    # 1. 数据预处理 (Preprocessing)
    # ------------------------------------------------
    # 检查缓存是否已存在，如果存在跳过（可选，为了节省时间）
    if [ -f "$FOLD_CACHE" ]; then
        echo "[Fold $i] Cache exists at $FOLD_CACHE, skipping preprocessing."
    else
        echo "[Fold $i] Step 1: Preprocessing..."
        CUDA_VISIBLE_DEVICES=${GPU_ID} python -m src.data.preprocess \
          --train_path "${FOLD_DIR}/NYC_train.csv" \
          --val_path   "${FOLD_DIR}/NYC_val.csv" \
          --test_path  "${FOLD_DIR}/NYC_test.csv" \
          --poi_info_path "${FOLD_DIR}/poi_info.csv" \
          --cache_path "${FOLD_CACHE}" \
          --max_len 128 \
          --num_tod_bins 24 \
          --region_scales_m 300,800,2100
          # --region_scales_m 700,1200,3000 
    fi

    # ------------------------------------------------
    # 2. 预训练 (Masked Pretraining)
    # ------------------------------------------------
    # 检查预训练权重是否已存在
    if [ -f "$PRETRAIN_CKPT" ]; then
        echo "[Fold $i] Pretrain ckpt exists at $PRETRAIN_CKPT, skipping pretrain."
    else
        echo "[Fold $i] Step 2: Pretraining..."
        CUDA_VISIBLE_DEVICES=${GPU_ID} python -m src.pretrain \
          --cache_path "${FOLD_CACHE}" \
          --save_dir "${CKPT_ROOT}" \
          --exp_name "${PRETRAIN_NAME}" \
          --epochs ${EPOCHS_PRETRAIN} \
          --batch_size ${BATCH_SIZE} \
          --lr 1e-3 \
          --mask_prob 0.2
    fi

    # ------------------------------------------------
    # 3. 微调 (Fine-tuning)
    # ------------------------------------------------
    echo "[Fold $i] Step 3: Finetuning..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} python -m src.finetune \
      --cache_path "${FOLD_CACHE}" \
      --save_dir "${CKPT_ROOT}" \
      --exp_name "${FINETUNE_NAME}" \
      --init_from "${PRETRAIN_CKPT}" \
      --epochs ${EPOCHS_FINETUNE} \
      --batch_size ${BATCH_SIZE} \
      --lr 5e-4 

    # ------------------------------------------------
    # 4. 评估 (Evaluation)
    # ------------------------------------------------
    echo "[Fold $i] Step 4: Evaluating..."
    echo "--- Fold $i Results ---" >> ${RESULT_FILE}
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python -m src.eval \
      --cache_path "${FOLD_CACHE}" \
      --ckpt_path "${FINETUNE_CKPT}" \
      --split test >> ${RESULT_FILE}

    echo "Fold $i Complete."
done

echo "All folds finished. Check ${RESULT_FILE} for details."