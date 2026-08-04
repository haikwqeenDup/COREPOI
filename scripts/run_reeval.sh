#!/usr/bin/env bash
set -e

# ================= 配置区域 =================
CACHE_ROOT="cache/nyc_kfold"
CKPT_ROOT="checkpoints/nyc_kfold"

# ====== 论文式 rerank 参数 ======
ALPHA=0.0   # 关闭原 region-consistency
BETA=0.0    # 关闭原 category-consistency

TOPK_REGION=500   # 论文：top 40 regions
TOPM_TIME=5     # 论文：top 5 time periods
GAMMA=20.0       # rerank bias 强度（建议试 0.5 / 1.0 / 2.0）



GPU_ID=0
# ===========================================

RESULT_FILE="reeval_paper_rerank_k${TOPK_REGION}_m${TOPM_TIME}_gamma${GAMMA}.txt"

echo "==================================================="
echo "Starting Re-evaluation (Paper-style Rerank)"
echo "topK Region: ${TOPK_REGION}"
echo "topM Time  : ${TOPM_TIME}"
echo "Gamma      : ${GAMMA}"
echo "Saving results to: ${RESULT_FILE}"
echo "==================================================="

echo "Paper-style Re-evaluation Results" > ${RESULT_FILE}
echo "Date: $(date)" >> ${RESULT_FILE}
echo "---------------------------------------------------" >> ${RESULT_FILE}

for i in {0..4}
do
    echo "Processing FOLD $i ..."

    FOLD_CACHE="${CACHE_ROOT}/nyc_fold_$i.pt"
    CKPT_PATH="${CKPT_ROOT}/finetune_fold_$i.pt"

    if [ ! -f "$CKPT_PATH" ]; then
        ALT_PATH="${CKPT_ROOT}/finetune_fold_$i/best_model.pt"
        if [ -f "$ALT_PATH" ]; then
            echo "  [Info] Found checkpoint at folder structure: $ALT_PATH"
            CKPT_PATH=$ALT_PATH
        else
            echo "  [Error] Checkpoint not found for Fold $i"
            continue
        fi
    fi

    echo "--- Fold $i Results ---" >> ${RESULT_FILE}

    CUDA_VISIBLE_DEVICES=${GPU_ID} python -m src.eval \
      --cache_path "${FOLD_CACHE}" \
      --ckpt_path "${CKPT_PATH}" \
      --split test \
      --topm_time ${TOPM_TIME} \
      --topk_region ${TOPK_REGION} \
      --gamma ${GAMMA} >> ${RESULT_FILE}

    echo "  Fold $i evaluation complete."
done
    #   --alpha ${ALPHA} \
    #   --beta ${BETA} \
echo " "
echo "==================================================="
echo "All Done. Summary of Results:"
echo "==================================================="
grep "TEST" ${RESULT_FILE}
echo "Detailed log saved to: ${RESULT_FILE}"
