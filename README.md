# CORE-POI: Consistency-aware Inference for Next POI Recommendation

This repository contains the reference implementation for the paper:

> **CORE-POI: Consistency-aware Inference for Next POI Recommendation**

CORE-POI is a next Point-of-Interest (POI) recommendation framework that addresses semantic inconsistency at inference time. It combines spatio-temporal representation learning with confidence-guided, user-conditioned re-ranking so that fine-grained POI predictions are better aligned with the model's inferred region and time intent.

## Overview

The implementation follows a two-stage training pipeline:

1. **Masked spatio-temporal pre-training**: a bidirectional Transformer learns POI, category, and multi-scale region representations from masked trajectories.
2. **Causal Next-POI fine-tuning**: the pre-trained encoder is adapted to autoregressive next-POI prediction with auxiliary category and region objectives.
3. **Consistency-aware inference**: POI logits can be calibrated at inference time using confidence-weighted region feasibility and personalized temporal history, without changing the trained backbone.

The model encodes the following signals:

- POI identity and category;
- user identity;
- time of day and day of week;
- normalized geographic coordinates with Fourier features;
- region IDs at multiple spatial scales.

## Repository structure

```text
.
├── src/
│   ├── data/
│   │   ├── preprocess.py        # CSV validation, feature construction, cache creation
│   │   ├── features.py          # Time and geographic feature utilities
│   │   ├── dataset_pretrain.py  # Masked trajectory dataset
│   │   ├── dataset_finetune.py  # Next-POI dataset
│   │   └── collate.py
│   ├── models/
│   │   └── model.py             # Spatio-temporal BERT encoder and task heads
│   ├── pretrain.py              # Masked multi-task pre-training
│   ├── finetune.py              # Causal Next-POI fine-tuning
│   ├── eval.py                  # Test evaluation and confidence-guided re-ranking
│   ├── eval_time_only.py        # Alternative time-focused evaluation entry point
│   └── utils/
├── scripts/
│   ├── run_kfold_nyc.sh
│   ├── run_reeval.sh
├── data/                        # Put datasets here; data are not redistributed
├── cache/                       # Generated preprocessing caches
├── checkpoints/                 # Generated model checkpoints
├── requirements.txt
└── README.md
```

## Installation

Python 3.10+ and PyTorch 2.0+ are recommended. A CUDA-enabled installation of PyTorch is recommended for training but is not required for evaluation.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The shell scripts assume Bash, Git Bash, WSL, or a Linux environment. The Python commands below can also be run directly from PowerShell.


## Quick start

For the default NYC pipeline, the  Bash commands are:

### 1. Train

```bash
bash ./scripts/run_kfold_nyc.sh
```

### 1. Eval

```bash
bash ./scripts/run_reeval.sh
```

## Detailed Usage

### 1. Preprocess the data

```bash
python -m src.data.preprocess \
  --train_path data/NYC/NYC_train.csv \
  --val_path data/NYC/NYC_val.csv \
  --test_path data/NYC/NYC_test.csv \
  --poi_info_path data/NYC/poi_info.csv \
  --cache_path cache/nyc_cache.pt \
  --max_len 128 \
  --num_tod_bins 48 \
  --region_scales_m 700,1200,3000
```

This creates a serialized cache containing the vocabulary, trajectory splits, spatial metadata, temporal features, and train-only user-conditioned feasibility mappings.

### 2. Masked spatio-temporal pre-training

```bash
python -m src.pretrain \
  --cache_path cache/nyc_cache.pt \
  --save_dir checkpoints \
  --exp_name planC_pretrain_nyc \
  --epochs 30 \
  --batch_size 256 \
  --lr 3e-4 \
  --weight_decay 1e-2 \
  --mask_prob 0.25 \
  --lambda_cat 0.2 \
  --lambda_region 0.2
```

The best checkpoint is saved as `checkpoints/planC_pretrain_nyc.pt`.

### 3. Causal Next-POI fine-tuning

```bash
python -m src.finetune \
  --cache_path cache/nyc_cache.pt \
  --save_dir checkpoints \
  --exp_name planC_finetune_nyc \
  --init_from checkpoints/planC_pretrain_nyc.pt \
  --epochs 60 \
  --batch_size 256 \
  --lr 3e-4 \
  --weight_decay 1e-2 \
  --label_smoothing 0.05 \
  --lambda_cat 0.2 \
  --lambda_region 0.2
```

The best checkpoint is saved as `checkpoints/planC_finetune_nyc.pt`.

### 4. Evaluate

Evaluate the base POI ranking with no inference-time calibration:

```bash
python -m src.eval \
  --cache_path cache/nyc_cache.pt \
  --ckpt_path checkpoints/planC_finetune_nyc.pt \
  --split test \
  --batch_size 256
```

Enable confidence-guided consistency-aware re-ranking by setting `--gamma` to a positive value:

```bash
python -m src.eval \
  --cache_path cache/nyc_cache.pt \
  --ckpt_path checkpoints/planC_finetune_nyc.pt \
  --split test \
  --topk_region 40 \
  --topm_time 1 \
  --gamma 1.0 \
  --conf_pow 1.0 \
  --conf_min 0.0
```

The evaluator reports `Acc@1`, `Acc@5`, `Acc@10`, `Acc@20`, and `MRR`. When re-ranking is enabled, region and temporal boosts are scaled by the normalized entropy-based confidence of the region prediction.


## Reported results

The following numbers are reproduced from the accompanying paper draft. They are provided as reference results; exact reproduction depends on the dataset split, preprocessing, random seed, hardware, and hyperparameters.

| Model | Dataset | Acc@1 | Acc@5 | Acc@10 | Acc@20 | MRR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CORE-POI (without consistency-aware inference) | NYC | 0.3522 | 0.6032 | 0.6895 | 0.7393 | 0.4641 |
| CORE-POI | NYC | **0.3722** | **0.6319** | **0.7171** | **0.7617** | **0.4871** |
| CORE-POI (without consistency-aware inference) | TKY | 0.3699 | 0.6045 | 0.6850 | 0.7463 | 0.4772 |
| CORE-POI | TKY | **0.3819** | **0.6352** | **0.7192** | **0.7778** | **0.4958** |

## Reproducibility notes

- Preprocessing builds spatial grids from the combined split metadata and builds user-conditioned feasibility mappings from the training trajectories only.
- Special tokens are reserved as `PAD=0`, `MASK=1`, and `CLS=2`.
- The default model uses `d_model=256`, 8 attention heads, 4 Transformer layers, and a maximum sequence length of 128.
- Training and evaluation automatically use CUDA when it is available; otherwise they fall back to CPU.
- Generated caches, checkpoints, logs, and local datasets are covered by the provided `.gitignore` and should remain untracked.