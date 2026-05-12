#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PWD/src:$PYTHONPATH

# CONFIG_PATH=$1

# python3 src/run_inference.py \
#     --config config/src/run_inference.yaml


### Run Inference (Original)
# torchrun --nproc_per_node=4 src/run_inference.py \
#     --config config/src/run_inference.yaml \
#     data.data_path=data/hotpotqa-w/retrieved/validation_p_top100_5000.jsonl

### Run Inference (Ablation)
DATA_PATH=data/hotpotqa-w/retrieved/validation_1000r_p_top100.jsonl
USE_CAUSAL=True
CKPT_PATH="/workspaces/kvzip_nlplab/checkpoint/stage3_clf_only_train/multi_LR=2e-05_fromST2_BS=2_ST=None_CMode=split_first_AMode=full_Pool=max_T1=0.5_Alpha=1.0_time=20260506-090239/multi-00-000239-valid_loss=3.8991.ckpt"
# torchrun --nproc_per_node=4 src/run_inference.py \
#     --config config/src/run_inference.yaml \
#     data.data_path=$DATA_PATH \
#     caformer.ckpt_path=$CKPT_PATH \
#     caformer.use_causal=$USE_CAUSAL \
#     experiment_name=ablation_causal

# USE_CAUSAL=False
# CKPT_PATH="/workspaces/kvzip_nlplab/checkpoint/stage3_clf_only_train/multi_LR=2e-05_fromST2_BS=2_ST=None_CMode=split_first_AMode=full_Pool=max_T1=0.5_Alpha=1.0_time=20260506-115621/multi-00-000239-valid_loss=3.8991.ckpt"
# torchrun --nproc_per_node=4 src/run_inference.py \
#     --config config/src/run_inference.yaml \
#     data.data_path=$DATA_PATH \
#     caformer.ckpt_path=$CKPT_PATH \
#     caformer.use_causal=$USE_CAUSAL \
#     experiment_name=ablation_non_causal

# echo "Sleep for 13h 40m"
# sleep 13h 40m

DATA_PATH=data/hotpotqa-w/retrieved/validation_p_top100_5000.jsonl
# CKPT_PATH="/workspaces/kvzip_nlplab/checkpoint/stage3_clf_only_train/multi_LR=2e-05_fromST2_BS=16_Epochs=2_Pool=max_T1=0.5_Alpha=1.0_time=20260506-163449/multi-00-000957-valid_loss=3.7837.ckpt"
CKPT_DIR="/workspaces/kvzip_nlplab/checkpoint/stage3_clf_only_train/multi_LR=2e-05_fromST2_BS=16_Epochs=2_Pool=max_T1=0.5_Alpha=1.0_time=20260510-074401"
USE_CAUSAL=False
# torchrun --nproc_per_node=4 src/run_inference.py \
#     --config config/src/run_inference.yaml \
#     data.data_path=$DATA_PATH \
#     caformer.ckpt_dir=$CKPT_DIR \
#     caformer.use_causal=$USE_CAUSAL \
#     experiment_name=ablation_0.5_5000


DATA_PATH=data/hotpotqa-w/retrieved/validation_1000r_p_top100.jsonl
torchrun --nproc_per_node=1 src/run_inference.py \
    --config config/src/run_inference.yaml \
    data.data_path=$DATA_PATH \
    caformer.ckpt_dir=$CKPT_DIR \
    caformer.use_causal=$USE_CAUSAL \
    experiment_name=ablation_0.5_1000