#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PWD/src:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
# export TORCH_LOGS="recompiles,graph_breaks"

# Base setting
# ALPHAS=(0.0 0.5 1.0)
# for ALPHA in "${ALPHAS[@]}"; do
#     python3 src/run_gg_clf_train.py \
#           --config config/src/train/run_gg_clf_train.yaml \
#           data.name=mhqa \
#           data.data_path=data/train/mhqa_train_test.jsonl \
#           train.max_epochs=1 \
#           train.alpha=$ALPHA \
#           train.gamma=10.0
# done
# Convert to oracle mode

# ALPHAS=(0.0 0.5 1.0)
# ALPHAS=(0.0 1.0)
# CMODES=(random split_first kmeans)
# AMODES=(block full)
# POOLINGS=(mean max)

# CNT=0
# for ALPHA in "${ALPHAS[@]}"; do
#     for CMODE in "${CMODES[@]}"; do
#         for AMODE in "${AMODES[@]}"; do
#             for POOL in "${POOLINGS[@]}"; do
#                 echo "Experiment $CNT"

#                 python3 src/run_gg_clf_train.py \
#                     --config config/src/train/run_gg_clf_train.yaml \
#                     data.name=mhqa \
#                     data.data_path=data/train/mhqa_train_pilot.jsonl \
#                     train.max_epochs=1 \
#                     train.alpha=$ALPHA \
#                     caformer.classifier_mode=$CMODE \
#                     caformer.attention_mode=$AMODE \
#                     caformer.pooling_strategy=$POOL
#                 CNT=$((CNT + 1))
#             done
#         done
#     done
# done


### Combination 1: Rankwise + Max Pooling (Loss scaling 문제)
# ALPHA=1.0
# CMODE=split_first
# AMODE=full
# POOL=max

# python3 src/run_gg_clf_train.py \
#     --config config/src/train/run_gg_clf_train.yaml \
#     data.name=mhqa \
#     data.data_path=data/train/mhqa_train_half_force.jsonl \
#     train.max_epochs=1 \
#     train.alpha=$ALPHA \
#     caformer.classifier_mode=$CMODE \
#     caformer.attention_mode=$AMODE \
#     caformer.pooling_strategy=$POOL


### Combination 2: Listwise + Max Pooling
# ALPHA=1.0
# CMODE=split_first
# AMODE=full
# POOL=max

# python3 src/run_gg_clf_train.py \
#     --config config/src/train/run_gg_clf_train.yaml \
#     data.name=mhqa \
#     data.data_path=data/train/mhqa_train_pilot.jsonl \
#     train.max_epochs=1 \
#     train.alpha=$ALPHA \
#     caformer.classifier_mode=$CMODE \
#     caformer.attention_mode=$AMODE \
#     caformer.pooling_strategy=$POOL \
#     caformer.use_causal=True


### Combination 3: Listwise + Max Pooling + Bidirectional (O)
# ALPHA=1.0
# CMODE=split_first
# AMODE=full
# POOL=max
# USE_CAUSAL=False

# python3 src/run_gg_clf_train.py \
#     --config config/src/train/run_gg_clf_train.yaml \
#     data.name=mhqa \
#     data.data_path=data/train/mhqa_train_pilot.jsonl \
#     train.max_epochs=1 \
#     train.alpha=$ALPHA \
#     caformer.classifier_mode=$CMODE \
#     caformer.attention_mode=$AMODE \
#     caformer.pooling_strategy=$POOL \
#     caformer.use_causal=$USE_CAUSAL


### Combination 4: Listwise + Max Pooling + Multi-epoch
# ALPHA=1.0
# CMODE=split_first
# AMODE=full
# POOL=max
# USE_CAUSAL=False
# EPOCHS=3

# python3 src/run_gg_clf_train.py \
#     --config config/src/train/run_gg_clf_train.yaml \
#     data.name=mhqa \
#     data.data_path=data/train/mhqa_train_pilot.jsonl \
#     train.max_epochs=$EPOCHS \
#     train.alpha=$ALPHA \
#     caformer.classifier_mode=$CMODE \
#     caformer.attention_mode=$AMODE \
#     caformer.pooling_strategy=$POOL \
#     caformer.use_causal=$USE_CAUSAL

### Combination 5: Listwise + Max Pooling + LR Scaling
# ALPHA=1.0
# CMODE=split_first
# AMODE=full
# POOL=max
# USE_CAUSAL=False
# LR=5e-5

# python3 src/run_gg_clf_train.py \
#     --config config/src/train/run_gg_clf_train.yaml \
#     data.name=mhqa \
#     data.data_path=data/train/mhqa_train_pilot.jsonl \
#     train.max_epochs=1 \
#     train.alpha=$ALPHA \
#     train.learning_rate=$LR \
#     caformer.classifier_mode=$CMODE \
#     caformer.attention_mode=$AMODE \
#     caformer.pooling_strategy=$POOL \
#     caformer.use_causal=$USE_CAUSAL



### Full-2
ALPHA=1.0
CMODE=split_first
AMODE=full
POOL=max
USE_CAUSAL=False
EPOCHS=2

python3 src/run_gg_clf_train.py \
    --config config/src/train/run_gg_clf_train.yaml \
    data.name=mhqa \
    data.data_path=data/train/mhqa_train_half_force.jsonl \
    train.max_epochs=$EPOCHS \
    train.alpha=$ALPHA \
    caformer.classifier_mode=$CMODE \
    caformer.attention_mode=$AMODE \
    caformer.pooling_strategy=$POOL \
    caformer.use_causal=$USE_CAUSAL




# ### Combination 2: Pointwise + Mean Pooling
# ALPHA=0.0
# CMODE=split_first
# AMODE=full
# POOL=mean

# python3 src/run_gg_clf_train.py \
#     --config config/src/train/run_gg_clf_train.yaml \
#     data.name=mhqa \
#     data.data_path=data/train/mhqa_train_half_force.jsonl \
#     train.max_epochs=1 \
#     train.alpha=$ALPHA \
#     caformer.classifier_mode=$CMODE \
#     caformer.attention_mode=$AMODE \
#     caformer.pooling_strategy=$POOL


# ### Combination 3: Mixing (1:1) + Mean Pooling
# ALPHA=0.5
# CMODE=split_first
# AMODE=full
# POOL=max

# python3 src/run_gg_clf_train.py \
#     --config config/src/train/run_gg_clf_train.yaml \
#     data.name=mhqa \
#     data.data_path=data/train/mhqa_train_half_force.jsonl \
#     train.max_epochs=1 \
#     train.alpha=$ALPHA \
#     caformer.classifier_mode=$CMODE \
#     caformer.attention_mode=$AMODE \
#     caformer.pooling_strategy=$POOL

### Combination 4: Mixing (1:1) + Mean Pooling




# Debugging test
# for CMODE in "${CMODES[@]}"; do
#     echo "Debugging with Classifier Mode: $CMODE"
#     python3 src/run_gg_clf_train.py \
#         --config config/src/train/run_gg_clf_train.yaml \
#         data.name=mhqa \
#         data.data_path=data/train/mhqa_train_test.jsonl \
#         train.max_epochs=1 \
#         caformer.classifier_mode=$CMODE
# done

# for AMODE in "${AMODES[@]}"; do
#     echo "Debugging with Attention Mode: $AMODE"
#     python3 src/run_gg_clf_train.py \
#         --config config/src/train/run_gg_clf_train.yaml \
#         data.name=mhqa \
#         data.data_path=data/train/mhqa_train_test.jsonl \
#         train.max_epochs=1 \
#         caformer.attention_mode=$AMODE
# done

# for POOL in "${POOLINGS[@]}"; do
#     echo "Debugging with Pooling Strategy: $POOL"
#     python3 src/run_gg_clf_train.py \
#         --config config/src/train/run_gg_clf_train.yaml \
#         data.name=mhqa \
#         data.data_path=data/train/mhqa_train_test.jsonl \
#         train.max_epochs=1 \
#         caformer.pooling_strategy=$POOL
# done