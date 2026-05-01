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
ALPHAS=(0.0 1.0)
CMODES=(random split_first kmeans)
AMODES=(block full)
POOLINGS=(mean max)

CNT=0
for ALPHA in "${ALPHAS[@]}"; do
    for CMODE in "${CMODES[@]}"; do
        for AMODE in "${AMODES[@]}"; do
            for POOL in "${POOLINGS[@]}"; do
                echo "Experiment $CNT"

                python3 src/run_gg_clf_train.py \
                    --config config/src/train/run_gg_clf_train.yaml \
                    data.name=mhqa \
                    data.data_path=data/train/mhqa_train_subset_force.jsonl \
                    train.max_epochs=1 \
                    train.alpha=$ALPHA \
                    caformer.classifier_mode=$CMODE \
                    caformer.attention_mode=$AMODE \
                    caformer.pooling_strategy=$POOL
                CNT=$((CNT + 1))
            done
        done
    done
done


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