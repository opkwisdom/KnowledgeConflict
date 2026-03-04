#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PWD/src:$PYTHONPATH

CONFIG_PATH=$1

### Hyperparameter search space ###
LAYERS=(16 20 24)
CTR_WEIGHTS=(1.0 0.5 0.0)
LEARNING_RATES=(1e-5 2e-5 5e-5)
POOLING_STRATEGIES=(mean max)
QUERY_LENGTHS=(4 8 16 32)


# Step 1: CTR loss influence & Learning rate investigation
echo "Starting Step 1: Core Hyperparams Investigation at $(date)"
for lr in "${LEARNING_RATES[@]}"; do
    for weight in "${CTR_WEIGHTS[@]}"; do
        echo "Running with LR=$lr, weight=$weight, layer=20"
        
        python3 src/run_train.py \
            --config $CONFIG_PATH \
            train.learning_rate=$lr \
            train.ctr_loss_weight=$weight \
            exp_type="search_LR=${lr}_CTR-W=${weight}"
    done
done
echo "Finished Step 1: Core Hyperparams Investigation at $(date)"

# Step 2: Layer & Pooling strategy investigation
echo "Starting Step 2: Layer & Pooling strategy investigation at $(date)"
for layer in "${LAYERS[@]}"; do
    for pooling in "${POOLING_STRATEGIES[@]}"; do
        echo "Running with layer=$layer, pooling=$pooling"

        python3 src/run_train.py \
            --config $CONFIG_PATH \
            model.hidden_extraction_layers="[$layer]" \
            caformer.pooling_strategy="$pooling" \
            exp_type="search_L=${layer}_Pool=${pooling}"
    done
done
echo "Finished Step 2: Layer & Pooling strategy investigation at $(date)"

# Step 3: Query length investigation
echo "Starting Step 3: Query length investigation at $(date)"
for query_length in "${QUERY_LENGTHS[@]}"; do
    echo "Running with query_length=$query_length"

    python3 src/run_train.py \
        --config $CONFIG_PATH \
        caformer.query_length=$query_length \
        exp_type="search_Qsize=${query_length}"
done
echo "Finished Step 3: Query length investigation at $(date)"



### Grid search ###
# for pooling in "${POOLING_STRATEGIES[@]}"; do
#     for layer in "${LAYERS[@]}"; do
#         for query_length in "${QUERY_LENGTHS[@]}"; do
#             for weight in "${CTR_WEIGHTS[@]}"; do
#                 echo "Running with pooling=$pooling, layer=$layer, ctr_loss_weight=$weight"

#                 python3 src/run_train.py \
#                     --config $CONFIG_PATH \
#                     model.hidden_extraction_layers="[$layer]" \
#                     caformer.query_length=$query_length \
#                     caformer.pooling_strategy="$pooling" \
#                     train.ctr_loss_weight=$weight
#             done
#         done
#     done
# done