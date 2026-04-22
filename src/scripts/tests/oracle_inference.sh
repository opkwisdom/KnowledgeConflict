#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=$PWD/src:$PYTHONPATH

# echo "Sleep for 1h to wait for the precompute tables to be generated..."
# sleep 1h

ORACLE_MODE=base
TOPK_PER_QUERIES=(100 50 30 20)
for TOPK in "${TOPK_PER_QUERIES[@]}"; do
    echo "Running oracle inference for top-$TOPK passages per query..."
    python3 src/tests/oracle_inference.py \
        --config config/src/tests/oracle_inference.yaml \
        data.score_mode=oracle \
        data.precompute_table_path=/workspaces/kvzip_nlplab/checkpoint/gen_emb_precompute_table/llama/hotpotqa-w_val_1000_emb_precompute_table_${TOPK}_${ORACLE_MODE}.h5
done

# echo "Running oracle inference for top-20 passages per query... (Sleep for 20 minutes)"
# sleep 20m
# python3 src/tests/oracle_inference.py \
#     --config config/src/tests/oracle_inference.yaml \
#     data.score_mode=oracle \
#     data.precompute_table_path=/workspaces/kvzip_nlplab/checkpoint/gen_emb_precompute_table/llama/nq_val_emb_precompute_table_20.h5