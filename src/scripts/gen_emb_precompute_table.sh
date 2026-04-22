#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PWD/src:$PYTHONPATH

ORACLE_MODES=(base isolated)
TOPK_PER_QUERIES=(100 50 30 20)
for ORACLE_MODE in "${ORACLE_MODES[@]}"; do
    for TOPK in "${TOPK_PER_QUERIES[@]}"; do
        echo "Generating precompute table for top-$TOPK passages with $ORACLE_MODE per query..."
        torchrun --nproc_per_node=3 src/gen_emb_precompute_table.py \
            --config config/src/gen_emb_precompute_table.yaml \
            data.topk_per_query=$TOPK \
            oracle_mode=$ORACLE_MODE \
            data.precompute_table_path=/workspaces/kvzip_nlplab/checkpoint/gen_emb_precompute_table/llama/hotpotqa-w_val_1000_emb_precompute_table_${TOPK}_${ORACLE_MODE}.h5
            # data.name=nq \
            # data.data_path=data/nq/retrieved/validation_top100_with_id.jsonl \
    done
done

# TOPK=20
# ORACLE_MODE=isolated

# echo "Generating precompute table for top-$TOPK passages per query..."
# torchrun --nproc_per_node=1 src/gen_emb_precompute_table.py \
#     --config config/src/gen_emb_precompute_table.yaml \
#     data.name=nq \
#     data.data_path=data/nq/retrieved/validation_top100_with_id.jsonl \
#     data.topk_per_query=$TOPK \
#     oracle_mode=$ORACLE_MODE \
#     data.precompute_table_path=/workspaces/kvzip_nlplab/checkpoint/gen_emb_precompute_table/llama/nq_val_emb_precompute_table_${TOPK}_${ORACLE_MODE}.h5