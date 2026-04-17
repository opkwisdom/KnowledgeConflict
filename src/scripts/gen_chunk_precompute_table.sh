#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PWD/src:$PYTHONPATH


torchrun --nproc_per_node=4 src/gen_chunk_precompute_table.py \
    --config config/src/gen_chunk_precompute_table.yaml \
    data.name=odqa \
    data.data_path=data/train/odqa_train_post.jsonl \
    output_file=precompute_table_expanded.h5
# data.name=nq \
# data.data_path=data/nq/parametric_relevance_tagged/validation_with_id.json
    