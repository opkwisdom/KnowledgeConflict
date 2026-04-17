#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PWD/src:$PYTHONPATH


torchrun --nproc_per_node=1 src/gen_precompute_table.py \
    --config config/src/gen_precompute_table.yaml \
    data.name=nq \
    data.data_path=data/nq/parametric_relevance_tagged/validation_with_id.json
    