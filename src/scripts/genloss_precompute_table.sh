#!/bin/bash

export PYTHONPATH=$PWD/src:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false


### Main train (pseudo-answer)
# GPU 4장에 대해 각각 프로세스 실행
# tmux new-session -d -s genloss

# # 0번 GPU
# tmux rename-window -t genloss:0 'GPU_0'
# tmux send-keys -t genloss:0 "conda activate vllm && CUDA_VISIBLE_DEVICES=0 python src/genloss_precompute_table.py --input_file mhqa_train_half_aa.jsonl --output_file out_0" C-m

# # 1번 GPU
# tmux new-window -t genloss -n 'GPU_1'
# tmux send-keys -t genloss:1 "conda activate vllm && CUDA_VISIBLE_DEVICES=1 python src/genloss_precompute_table.py --input_file mhqa_train_half_ab.jsonl --output_file out_1" C-m

# # 2번 GPU (동일 반복...)
# tmux new-window -t genloss -n 'GPU_2'
# tmux send-keys -t genloss:2 "conda activate vllm && CUDA_VISIBLE_DEVICES=2 python src/genloss_precompute_table.py --input_file mhqa_train_half_ac.jsonl --output_file out_2" C-m

# # 3번 GPU
# tmux new-window -t genloss -n 'GPU_3'
# tmux send-keys -t genloss:3 "conda activate vllm && CUDA_VISIBLE_DEVICES=3 python src/genloss_precompute_table.py --input_file mhqa_train_half_ad.jsonl --output_file out_3" C-m

# tmux attach-session -t genloss


### Main train (short answer)
# GPU 4장에 대해 각각 프로세스 실행
tmux new-session -d -s genloss

# 0번 GPU
tmux rename-window -t genloss:0 'GPU_0'
tmux send-keys -t genloss:0 "conda activate vllm && CUDA_VISIBLE_DEVICES=0 python src/genloss_precompute_table.py --input_file mhqa_train_half_aa.jsonl --output_file short_out_0" C-m

# 1번 GPU
tmux new-window -t genloss -n 'GPU_1'
tmux send-keys -t genloss:1 "conda activate vllm && CUDA_VISIBLE_DEVICES=1 python src/genloss_precompute_table.py --input_file mhqa_train_half_ab.jsonl --output_file short_out_1" C-m

# 2번 GPU (동일 반복...)
tmux new-window -t genloss -n 'GPU_2'
tmux send-keys -t genloss:2 "conda activate vllm && CUDA_VISIBLE_DEVICES=2 python src/genloss_precompute_table.py --input_file mhqa_train_half_ac.jsonl --output_file short_out_2" C-m

# 3번 GPU
tmux new-window -t genloss -n 'GPU_3'
tmux send-keys -t genloss:3 "conda activate vllm && CUDA_VISIBLE_DEVICES=3 python src/genloss_precompute_table.py --input_file mhqa_train_half_ad.jsonl --output_file short_out_3" C-m

tmux attach-session -t genloss

wait
python3 src/merge_h5.py


### HotpotQA val
# tmux new-session -d -s genloss

# # 0번 GPU
# tmux rename-window -t genloss:0 'GPU_0'
# tmux send-keys -t genloss:0 "conda activate vllm && CUDA_VISIBLE_DEVICES=0 python src/genloss_precompute_table.py --input_file val_whole_split00.jsonl --output_file val_whole_out_0" C-m

# # 1번 GPU
# tmux new-window -t genloss -n 'GPU_1'
# tmux send-keys -t genloss:1 "conda activate vllm && CUDA_VISIBLE_DEVICES=1 python src/genloss_precompute_table.py --input_file val_whole_split01.jsonl --output_file val_whole_out_1" C-m

# # 2번 GPU (동일 반복...)
# tmux new-window -t genloss -n 'GPU_2'
# tmux send-keys -t genloss:2 "conda activate vllm && CUDA_VISIBLE_DEVICES=2 python src/genloss_precompute_table.py --input_file val_whole_split02.jsonl --output_file val_whole_out_2" C-m

# # 3번 GPU
# tmux new-window -t genloss -n 'GPU_3'
# tmux send-keys -t genloss:3 "conda activate vllm && CUDA_VISIBLE_DEVICES=3 python src/genloss_precompute_table.py --input_file val_whole_split03.jsonl --output_file val_whole_out_3" C-m

# tmux attach-session -t genloss