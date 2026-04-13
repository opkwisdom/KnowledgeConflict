import h5py
import os
from tqdm import tqdm

# 경로 설정 (사용하시던 경로 그대로입니다)
output_path = "/workspaces/kvzip_nlplab/checkpoint/gen_precompute_table/llama/precompute_table_test.h5"

print("🚀 안전한 HDF5 병합(Merge)을 시작합니다...")

with h5py.File(output_path, "w") as final_h5:
    for rank in range(3): # 3개의 GPU Rank
        temp_path = f"{output_path}_rank{rank}.tmp"
        
        if not os.path.exists(temp_path):
            print(f"⚠️ {temp_path} 파일이 없습니다. 스킵합니다.")
            continue
        
        with h5py.File(temp_path, "r") as temp_h5:
            keys = list(temp_h5.keys())
            
            for group_name in tqdm(keys, desc=f"Merging Rank {rank}"):
                # ✨ 핵심 해결책: 이미 복사된 ID라면 충돌을 피해 부드럽게 넘어갑니다.
                if group_name not in final_h5:
                    temp_h5.copy(group_name, final_h5)

    print(f"final output length: {len(list(final_h5.keys()))}")
print(f"\n🎉 병합 대성공! 최종 파일이 완성되었습니다: {output_path}")