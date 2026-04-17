import h5py
import os

paths = [
    "/workspaces/kvzip_nlplab/checkpoint/gen_precompute_table/llama/precompute_table_test.h5_rank0.tmp",
    "/workspaces/kvzip_nlplab/checkpoint/gen_precompute_table/llama/precompute_table_test.h5_rank1.tmp",
    "/workspaces/kvzip_nlplab/checkpoint/gen_precompute_table/llama/precompute_table_test.h5_rank2.tmp"
]

for i, path in enumerate(paths):
    print(f"\n🔍 [Rank {i}] 파일 검사 중...")
    
    if not os.path.exists(path):
        print(f"❓ 파일이 존재하지 않습니다.")
        continue

    try:
        # 안전하게 하나씩 열어봅니다.
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            print(f"✅ 정상! 총 {len(keys)}개의 ID가 안전하게 저장되어 있습니다.")
            
            if len(keys) > 0:
                first_id = keys[0]
                available_scores = list(f[first_id].keys())
                print(f"   -> [샘플] ID '{first_id}' 내부 구조: {available_scores}")
                
                # 저장된 배열의 크기 살짝 확인
                sample_score = available_scores[0]
                print(f"   -> [샘플] '{sample_score}' shape: {f[first_id][sample_score].shape}")
                
    except Exception as e:
        print(f"💀 파일이 완전히 손상되었습니다 (삭제 필요): {e}")