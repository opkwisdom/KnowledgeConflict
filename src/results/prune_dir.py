from pathlib import Path
import os

def dry_run_cleanup(target_dir):
    print(f"🔍 탐색 시작: {target_dir}\n")
    
    # topdown=False: 제일 깊은 폴더부터 위로 올라오면서 검사 (필수)
    for root, dirs, files in os.walk(target_dir, topdown=False):
        
        # 1. 해당 폴더에 .log가 아닌 파일이 하나라도 있는지 확인
        has_non_log_file = False
        for file in files:
            if not file.endswith('.log'):
                has_non_log_file = True
                break
        
        # 2. 해당 폴더에 (삭제되지 않고 살아남은) 하위 폴더가 있는지 확인
        # topdown=False이므로, 앞선 루프에서 하위 폴더가 삭제되었다면 dirs 리스트는 비어있거나 줄어들어 있음
        has_subdirs = len(dirs) > 0
        
        # 3. .log 파일만 있거나, 아예 빈 폴더이고 + 하위 폴더도 없으면 삭제 대상
        if not has_non_log_file and not has_subdirs:
            print(f"🗑️ [삭제 예정] {root}")

def clean_log_only_directories(target_dir):
    deleted_count = 0
    
    # 상향식 탐색 (깊은 곳 -> 얕은 곳)
    for root, dirs, files in os.walk(target_dir, topdown=False):
        
        # 1. 중요 파일(.log 아닌 것) 체크
        has_non_log_file = False
        for file in files:
            if not file.endswith('.log'):
                has_non_log_file = True
                break
        
        # 2. 하위 디렉토리 존재 여부 체크
        has_subdirs = len(dirs) > 0
        
        # 3. 삭제 조건: 중요 파일 없음 AND 하위 폴더 없음
        if not has_non_log_file and not has_subdirs:
            try:
                # 내부의 .log 파일들 먼저 삭제
                for f in files:
                    os.remove(os.path.join(root, f))
                
                # 비어있는 디렉토리 삭제
                os.rmdir(root)
                print(f"✅ Deleted: {root}")
                deleted_count += 1
                
            except Exception as e:
                print(f"⚠️ Error deleting {root}: {e}")

    print(f"\n총 {deleted_count}개의 디렉토리를 삭제했습니다.")

# 사용 예시
target_directory = "."  # 탐색할 경로 입력
dry_run_cleanup(target_directory)
clean_log_only_directories(target_directory)