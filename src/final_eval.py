data = {
    "param_true": {
        "total": 59,
        "correct": 36,
        "accuracy": 0.6102,
        "recall": 0.7291,
        "precision": 0.5847,
        "f1": 0.6142
    },
    "param_positive": {
        "total": 13,
        "correct": 9,
        "accuracy": 0.6923,
        "recall": 0.7692,
        "precision": 0.6467,
        "f1": 0.6388
    },
    "param_negative": {
        "total": 16,
        "correct": 5,
        "accuracy": 0.3125,
        "recall": 0.3438,
        "precision": 0.2634,
        "f1": 0.2846
    },
    "param_irrelevant": {
        "total": 12,
        "correct": 1,
        "accuracy": 0.0833,
        "recall": 0.1944,
        "precision": 0.0277,
        "f1": 0.048
    }
}

# 1. 전체 개수 및 가중치 합계를 저장할 변수 초기화
total_samples = 0
weighted_sums = {'accuracy': 0, 'recall': 0, 'precision': 0, 'f1': 0}

# 2. 각 항목을 순회하며 가중 합계 계산 (지표 * 해당 항목의 total)
for key, metrics in data.items():
    count = metrics['total']
    total_samples += count
    
    # 0개인 항목은 계산에서 제외 (0을 곱하므로 결과는 같음)
    for metric_name in weighted_sums:
        weighted_sums[metric_name] += metrics[metric_name] * count

# 3. 최종 평균 계산 (가중 합계 / 전체 개수)
results = {}
if total_samples > 0:
    for metric, value in weighted_sums.items():
        results[metric] = round(value / total_samples, 4) # 소수점 4째자리 반올림

# 4. 결과 출력
print(f"=== 전체 샘플 수: {total_samples} ===")
for metric, avg_value in results.items():
    print(f"Average {metric.capitalize()}: {avg_value}")