import json

ref_path = 'results/main/e2e_openai/ratio=0.3_prompt=base/20260119_041439/inference_results.json'
target_path = 'results/main/e2e_openai/ratio=0.3_prompt=base/20260119_063144/inference_results.json'

target_col = "param_negative"
with open(ref_path, 'r') as f:
    ref_data = json.load(f)[target_col]

with open(target_path, 'r') as f:
    target_data = json.load(f)[target_col]

### Compare
ref_t_target_f = 0
ref_f_target_t = 0

for ref, target in zip(ref_data, target_data):
    if ref["metrics"]["soft_em"] and not target["metrics"]["soft_em"]:
        print("===== Ref=True, Target=False =====")
        print(f'Sample id: {ref["id"]}')
        print(f'Question: {ref["question"]}')
        print(f'Ref Prediction: {ref["pred_answer"]}')
        print(f'Target Prediction: {target["pred_answer"]}')
        print(f'Answer: {", ".join(ref["answers"])}\n')
        ref_t_target_f += 1

    if not ref["metrics"]["soft_em"] and target["metrics"]["soft_em"]:
        print("===== Ref=False, Target=True =====")
        print(f'Sample id: {ref["id"]}')
        print(f'Question: {ref["question"]}')
        print(f'Ref Prediction: {ref["pred_answer"]}')
        print(f'Target Prediction: {target["pred_answer"]}')
        print(f'Answer: {", ".join(ref["answers"])}\n')
        ref_f_target_t += 1

print(f"Ref Win: {ref_t_target_f}")
print(f"Target Win: {ref_f_target_t}")