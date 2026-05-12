import json
import os
from scipy.stats import spearmanr

ret_path = "/workspaces/kvzip_nlplab/DISCA/results/rag/Meta-Llama-3-8B-Instruct/hotpotqa-w/prompt=base_bge-reranker-base/inference_results.json"
# gen_path1 = "/workspaces/kvzip_nlplab/DISCA/results/disca/Alpha=0.5/inference_results.json"
gen_path1 = "/workspaces/kvzip_nlplab/DISCA/results/rag/Meta-Llama-3-8B-Instruct/hotpotqa-w/prompt=base_ms-marco-MiniLM-L-6-v2/inference_results.json"
# gen_path1 = "/workspaces/kvzip_nlplab/DISCA/results/disca/ablation_0.5_5000/inference_results.json"
# gen_path2 = "/workspaces/kvzip_nlplab/DISCA/results/oracle_loss_gen_test/hotpotqa-w/prompt=base/20260428_153901/inference_results.json"

# No Sys
gen_path2 = "/workspaces/kvzip_nlplab/DISCA/results/oracle_loss_gen_test/hotpotqa-w/No_Sys_5000/inference_results.json"
# Sys
gen_path3 = "/workspaces/kvzip_nlplab/DISCA/results/oracle_loss_gen_test/hotpotqa-w/Sys_5000/inference_results.json"


with open(ret_path, 'r') as f:
    ret_results = json.load(f)

with open(gen_path1, 'r') as f:
    gen_results_1 = json.load(f)

with open(gen_path2, 'r') as f:
    gen_results_2 = json.load(f)

with open(gen_path3, 'r') as f:
    gen_results_3 = json.load(f)



def extract_metrics(ret_results, gen_results):
    ret_over_gen = []
    gen_over_ret = []
    overlap = []
    fail = []
    for ret, gen in zip(ret_results, gen_results):
        ret_metrics = ret['metrics']["soft_em"]
        gen_metrics = gen['metrics']["soft_em"]
        
        if ret_metrics and not gen_metrics:
            ret_over_gen.append(ret['id'])
        elif gen_metrics and not ret_metrics:
            gen_over_ret.append(gen['id'])
        elif ret_metrics and gen_metrics:
            overlap.append(ret['id'])
        else:
            fail.append(ret['id'])

    total_len = len(ret_results)

    print(f"Retrieval better than Generation: {len(ret_over_gen)} ({len(ret_over_gen) / total_len * 100:.2f}%)")
    print(f"Generation better than Retrieval: {len(gen_over_ret)} ({len(gen_over_ret) / total_len * 100:.2f}%)")
    print(f"Both correct: {len(overlap)} ({len(overlap) / total_len * 100:.2f}%)")
    print(f"Both incorrect: {len(fail)} ({len(fail) / total_len * 100:.2f}%)")
    theoretical_best = total_len - len(fail)
    print(f"\nTheoretical Best: {theoretical_best / total_len * 100:.2f}%\n")

    return ret_over_gen, gen_over_ret, overlap, fail

def save_overlap_results(ret_over_gen, gen_over_ret, overlap, fail, output_path):
    with open(output_path, 'w') as f:
        json.dump({
            "retrieval_better": ret_over_gen,
            "generation_better": gen_over_ret,
            "both_correct": overlap,
            "both_incorrect": fail
        }, f, indent=4)


ret_over_gen_1, gen_over_ret_1, overlap_1, fail_1 = extract_metrics(ret_results, gen_results_1)
ret_over_gen_2, gen_over_ret_2, overlap_2, fail_2 = extract_metrics(ret_results, gen_results_2)
ret_over_gen_3, gen_over_ret_3, overlap_3, fail_3 = extract_metrics(ret_results, gen_results_3)

base_output_dir = "/workspaces/kvzip_nlplab/DISCA/src/tests/ret_vs_gen_overlap"
os.makedirs(base_output_dir, exist_ok=True)
save_overlap_results(ret_over_gen_1, gen_over_ret_1, overlap_1, fail_1, f"{base_output_dir}/ret_vs_gen_overlap.json")
save_overlap_results(ret_over_gen_2, gen_over_ret_2, overlap_2, fail_2, f"{base_output_dir}/oracle_ret_vs_gen_overlap.json")
save_overlap_results(ret_over_gen_3, gen_over_ret_3, overlap_3, fail_3, f"{base_output_dir}/oracle_sys_ret_vs_gen_overlap.json")