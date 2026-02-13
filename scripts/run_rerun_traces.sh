#!/bin/bash
set -euo pipefail

# Resolve paths relative to this script so it works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLKIT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RERUN_PY="${TOOLKIT_DIR}/trace_eval/rerun_traces.py"

# GPU configuration
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Experiment configuration
experiment_names=("hotpotqa" "musique" "triviaqa" "popqa" "2wikimultihopqa" "bamboogle" "nq_search")
model_a_list=("ASearcher-Local-7B")
model_b_list=("inclusionAI/ASearcher-Local-7B")

for model_a in "${model_a_list[@]}"; do
    for model_b in "${model_b_list[@]}"; do
        model_b_id=$(echo "$model_b" | cut -d'/' -f2)
        for experiment_name in "${experiment_names[@]}"; do
            echo "Rerunning ${model_a} on ${experiment_name} with ${model_b_id}"
            input_path="${TOOLKIT_DIR}/eval_results/${experiment_name}/final_processed_${model_a}.jsonl"
            output_path="${TOOLKIT_DIR}/eval_results/${experiment_name}/rerun_processed_${model_b_id}-${model_a}.jsonl"

            if [ ! -f "${input_path}" ]; then
                echo "Skip ${experiment_name}: missing ${input_path}"
                continue
            fi

            mkdir -p "$(dirname "${output_path}")"

            python "${RERUN_PY}" \
                "${input_path}" \
                "${output_path}" \
                --model_name ${model_b} \
                --local_inference \
                --local_batch_size 64
        done
    done
done
