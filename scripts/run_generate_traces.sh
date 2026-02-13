#!/bin/bash
set -euo pipefail

# Resolve paths relative to this script so it works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLKIT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_PY="${TOOLKIT_DIR}/trace_eval/generate_traces.py"

# GPU placement
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6,7}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_NO_TORCHVISION=1
export VLLM_ATTENTION_BACKEND=XFORMERS

# Source proxy configuration if available
if [ -f "/datapool/proxy_utils/set_proxy.sh" ]; then
    # shellcheck disable=SC1091
    source /datapool/proxy_utils/set_proxy.sh
fi

# Model configurations
models=("${MODELS_OVERRIDE:-Qwen/Qwen2.5-3B-Instruct}")
dataset_names=("hotpotqa" "nq_search" "2wikimultihopqa" "musique" "triviaqa" "popqa" "bamboogle")

for base_model in "${models[@]}"; do
    echo "Evaluating ${base_model}"
    model_id=$(echo "$base_model" | cut -d'/' -f2)
    if [ "$model_id" = "Qwen2.5-7B-Instruct" ] || [ "$model_id" = "Qwen2.5-3B-Instruct" ]; then
        fewshot=true
    else
        fewshot=false
    fi

    for dataset_name in "${dataset_names[@]}"; do
        echo "Running dataset: ${dataset_name}"
        
        mkdir -p "${TOOLKIT_DIR}/eval_results/${dataset_name}"
        
        data_path="${TOOLKIT_DIR}/data/${dataset_name}/process_test.parquet"
        if [ "$fewshot" = true ]; then
            output_path="${TOOLKIT_DIR}/eval_results/${dataset_name}/final_processed_fewshot-${model_id}.jsonl"
        else
            output_path="${TOOLKIT_DIR}/eval_results/${dataset_name}/final_processed_${model_id}.jsonl"
        fi

        if [ ! -f "${data_path}" ]; then
            echo "Skip ${dataset_name}: missing ${data_path}"
            continue
        fi

        if [ -f "${output_path}" ]; then
            rm "${output_path}"
        fi

        fewshot_arg=""
        if [ "$fewshot" = true ]; then
            fewshot_arg="--fewshot"
        fi

        python "${EVAL_PY}" \
            --model_path "${base_model}" \
            --data_path "${data_path}" \
            --output_path "${output_path}" \
            --do_search \
            --val_batch_size 32 \
            $fewshot_arg

    done
done

echo "All evaluations complete."
