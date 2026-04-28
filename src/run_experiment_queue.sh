#!/bin/bash
# =============================================================================
# Master Experiment Queue with GPU Monitoring
# =============================================================================
# Runs revision experiments in priority order, automatically dispatching
# to GPUs as they become available.
#
# Priority order (highest first):
#   A. Chunk-size sweep     [RUNNING on tmux: sweep0, sweep2]
#   B. Leave-one-out        [GPU: 1 per model]
#   C. Non-contiguous       [GPU: 1 per model]
#   D. Gradient/saliency    [GPU: 1 per model]
#   E. Cross-task transfer  [GPU: 1 per model]
#   F. OOD robustness       [GPU: 1 per model]
#   G. Retrieval cost       [GPU: 1 per model]
#   H. Regression analysis  [CPU only]
#   I. Random variance      [CPU only]
#   J. PCA baseline         [GPU: 1 per model]
#
# Usage:
#   bash src/run_experiment_queue.sh              # Auto-dispatch to free GPUs
#   bash src/run_experiment_queue.sh --dry-run     # Show what would run
#   bash src/run_experiment_queue.sh --gpu 0       # Only use GPU 0
# =============================================================================

set -euo pipefail

PYTHON="/home/linkco/anaconda3/envs/llama/bin/python"
PROJECT_DIR="/home/linkco/exa/llm-usefulEeb"
LOG_DIR="${PROJECT_DIR}/logs"
RESULT_DIR="${PROJECT_DIR}/data/experiment_results"

mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

# ---- Configuration ----
GPUS=(0 1 2)
DRY_RUN=false
SINGLE_GPU=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --gpu) SINGLE_GPU="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -n "${SINGLE_GPU}" ]]; then
    GPUS=("${SINGLE_GPU}")
fi

# ---- GPU Monitoring ----

get_gpu_memory_free() {
    # Returns free memory in MB for GPU $1
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '
}

get_gpu_util() {
    # Returns GPU utilization % for GPU $1
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '
}

is_gpu_free() {
    # A GPU is "free" if > 15GB free (enough to load a model)
    local gpu_id="$1"
    local free_mb
    free_mb=$(get_gpu_memory_free "$gpu_id")

    if [[ -z "${free_mb}" ]]; then
        return 1  # Can't query, assume busy
    fi

    # Free > 15000MB means enough room for our models (~1-2GB)
    if [[ "${free_mb}" -gt 15000 ]]; then
        return 0
    else
        return 1
    fi
}

wait_for_free_gpu() {
    # Wait until at least one GPU is free, print its ID
    while true; do
        for gpu_id in "${GPUS[@]}"; do
            if is_gpu_free "${gpu_id}"; then
                echo "${gpu_id}"
                return
            fi
        done
        echo "[$(date +%H:%M:%S)] All GPUs busy, waiting 60s..."
        sleep 60
    done
}

# ---- Experiment Definitions ----
# Each experiment: name, type (gpu/cpu), command template

declare -a EXPERIMENTS=()

# B. Leave-one-out importance (4 models, fast with pre-computed embeddings)
for model in gte-large-en-v1.5 stella_en_400M_v5 roberta-large roberta-large-InBedder; do
    EXPERIMENTS+=("B_loo_${model}|gpu|${PYTHON} ${PROJECT_DIR}/src/leave_one_out_fast.py --models ${model} --device cuda:GPU")
done

# J. PCA baseline (5 models, fast with pre-computed embeddings)
for model in gte-large-en-v1.5 stella_en_400M_v5 bge-m3 roberta-large roberta-large-InBedder; do
    EXPERIMENTS+=("J_pca_${model}|gpu|${PYTHON} ${PROJECT_DIR}/src/pca_baseline_fast.py --models ${model} --device cuda:GPU")
done

# D. Gradient/saliency baselines (5 models, fast with pre-computed embeddings)
for model in gte-large-en-v1.5 stella_en_400M_v5 roberta-large roberta-large-InBedder bge-m3; do
    EXPERIMENTS+=("D_grad_${model}|gpu|${PYTHON} ${PROJECT_DIR}/src/gradient_saliency_fast.py --models ${model} --device cuda:GPU")
done

# C. Non-contiguous selection (4 models, fast with pre-computed embeddings)
for model in gte-large-en-v1.5 stella_en_400M_v5 roberta-large roberta-large-InBedder; do
    EXPERIMENTS+=("C_noncont_${model}|gpu|${PYTHON} ${PROJECT_DIR}/src/non_contiguous_fast.py --models ${model} --device cuda:GPU")
done

# G. Retrieval cost analysis (4 models, fast with pre-computed embeddings)
for model in gte-large-en-v1.5 stella_en_400M_v5 bge-m3 roberta-large-InBedder; do
    EXPERIMENTS+=("G_retrieval_${model}|gpu|${PYTHON} ${PROJECT_DIR}/src/retrieval_cost_analysis_fast.py --models ${model} --device cuda:GPU")
done

# F. OOD robustness (4 models, fast with pre-computed embeddings)
for model in gte-large-en-v1.5 bge-m3 stella_en_400M_v5 roberta-large-InBedder; do
    EXPERIMENTS+=("F_ood_${model}|gpu|${PYTHON} ${PROJECT_DIR}/src/ood_robustness_fast.py --models ${model} --device cuda:GPU")
done

# E. Cross-task transfer (5 models, fast with pre-computed embeddings)
for model in gte-large-en-v1.5 stella_en_400M_v5 bge-m3 roberta-large roberta-large-InBedder; do
    EXPERIMENTS+=("E_transfer_${model}|gpu|${PYTHON} ${PROJECT_DIR}/src/cross_task_transfer_fast.py --models ${model} --device cuda:GPU")
done

# H. Training paradigm regression (CPU only, ~1 min)
EXPERIMENTS+=("H_regression|cpu|${PYTHON} ${PROJECT_DIR}/src/training_paradigm_regression.py")

# I. Random variance / tail risk (CPU only, ~1 min)
EXPERIMENTS+=("I_variance|cpu|${PYTHON} ${PROJECT_DIR}/src/random_variance_tail_risk.py --n-seeds 100")

# ---- Dispatcher ----

log() {
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $*"
}

dispatch_experiment() {
    local exp_string="$1"
    local gpu_id="$2"

    local name="${exp_string%%|*}"
    local rest="${exp_string#*|}"
    local exp_type="${rest%%|*}"
    local cmd="${rest#*|}"

    # Replace GPU placeholder — always use cuda:0 since CUDA_VISIBLE_DEVICES selects the physical GPU
    cmd="${cmd//GPU/0}"

    local log_file="${LOG_DIR}/${name}.log"

    # Check if already completed (result file exists)
    case "${name}" in
        B_loo_*) result_file="${RESULT_DIR}/leave_one_out_${name#B_loo_}.json" ;;
        J_pca_*) result_file="${RESULT_DIR}/pca_baseline_${name#J_pca_}.json" ;;
        D_grad_*) result_file="${RESULT_DIR}/gradient_saliency_${name#D_grad_}.json" ;;
        C_noncont_*) result_file="${RESULT_DIR}/non_contiguous_${name#C_noncont_}.json" ;;
        G_retrieval_*) result_file="${RESULT_DIR}/retrieval_cost_${name#G_retrieval_}.json" ;;
        F_ood_*) result_file="${RESULT_DIR}/ood_robustness_${name#F_ood_}.json" ;;
        E_transfer_*) result_file="${RESULT_DIR}/cross_task_transfer_${name#E_transfer_}.json" ;;
        H_regression) result_file="${RESULT_DIR}/training_paradigm_regression.json" ;;
        I_variance) result_file="${RESULT_DIR}/random_variance_tail_risk.json" ;;
        *) result_file="" ;;
    esac

    if [[ -n "${result_file}" && -f "${result_file}" ]]; then
        log "SKIP ${name} (result already exists: ${result_file})"
        return 1
    fi

    if [[ "${DRY_RUN}" == "true" ]]; then
        log "WOULD RUN: ${name} on GPU ${gpu_id}"
        log "  Command: ${cmd}"
        log "  Log: ${log_file}"
        return 0
    fi

    log "DISPATCH: ${name} on GPU ${gpu_id}"
    log "  Command: ${cmd}"
    log "  Log: ${log_file}"

    # Run in background, redirect to log
    CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 nohup ${cmd} > "${log_file}" 2>&1 &
    local pid=$!
    log "  PID: ${pid}"

    # Give it a few seconds to start and allocate GPU memory
    sleep 5

    return 0
}

run_cpu_experiment() {
    local exp_string="$1"
    local name="${exp_string%%|*}"
    local rest="${exp_string#*|}"
    local cmd="${rest#*|}"
    local log_file="${LOG_DIR}/${name}.log"

    # Check if already done
    case "${name}" in
        H_regression) result_file="${RESULT_DIR}/training_paradigm_regression.json" ;;
        I_variance) result_file="${RESULT_DIR}/random_variance_tail_risk.json" ;;
        *) result_file="" ;;
    esac

    if [[ -n "${result_file}" && -f "${result_file}" ]]; then
        log "SKIP ${name} (result already exists)"
        return 0
    fi

    if [[ "${DRY_RUN}" == "true" ]]; then
        log "WOULD RUN (CPU): ${name}"
        log "  Command: ${cmd}"
        return 0
    fi

    log "RUN (CPU): ${name}"
    cd "${PROJECT_DIR}" && ${cmd} > "${log_file}" 2>&1
    log "DONE (CPU): ${name}"
}

# ---- Status Check ----

show_status() {
    log "===== Experiment Queue Status ====="
    log ""

    local total=0
    local completed=0
    local pending=0

    for exp in "${EXPERIMENTS[@]}"; do
        total=$((total + 1))
        local name="${exp%%|*}"
        local rest="${exp#*|}"
        local exp_type="${rest%%|*}"

        # Check log existence and completion
        local log_file="${LOG_DIR}/${name}.log"
        if [[ -f "${log_file}" ]]; then
            if grep -q "Results saved" "${log_file}" 2>/dev/null; then
                log "  [DONE] ${name}"
                completed=$((completed + 1))
            else
                log "  [RUNNING?] ${name}"
                pending=$((pending + 1))
            fi
        else
            log "  [PENDING] ${name}"
            pending=$((pending + 1))
        fi
    done

    log ""
    log "Total: ${total} | Completed: ${completed} | Pending: ${pending}"

    # GPU status
    log ""
    log "GPU Status:"
    for gpu_id in "${GPUS[@]}"; do
        local free_mb util
        free_mb=$(get_gpu_memory_free "${gpu_id}")
        util=$(get_gpu_util "${gpu_id}")
        log "  GPU ${gpu_id}: ${free_mb}MB free, ${util}% util"
    done
}

# ---- Main ----

main() {
    log "Starting experiment queue dispatcher"
    log "GPUs: ${GPUS[*]}"

    # 1. Run CPU-only experiments first (fast, no GPU needed)
    log ""
    log "--- Phase 1: CPU-only experiments ---"
    for exp in "${EXPERIMENTS[@]}"; do
        local exp_type="${exp#*|}"
        exp_type="${exp_type%%|*}"
        if [[ "${exp_type}" == "cpu" ]]; then
            run_cpu_experiment "${exp}"
        fi
    done

    # 2. Check status and dispatch GPU experiments
    log ""
    log "--- Phase 2: GPU experiments ---"

    if [[ "${DRY_RUN}" == "true" ]]; then
        for exp in "${EXPERIMENTS[@]}"; do
            local exp_type="${exp#*|}"
            exp_type="${exp_type%%|*}"
            if [[ "${exp_type}" == "gpu" ]]; then
                local gpu_id
                gpu_id=$(wait_for_free_gpu)
                dispatch_experiment "${exp}" "${gpu_id}"
            fi
        done
    else
        # Track running jobs
        declare -A gpu_running  # gpu_id -> "exp_name:pid"
        local exp_idx=0
        local gpu_experiments=()

        # Filter GPU experiments
        for exp in "${EXPERIMENTS[@]}"; do
            local exp_type="${exp#*|}"
            exp_type="${exp_type%%|*}"
            if [[ "${exp_type}" == "gpu" ]]; then
                gpu_experiments+=("${exp}")
            fi
        done

        log "GPU experiments to run: ${#gpu_experiments[@]}"

        while [[ ${exp_idx} -lt ${#gpu_experiments[@]} ]]; do
            # Check for completed jobs and free GPUs
            for gpu_id in "${GPUS[@]}"; do
                if [[ -n "${gpu_running[${gpu_id}]:-}" ]]; then
                    local pid="${gpu_running[${gpu_id}]#*:}"
                    if ! kill -0 "${pid}" 2>/dev/null; then
                        local finished_name="${gpu_running[${gpu_id}]%%:*}"
                        log "COMPLETED: ${finished_name} on GPU ${gpu_id}"
                        unset gpu_running[${gpu_id}]
                    fi
                fi
            done

            # Find free GPU
            local free_gpu=""
            for gpu_id in "${GPUS[@]}"; do
                if [[ -z "${gpu_running[${gpu_id}]:-}" ]] && is_gpu_free "${gpu_id}"; then
                    free_gpu="${gpu_id}"
                    break
                fi
            done

            if [[ -n "${free_gpu}" ]]; then
                local exp="${gpu_experiments[${exp_idx}]}"
                local name="${exp%%|*}"

                if dispatch_experiment "${exp}" "${free_gpu}"; then
                    # Find the PID we just launched
                    local pid=$!
                    gpu_running["${free_gpu}"]="${name}:${pid}"
                fi
                exp_idx=$((exp_idx + 1))
            else
                # No free GPU, wait
                sleep 30
            fi
        done

        # Wait for all remaining jobs
        log "All experiments dispatched. Waiting for completion..."
        while true; do
            local any_running=false
            for gpu_id in "${GPUS[@]}"; do
                if [[ -n "${gpu_running[${gpu_id}]:-}" ]]; then
                    local pid="${gpu_running[${gpu_id}]#*:}"
                    if kill -0 "${pid}" 2>/dev/null; then
                        any_running=true
                    else
                        local finished_name="${gpu_running[${gpu_id}]%%:*}"
                        log "COMPLETED: ${finished_name} on GPU ${gpu_id}"
                        unset gpu_running[${gpu_id}]
                    fi
                fi
            done

            if [[ "${any_running}" == "false" ]]; then
                break
            fi
            sleep 30
        done
    fi

    # 3. Final status
    log ""
    log "--- All experiments complete ---"
    show_status
}

# If called with --status, just show status
if [[ $# -gt 0 && "$1" == "--status" ]]; then
    show_status
    exit 0
fi

main "$@"
