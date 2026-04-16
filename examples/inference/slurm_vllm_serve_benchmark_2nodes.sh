#!/bin/bash
#SBATCH --partition=batch
#SBATCH --account=llmservice_fm_text
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --segment=2
#SBATCH --time=03:00:00
#SBATCH --job-name=llmservice-vllm-serve
#SBATCH --output=/lustre/fsw/portfolios/llmservice/users/helenn/logs/vllm/vllm-realdata-flashinfer-2node-tp8ep8dp1-bs8.out
#SBATCH --error=/lustre/fsw/portfolios/llmservice/users/helenn/logs/vllm/vllm-realdata-flashinfer-2node-tp8ep8dp1-bs8.out
#SBATCH --exclusive

# =============================================================================
# Multi-node vLLM serve + benchmark (adapted from working reference script)
#
# Head (rank 0): starts Ray head + vllm serve + benchmark client
# Workers (rank 1+): join Ray cluster + block
#
# Usage:
#   MODEL_PATH=/path/to/hf TP_SIZE=4 DP_SIZE=4 sbatch slurm_vllm_serve_benchmark.sh
# =============================================================================

set -euo pipefail

# --------------- configuration (override via env) ----------------------------
MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nemotron_ultra/nemo_rl/ci/checkpoints/ultra-v3-sft-hsg-mainfeb19merge-mxfp8_fixed-hf_converted}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/helenn/docker/vllm-hsg-20260413.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/lustre:/lustre,/home:/home}"
REPO_DIR="${REPO_DIR:-/lustre/fsw/portfolios/llmservice/users/${USER}/megatron-lm}"
SCRIPT_DIR="${SCRIPT_DIR:-${REPO_DIR}/examples/inference}"

GPUS_PER_NODE=4
SERVE_PORT="${SERVE_PORT:-8000}"
SERVE_PORT="${SERVE_PORT:-8000}"
RAY_PORT="${RAY_PORT:-6379}"

# Parallelism
TP_SIZE="${TP_SIZE:-$GPUS_PER_NODE}"
DP_SIZE="${DP_SIZE:-$SLURM_JOB_NUM_NODES}"
PP_SIZE="${PP_SIZE:-1}"

# Benchmark parameters
DATASET="${DATASET:-gsm8k}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_OUTPUT_TOKENS="${NUM_OUTPUT_TOKENS:-512}"
NUM_ITERS="${NUM_ITERS:-1}"
NUM_WARMUP_ITERS="${NUM_WARMUP_ITERS:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-73728}"

TOTAL_GPUS=$(( SLURM_JOB_NUM_NODES * GPUS_PER_NODE ))
EP_SIZE=$(( TP_SIZE * DP_SIZE ))

echo "========================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Nodes        : $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')"
echo "GPUs / node  : $GPUS_PER_NODE"
echo "Total GPUs   : $TOTAL_GPUS"
echo "Parallelism  : TP=$TP_SIZE  DP=$DP_SIZE  PP=$PP_SIZE  EP=$EP_SIZE (auto)"
echo "Model        : $MODEL_PATH"
echo "Container    : $CONTAINER_IMAGE"
echo "Benchmark    : batch=$BATCH_SIZE  dataset=$DATASET  OSL=$NUM_OUTPUT_TOKENS  iters=$NUM_ITERS"
echo "========================================"

export MODEL_PATH CONTAINER_IMAGE REPO_DIR SCRIPT_DIR GPUS_PER_NODE SERVE_PORT RAY_PORT
export TP_SIZE DP_SIZE PP_SIZE MAX_MODEL_LEN DATASET BATCH_SIZE NUM_OUTPUT_TOKENS NUM_ITERS NUM_WARMUP_ITERS

srun \
    --ntasks="${SLURM_JOB_NUM_NODES}" \
    --ntasks-per-node=1 \
    --mpi=pmix \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --export=ALL \
    bash -lc '
    set -euo pipefail

    export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}_rank${SLURM_PROCID}"
    mkdir -p "$TRITON_CACHE_DIR"

    HEAD_IP_FILE="/lustre/fsw/portfolios/llmservice/users/helenn/tmp/.ray_head_ip_${SLURM_JOB_ID}"

    if [ "$SLURM_PROCID" -eq 0 ]; then
        rm -f "$HEAD_IP_FILE"
        mkdir -p "$(dirname "$HEAD_IP_FILE")"

        HEAD_IP=$(hostname -I | awk "{print \$1}")
        if [ -z "$HEAD_IP" ]; then
            HEAD_IP=$(getent hosts "$(hostname)" | awk "{print \$1; exit}")
        fi
        if [ -z "$HEAD_IP" ]; then
            echo "ERROR: could not determine head node IP"
            exit 1
        fi

        echo "$HEAD_IP" > "$HEAD_IP_FILE"
        echo "[$(date +%H:%M:%S)] Starting Ray head on ${HEAD_IP}:${RAY_PORT}"
        ray start --head --node-ip-address="${HEAD_IP}" --port="${RAY_PORT}" --disable-usage-stats

        # Wait for all workers to join
        EXPECTED_GPUS=$(( SLURM_JOB_NUM_NODES * GPUS_PER_NODE ))
        echo "Waiting for $EXPECTED_GPUS GPUs in Ray cluster..."
        for i in $(seq 1 120); do
            AVAILABLE=$(python3 -c "import ray; ray.init(address=\"auto\"); print(int(ray.cluster_resources().get(\"GPU\", 0)))" 2>/dev/null || echo 0)
            if [ "$AVAILABLE" -ge "$EXPECTED_GPUS" ]; then
                echo "[$(date +%H:%M:%S)] Ray cluster ready: $AVAILABLE GPUs available"
                break
            fi
            if [ "$i" -eq 120 ]; then
                echo "ERROR: Timed out waiting for Ray workers (got $AVAILABLE/$EXPECTED_GPUS GPUs)"
                rm -f "$HEAD_IP_FILE"
                exit 1
            fi
            sleep 2
        done

        echo "[$(date +%H:%M:%S)] Starting vLLM server (TP=${TP_SIZE}, DP=${DP_SIZE})"
        LOG_FILE="/tmp/vllm_serve_${SLURM_JOB_ID}.log"

        export VLLM_USE_FLASHINFER_MOE_FP16=1
        export VLLM_FLASHINFER_MOE_BACKEND=latency

        DP_ARGS=""
        if [ "${DP_SIZE}" -gt 1 ]; then
            DP_ARGS="--data-parallel-size ${DP_SIZE} --data-parallel-backend ray --data-parallel-address ${HEAD_IP} --data-parallel-size-local 1"
        fi

        vllm serve "${MODEL_PATH}" \
            --host 0.0.0.0 \
            --port "${SERVE_PORT}" \
            --tensor-parallel-size "${TP_SIZE}" \
            --pipeline-parallel-size "${PP_SIZE}" \
            --distributed-executor-backend ray \
            ${DP_ARGS} \
            --enable-expert-parallel \
            --enable-chunked-prefill \
            --dtype bfloat16 \
            --trust-remote-code \
            --max-model-len "${MAX_MODEL_LEN}" \
            --gpu-memory-utilization 0.9 \
            --compilation-config "{\"pass_config\": {\"fuse_allreduce_rms\": false}}" \
            > "${LOG_FILE}" 2>&1 &
        VLLM_PID=$!

        # Tail log so output appears in SLURM log
        tail -f "${LOG_FILE}" &
        TAIL_PID=$!

        # Wait for server readiness
        echo "Waiting for server to be ready..."
        while ! grep -q -e "Uvicorn running on" -e "Application startup complete" "${LOG_FILE}" 2>/dev/null; do
            if ! kill -0 "$VLLM_PID" 2>/dev/null; then
                echo "ERROR: vLLM server process died. Last 50 lines:"
                tail -50 "${LOG_FILE}"
                kill "$TAIL_PID" 2>/dev/null || true
                ray stop || true
                rm -f "$HEAD_IP_FILE"
                exit 1
            fi
            sleep 2
        done

        echo ""
        echo "[$(date +%H:%M:%S)] Server is ready on http://${HEAD_IP}:${SERVE_PORT}"
        echo ""

        # Install benchmark dependencies
        pip install openai httpx datasets

        # Run benchmark
        echo "[$(date +%H:%M:%S)] Running benchmark: batch=$BATCH_SIZE  dataset=$DATASET  OSL=$NUM_OUTPUT_TOKENS"
        python3 -u "${REPO_DIR}/inference-bench/static_benchmark.py" \
            --server-url "http://localhost:${SERVE_PORT}/v1" \
            --model "${MODEL_PATH}" \
            --batch-size "${BATCH_SIZE}" \
            --dataset "${DATASET}" \
            --num-output-tokens "${NUM_OUTPUT_TOKENS}" \
            --num-iters "${NUM_ITERS}" \
            --num-warmup-iters "${NUM_WARMUP_ITERS}"

        # Shutdown
        echo "[$(date +%H:%M:%S)] Benchmark complete. Shutting down server..."
        kill "$VLLM_PID" 2>/dev/null || true
        kill "$TAIL_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        ray stop || true
        rm -f "$HEAD_IP_FILE"

    else
        # Worker: wait for head IP file
        for _ in $(seq 1 120); do
            [ -s "$HEAD_IP_FILE" ] && break
            sleep 1
        done

        if [ ! -s "$HEAD_IP_FILE" ]; then
            echo "ERROR: timed out waiting for Ray head IP file"
            exit 1
        fi

        HEAD_IP=$(cat "$HEAD_IP_FILE")

        echo "=== [rank${SLURM_PROCID}] Waiting for Ray head ${HEAD_IP}:${RAY_PORT} ==="
        for _ in $(seq 1 120); do
            if ray status --address "${HEAD_IP}:${RAY_PORT}" >/dev/null 2>&1; then
                break
            fi
            sleep 2
        done

        echo "=== [rank${SLURM_PROCID}] Starting Ray worker ==="
        ray start --address "${HEAD_IP}:${RAY_PORT}" --disable-usage-stats

        # Keep worker alive while rank0 runs
        tail -f /dev/null
    fi
    '

# --------------- cleanup -----------------------------------------------------
echo "Done."
