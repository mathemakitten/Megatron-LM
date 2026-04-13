#!/bin/bash
#SBATCH --partition=batch
#SBATCH --account=llmservice_fm_text
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --segment=4
#SBATCH --time=00:25:00
#SBATCH --job-name=llmservice-vllm-profile
#SBATCH --output=/lustre/fsw/portfolios/llmservice/users/helenn/logs/vllm/vllm-profile.out
#SBATCH --error=/lustre/fsw/portfolios/llmservice/users/helenn/logs/vllm/vllm-profile.out
#SBATCH --exclusive

# =============================================================================
# Multi-node vLLM serve + nsys profile
#
# Same Ray-based multi-node setup as slurm_vllm_serve_benchmark.sh, but
# wraps vllm serve with nsys on the head node. Sends a warmup request
# then a single short request for profiling.
#
# Usage:
#   sbatch slurm_vllm_profile.sh
#   NUM_INPUT_TOKENS=256 NUM_OUTPUT_TOKENS=64 sbatch slurm_vllm_profile.sh
# =============================================================================

set -euo pipefail

# --------------- configuration (override via env) ----------------------------
MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nemotron_ultra/nemo_rl/ci/checkpoints/ultra-v3-sft-hsg-mainfeb19merge-mxfp8_fixed-hf_converted}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_text/users/helenn/docker/vllm-hsg-nightly.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/lustre:/lustre,/home:/home}"
SCRIPT_DIR="${SCRIPT_DIR:-/lustre/fsw/portfolios/llmservice/users/${USER}/megatron-lm/examples/inference}"

GPUS_PER_NODE=4
SERVE_PORT="${SERVE_PORT:-8000}"
RAY_PORT="${RAY_PORT:-6379}"

# Parallelism
TP_SIZE="${TP_SIZE:-$GPUS_PER_NODE}"
DP_SIZE="${DP_SIZE:-$SLURM_JOB_NUM_NODES}"
PP_SIZE="${PP_SIZE:-1}"

# Short sequences for profiling
NUM_INPUT_TOKENS="${NUM_INPUT_TOKENS:-128}"
NUM_OUTPUT_TOKENS="${NUM_OUTPUT_TOKENS:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$(( NUM_INPUT_TOKENS + NUM_OUTPUT_TOKENS ))}"

# Profile output
PROFILE_DIR="${PROFILE_DIR:-/lustre/fsw/portfolios/llmservice/users/${USER}/profiles}"
PROFILE_NAME="${PROFILE_NAME:-vllm-profile-is${NUM_INPUT_TOKENS}-os${NUM_OUTPUT_TOKENS}}"

TOTAL_GPUS=$(( SLURM_JOB_NUM_NODES * GPUS_PER_NODE ))
EP_SIZE=$(( TP_SIZE * DP_SIZE ))

echo "========================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Nodes        : $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')"
echo "GPUs / node  : $GPUS_PER_NODE"
echo "Total GPUs   : $TOTAL_GPUS"
echo "Parallelism  : TP=$TP_SIZE  DP=$DP_SIZE  PP=$PP_SIZE  EP=$EP_SIZE (auto)"
echo "Model        : $MODEL_PATH"
echo "Sequences    : ISL=$NUM_INPUT_TOKENS  OSL=$NUM_OUTPUT_TOKENS"
echo "Profile      : $PROFILE_DIR/$PROFILE_NAME"
echo "========================================"

export MODEL_PATH CONTAINER_IMAGE SCRIPT_DIR GPUS_PER_NODE SERVE_PORT RAY_PORT
export TP_SIZE DP_SIZE PP_SIZE MAX_MODEL_LEN NUM_INPUT_TOKENS NUM_OUTPUT_TOKENS
export PROFILE_DIR PROFILE_NAME

srun \
    -q interactive \
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
        mkdir -p "$PROFILE_DIR"

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

        echo "[$(date +%H:%M:%S)] Starting vLLM server under nsys (TP=${TP_SIZE}, DP=${DP_SIZE})"
        LOG_FILE="/tmp/vllm_serve_${SLURM_JOB_ID}.log"

        # Disabled: FlashInfer MoE not compatible with this container version
        # export VLLM_USE_FLASHINFER_MOE_FP16=1
        # export VLLM_FLASHINFER_MOE_BACKEND=latency

        nsys profile \
            --trace=cuda,nvtx,osrt \
            --cuda-graph-trace=node \
            --sample=none \
            --output="${PROFILE_DIR}/${PROFILE_NAME}" \
            --force-overwrite=true \
            -- \
            vllm serve "${MODEL_PATH}" \
                --host 0.0.0.0 \
                --port "${SERVE_PORT}" \
                --tensor-parallel-size "${TP_SIZE}" \
                --pipeline-parallel-size "${PP_SIZE}" \
                --distributed-executor-backend ray \
                --data-parallel-size "${DP_SIZE}" \
                --data-parallel-backend ray \
                --data-parallel-address "${HEAD_IP}" \
                --data-parallel-size-local 1 \
                --enable-expert-parallel \
                --enable-chunked-prefill \
                --dtype bfloat16 \
                --trust-remote-code \
                --max-model-len "${MAX_MODEL_LEN}" \
                --gpu-memory-utilization 0.9 \
                --compilation-config "{\"pass_config\": {\"fuse_allreduce_rms\": false}}" \
                > "${LOG_FILE}" 2>&1 &
        NSYS_PID=$!

        # Tail log so output appears in SLURM log
        tail -f "${LOG_FILE}" &
        TAIL_PID=$!

        # Wait for server readiness
        echo "Waiting for server to be ready..."
        for i in $(seq 1 300); do
            if grep -q -e "Uvicorn running on" -e "Application startup complete" "${LOG_FILE}" 2>/dev/null; then
                break
            fi
            if ! kill -0 "$NSYS_PID" 2>/dev/null; then
                echo "ERROR: server process died. Last 50 lines:"
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

        # Warmup: 1 request to trigger CUDA graphs etc.
        echo "[$(date +%H:%M:%S)] Sending warmup request..."
        python "${SCRIPT_DIR}/run_vllm_serve_benchmark.py" \
            --server-url "http://localhost:${SERVE_PORT}" \
            --model "${MODEL_PATH}" \
            --batch-size 1 \
            --num-input-tokens "${NUM_INPUT_TOKENS}" \
            --num-output-tokens "${NUM_OUTPUT_TOKENS}" \
            --num-warmup-iters 1 \
            --num-iters 0

        # Profiled request: single forward pass
        echo "[$(date +%H:%M:%S)] Sending profiled request..."
        python "${SCRIPT_DIR}/run_vllm_serve_benchmark.py" \
            --server-url "http://localhost:${SERVE_PORT}" \
            --model "${MODEL_PATH}" \
            --batch-size 1 \
            --num-input-tokens "${NUM_INPUT_TOKENS}" \
            --num-output-tokens "${NUM_OUTPUT_TOKENS}" \
            --num-warmup-iters 0 \
            --num-iters 1

        # Shutdown — killing the server causes nsys to finalize the profile
        echo "[$(date +%H:%M:%S)] Request complete. Shutting down server..."
        kill "$NSYS_PID" 2>/dev/null || true
        kill "$TAIL_PID" 2>/dev/null || true
        wait "$NSYS_PID" 2>/dev/null || true
        ray stop || true
        rm -f "$HEAD_IP_FILE"

        echo "[$(date +%H:%M:%S)] Profile written to: ${PROFILE_DIR}/${PROFILE_NAME}.nsys-rep"

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

echo "Done."
