#!/bin/bash
#SBATCH --partition=batch
#SBATCH --account=llmservice_fm_text
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --segment=4
#SBATCH --time=00:45:00
#SBATCH --job-name=llmservice-vllm-inference
#SBATCH --output=/lustre/fsw/portfolios/llmservice/users/helenn/logs/vllm-debug.out
#SBATCH --error=/lustre/fsw/portfolios/llmservice/users/helenn/logs/vllm-debug.out
#SBATCH --exclusive

# =============================================================================
# Multi-node vLLM inference via DP (data-parallel) + EP (expert-parallel)
#
# Each node runs its own vLLM process with TP=local GPUs.
# DP coordinates across nodes for expert all-to-all.
# EP = TP x DP (automatic when --enable-expert-parallel is set).
#
# Usage:
#   MODEL_PATH=/path/to/hf TP_SIZE=4 DP_SIZE=4 sbatch slurm_vllm_multinode_inference.sh
#   BATCH_SIZE=8 NUM_INPUT_TOKENS=4096 sbatch slurm_vllm_multinode_inference.sh
# =============================================================================

set -euo pipefail

# --------------- configuration (override via env) ----------------------------
MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nemotron_ultra/nemo_rl/ci/checkpoints/ultra-v3-sft-hsg-mainfeb19merge-mxfp8_fixed-hf_converted}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/helenn/docker/vllm-hsg-nightly.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/lustre:/lustre,/home:/home}"
SCRIPT_DIR="${SCRIPT_DIR:-/lustre/fsw/portfolios/llmservice/users/${USER}/megatron-lm/examples/inference}"

GPUS_PER_NODE=4

# Parallelism
# TP = per-node tensor parallelism (default: all local GPUs)
# DP = number of data-parallel ranks (default: number of nodes)
# EP = TP x DP (automatic with --enable-expert-parallel)
TP_SIZE="${TP_SIZE:-$GPUS_PER_NODE}"
DP_SIZE="${DP_SIZE:-$SLURM_JOB_NUM_NODES}"
PP_SIZE="${PP_SIZE:-1}"

# Benchmark parameters
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_INPUT_TOKENS="${NUM_INPUT_TOKENS:-8192}"
NUM_OUTPUT_TOKENS="${NUM_OUTPUT_TOKENS:-65536}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-73728}"

# --------------- resolve head node -------------------------------------------
read -ra NODES <<< "$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')"
HEAD_NODE="${NODES[0]}"
HEAD_ADDR=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" \
    hostname -I | awk '{print $1}')

TOTAL_GPUS=$(( SLURM_JOB_NUM_NODES * GPUS_PER_NODE ))
EP_SIZE=$(( TP_SIZE * DP_SIZE ))

echo "========================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Nodes        : ${NODES[*]}"
echo "Head node    : $HEAD_NODE ($HEAD_ADDR)"
echo "GPUs / node  : $GPUS_PER_NODE"
echo "Total GPUs   : $TOTAL_GPUS"
echo "Parallelism  : TP=$TP_SIZE  DP=$DP_SIZE  PP=$PP_SIZE  EP=$EP_SIZE (auto)"
echo "Model        : $MODEL_PATH"
echo "Container    : $CONTAINER_IMAGE"
echo "Benchmark    : batch=$BATCH_SIZE  ISL=$NUM_INPUT_TOKENS  OSL=$NUM_OUTPUT_TOKENS"
echo "========================================"

# --------------- write per-node launcher to shared storage -------------------
LAUNCHER="/lustre/fsw/portfolios/llmservice/users/${USER}/tmp/vllm_launcher_${SLURM_JOB_ID}.sh"
mkdir -p "$(dirname "$LAUNCHER")"

cat > "$LAUNCHER" << 'LAUNCHER_EOF'
#!/bin/bash
set -euo pipefail

HEAD_ADDR="$1"
DP_SIZE="$2"
GPUS_PER_NODE="$3"
RAY_PORT="${4:-6379}"
shift 4
# Remaining args are the benchmark command

NODE_RANK="${SLURM_NODEID:-0}"

# Avoid Triton cache race conditions on shared filesystems.
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}_node${NODE_RANK}"
mkdir -p "$TRITON_CACHE_DIR"

echo "[Node $(hostname -s)] rank=$NODE_RANK  head=$HEAD_ADDR  dp_size=$DP_SIZE"

if [ "$NODE_RANK" -eq 0 ]; then
    # Head node: start Ray head and run the benchmark.
    ray start --head --port="$RAY_PORT" --num-gpus="$GPUS_PER_NODE"

    # Wait for all worker nodes to join the Ray cluster
    EXPECTED_GPUS=$(( SLURM_JOB_NUM_NODES * GPUS_PER_NODE ))
    echo "Waiting for $EXPECTED_GPUS GPUs in Ray cluster..."
    for i in $(seq 1 120); do
        AVAILABLE=$(python3 -c "import ray; ray.init(address='auto'); print(int(ray.cluster_resources().get('GPU', 0)))" 2>/dev/null || echo 0)
        if [ "$AVAILABLE" -ge "$EXPECTED_GPUS" ]; then
            echo "Ray cluster ready: $AVAILABLE GPUs available"
            break
        fi
        if [ "$i" -eq 120 ]; then
            echo "ERROR: Timed out waiting for Ray workers (got $AVAILABLE/$EXPECTED_GPUS GPUs)"
            exit 1
        fi
        sleep 2
    done

    if [ "$DP_SIZE" -gt 1 ]; then
        # Head spawns 1 local DP engine core; Ray places the remaining
        # DP-1 engine cores on worker nodes (each with TP GPUs).
        "$@" \
            --data-parallel-size "$DP_SIZE" \
            --data-parallel-size-local 1 \
            --data-parallel-address "$HEAD_ADDR" \
            --distributed-executor-backend ray
    else
        "$@"
    fi
else
    # Worker node: join the Ray cluster and block until the job ends.
    ray start --address="$HEAD_ADDR:$RAY_PORT" --num-gpus="$GPUS_PER_NODE" --block
fi
LAUNCHER_EOF
chmod +x "$LAUNCHER"

# --------------- launch on all nodes -----------------------------------------
srun \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$CONTAINER_MOUNTS" \
    bash "$LAUNCHER" \
    "$HEAD_ADDR" \
    "$DP_SIZE" \
    "$GPUS_PER_NODE" \
    "6379" \
    python "$SCRIPT_DIR/run_vllm_benchmark.py" \
        --model "$MODEL_PATH" \
        --tensor-parallel-size "$TP_SIZE" \
        --pipeline-parallel-size "$PP_SIZE" \
        --enable-expert-parallel \
        --enable-chunked-prefill \
        --dtype bfloat16 \
        --trust-remote-code \
        --max-model-len "$MAX_MODEL_LEN" \
        --compilation-config '{"pass_config": {"fuse_allreduce_rms": false}}' \
        --batch-size "$BATCH_SIZE" \
        --num-input-tokens "$NUM_INPUT_TOKENS" \
        --num-output-tokens "$NUM_OUTPUT_TOKENS"

# --------------- cleanup -----------------------------------------------------
rm -f "$LAUNCHER"
echo "Done."
