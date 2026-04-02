#!/bin/bash
#SBATCH --partition=batch
#SBATCH --account=llmservice_fm_text
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --segment=4
#SBATCH --time=01:00:00
#SBATCH --job-name=llmservice-megatron-inference
#SBATCH --output=/lustre/fsw/portfolios/llmservice/users/helenn/logs/keshav_ultra_20260331/sanity.log
#SBATCH --error=/lustre/fsw/portfolios/llmservice/users/helenn/logs/ultra-debug.out
#SBATCH --exclusive

# =============================================================================
# Usage:
#   sbatch slurm_megatron_multinode_inference.sh
#   sbatch --nodes=4 slurm_megatron_multinode_inference.sh
#   TP_SIZE=4 EP_SIZE=4 PP_SIZE=2 sbatch --nodes=2 slurm_megatron_multinode_inference.sh
# =============================================================================

set -euo pipefail

# --------------- configuration (override via env) ----------------------------
MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/ksanthanam/nanov3}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/portfolios/llmservice/users/helenn/docker/vllm-hsg-nightly.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/lustre:/lustre,/home:/home}"
MEGATRON_DIR="${MEGATRON_DIR:-/lustre/fsw/portfolios/llmservice/users/${USER}/megatron-lm}"

GPUS_PER_NODE=4
MASTER_PORT="${MASTER_PORT:-29500}"

# Parallelism defaults: TP and EP scale with total GPUs, PP=1
TOTAL_GPUS=$(( SLURM_JOB_NUM_NODES * GPUS_PER_NODE ))
TP_SIZE="${TP_SIZE:-$TOTAL_GPUS}"
EP_SIZE="${EP_SIZE:-$TOTAL_GPUS}"
ETP_SIZE="${ETP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"

# Benchmark parameters
NUM_TOKENS_TO_GENERATE="${NUM_TOKENS_TO_GENERATE:-65536}"
INFERENCE_MAX_REQUESTS="${INFERENCE_MAX_REQUESTS:-4}"
NUM_INPUT_TOKENS="${NUM_INPUT_TOKENS:-8192}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-73728}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
DYNAMIC_BATCHING_MAX_REQUESTS="${DYNAMIC_BATCHING_MAX_REQUESTS:-128}"
DYNAMIC_BATCHING_BUFFER_SIZE_GB="${DYNAMIC_BATCHING_BUFFER_SIZE_GB:-45}"
DYNAMIC_BATCHING_MAX_TOKENS="${DYNAMIC_BATCHING_MAX_TOKENS:-2048}"
MAMBA_MEMORY_RATIO="${MAMBA_MEMORY_RATIO:-0.3}"

# --------------- resolve master address -------------------------------------
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

echo "========================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Nodes        : $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')"
echo "Master       : $MASTER_ADDR:$MASTER_PORT"
echo "GPUs / node  : $GPUS_PER_NODE"
echo "Total GPUs   : $TOTAL_GPUS"
echo "Parallelism  : TP=$TP_SIZE  EP=$EP_SIZE  ETP=$ETP_SIZE  PP=$PP_SIZE"
echo "Model        : $MODEL_PATH"
echo "Container    : $CONTAINER_IMAGE"
echo "Megatron dir : $MEGATRON_DIR"
echo "========================================"

# --------------- write per-node launcher to shared storage -------------------
LAUNCHER="/lustre/fsw/portfolios/llmservice/users/${USER}/tmp/megatron_launcher_${SLURM_JOB_ID}.sh"
mkdir -p "$(dirname "$LAUNCHER")"

cat > "$LAUNCHER" << 'LAUNCHER_EOF'
#!/bin/bash
set -euo pipefail

MASTER_ADDR="$1"
MASTER_PORT="$2"
NNODES="$3"
GPUS_PER_NODE="$4"
MEGATRON_DIR="$5"
shift 5
# Remaining args are passed to the inference script

NODE_RANK="${SLURM_NODEID:-0}"

echo "[Node $(hostname -s)] rank=$NODE_RANK  master=$MASTER_ADDR:$MASTER_PORT  nnodes=$NNODES"

cd "$MEGATRON_DIR"

# Avoid Triton cache stale file handles on shared filesystems.
# Each node gets its own cache dir under local /tmp.
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}_node${NODE_RANK}"
mkdir -p "$TRITON_CACHE_DIR"

CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun \
    --nproc-per-node "$GPUS_PER_NODE" \
    --nnodes "$NNODES" \
    --node-rank "$NODE_RANK" \
    --master-addr "$MASTER_ADDR" \
    --master-port "$MASTER_PORT" \
    -m tools.run_inference_performance_test \
    "$@"
LAUNCHER_EOF
chmod +x "$LAUNCHER"

# --------------- launch on all nodes -----------------------------------------
srun \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$CONTAINER_MOUNTS" \
    bash "$LAUNCHER" \
    "$MASTER_ADDR" \
    "$MASTER_PORT" \
    "$SLURM_JOB_NUM_NODES" \
    "$GPUS_PER_NODE" \
    "$MEGATRON_DIR" \
    --load "$MODEL_PATH" \
    --attention-backend flash \
    --bf16 \
    --tensor-model-parallel-size "$TP_SIZE" \
    --expert-model-parallel-size "$EP_SIZE" \
    --expert-tensor-parallel-size "$ETP_SIZE" \
    --pipeline-model-parallel-size "$PP_SIZE" \
    --micro-batch-size "$MICRO_BATCH_SIZE" \
    --num-tokens-to-generate "$NUM_TOKENS_TO_GENERATE" \
    --inference-max-requests "$INFERENCE_MAX_REQUESTS" \
    --num-input-tokens "$NUM_INPUT_TOKENS" \
    --use-checkpoint-args \
    --dist-ckpt-strictness log_unexpected \
    --cuda-graph-impl local \
    --inference-dynamic-batching-num-cuda-graphs -1 \
    --engine-type dynamic \
    --inference-dynamic-batching-max-requests "$DYNAMIC_BATCHING_MAX_REQUESTS" \
    --model-provider mamba \
    --moe-router-dtype fp32 \
    --moe-token-dispatcher-type alltoall \
    --inference-dynamic-batching-buffer-size-gb "$DYNAMIC_BATCHING_BUFFER_SIZE_GB" \
    --inference-dynamic-batching-max-tokens "$DYNAMIC_BATCHING_MAX_TOKENS" \
    --enable-chunked-prefill \
    --inference-max-seq-length "$MAX_SEQ_LENGTH" \
    --transformer-impl inference_optimized \
    --sequence-parallel \
    --inference-logging-step-interval 10000 \
    --num-layers 108 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --max-position-embeddings 262144 \
    --num-experts 512 \
    --tokenizer-type TikTokenizer \
    --tokenizer-model /lustre/fsw/portfolios/llmservice/projects/llmservice_fm_text/megatron_rl/models/nemotron6/tokenizers/multiMixV8.gpt4o_nc_sd.500000.128k.vocab.json \
    --normalization RMSNorm \
    --disable-bias-linear \
    --inference-dynamic-batching-mamba-memory-ratio "$MAMBA_MEMORY_RATIO"

# --------------- cleanup -----------------------------------------------------
rm -f "$LAUNCHER"
echo "Done."
