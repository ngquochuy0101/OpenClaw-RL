#!/bin/bash

# ============================================================================
# OpenClaw-RL: Qwen2.5-1.5B LoRA — 1-GPU, 4GB VRAM optimized
# ============================================================================
# Target: NVIDIA RTX 3050 Laptop (4 GB VRAM) on WSL2
#
# Key VRAM-saving changes vs the 0.5B default script:
#   - lora-rank        64 → 16        (fewer trainable params)
#   - lora-alpha       64 → 32        (standard 2× rank)
#   - context-length   32768 → 4096   (shorter KV cache)
#   - max-tokens-per-gpu 8192 → 2048  (smaller training micro-batch)
#   - rollout-max-response-len 8192 → 2048
#   - rollout-max-context-len  32768 → 4096
#   - rollout-batch-size 16 → 4       (fewer concurrent sequences)
#   - mem-fraction-static 0.85 → 0.90 (more VRAM for static model weights)
#   - gradient-checkpointing ON       (trade compute for memory)
#   - optimizer-cpu-offload ON        (Adam states → CPU RAM)
#   - TP=1                            (no tensor parallelism for 1 GPU)
#   - PRM_GPUS=0                      (PRM disabled — no room for a second model)
#
# Before running:
#   1. Download Qwen2.5-1.5B-Instruct to a local folder, e.g.:
#        huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
#          --local-dir /path/to/Qwen2.5-1.5B-Instruct
#
#   2. Set env vars (or edit the defaults below):
#        export HF_CKPT=/path/to/Qwen2.5-1.5B-Instruct
#        export SAVE_CKPT=/path/to/output/ckpt
#
#   3. Run:
#        cd slime
#        bash ../openclaw-combine/run_qwen2.5_1.5b_lora_1gpu_4gb.sh
# ============================================================================

pkill -9 sglang  2>/dev/null
sleep 2
ray stop --force 2>/dev/null
pkill -9 ray     2>/dev/null
pkill -9 python  2>/dev/null
sleep 2
pkill -9 ray     2>/dev/null
pkill -9 python  2>/dev/null

set -ex

# ── Unbuffered output ───────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# ── GPU layout ──────────────────────────────────────────────────────────────
NUM_GPUS=${NUM_GPUS:-1}
ACTOR_GPUS=${ACTOR_GPUS:-1}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-0}      # rollout shares the actor GPU
PRM_GPUS=${PRM_GPUS:-0}              # PRM disabled (no VRAM headroom)

if (( ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS > NUM_GPUS )); then
    echo "ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS must be <= NUM_GPUS"
    echo "ACTOR_GPUS=${ACTOR_GPUS}, ROLLOUT_GPUS=${ROLLOUT_GPUS}, PRM_GPUS=${PRM_GPUS}, NUM_GPUS=${NUM_GPUS}"
    exit 1
fi

export RAY_health_check_failure_threshold=20
export RAY_health_check_period_ms=5000
export RAY_health_check_timeout_ms=30000
export RAY_num_heartbeats_timeout=60
export RAY_memory_monitor_refresh_ms=0

# Add PyTorch memory fragmentation optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/../slime" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

# ── Model checkpoint paths (override via env) ──────────────────────────────
HF_CKPT=${HF_CKPT:-${REPO_ROOT}/models/Qwen2.5-1.5B-Instruct}
REF_LOAD=${REF_LOAD:-${HF_CKPT}}
SAVE_CKPT=${SAVE_CKPT:-${REPO_ROOT}/ckpt/qwen2.5-1.5b-openclaw-lora-4gb}
PRM_MODEL_PATH=${PRM_MODEL_PATH:-${HF_CKPT}}

# ── Serving config ─────────────────────────────────────────────────────────
export SGLANG_API_KEY="${SGLANG_API_KEY}"
export SERVED_MODEL_NAME="qwen2.5-1.5b"
export HOST="0.0.0.0"
export PORT="${PORT:-30000}"
export OPENCLAW_RECORD_ENABLED="${OPENCLAW_RECORD_ENABLED:-1}"
export OPENCLAW_RECORD_FILE="${SCRIPT_DIR}/results/qwen2.5_1.5b_lora_4gb_record.jsonl"

# ── 4 GB VRAM-critical settings ────────────────────────────────────────────
export TP="1"                        # single GPU → no tensor parallelism
export CONTEXT_LENGTH="4096"         # ↓ from 32768 — huge VRAM saving
export MEM_FRACTION_STATIC="0.90"    # ↑ from 0.85  — more room for weights
export REASONING_PARSER="qwen3"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen25}"
export PRM_M="${PRM_M:-1}"
export OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY="${OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY:-1}"
export OPENCLAW_COMBINE_W_RL="${OPENCLAW_COMBINE_W_RL:-1.0}"
export OPENCLAW_COMBINE_W_OPD="${OPENCLAW_COMBINE_W_OPD:-1.0}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}"

# ── Checkpoint args ─────────────────────────────────────────────────────────
CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_CKPT}"
   --save-interval 100
)

# ── Rollout args (trimmed for 4 GB) ────────────────────────────────────────
ROLLOUT_ARGS=(
   --disable-rollout-global-dataset
   --rollout-function-path openclaw_combine_rollout.generate_rollout_openclaw_combine

   --num-rollout 100000000
   --rollout-batch-size 4            # ↓ from 16 — fewer concurrent sequences
   --n-samples-per-prompt 1
   --rollout-max-response-len 2048   # ↓ from 8192
   --rollout-max-context-len 4096    # ↓ from 32768
   --rollout-temperature 0.6
   --reward-key score

   --num-steps-per-rollout 1
)

# ── Performance / memory args ──────────────────────────────────────────────
PERF_ARGS=(
   --use-dynamic-batch-size
   --max-tokens-per-gpu 2048         # ↓ from 8192 — critical for 4 GB
   --gradient-checkpointing          # recompute activations to save VRAM
   --fp16                            # Use fp16 for 50% VRAM savings vs fp32
)

# ── Loss / advantage args ──────────────────────────────────────────────────
COMBINE_ARGS=(
   --advantage-estimator grpo
   --disable-rewards-normalization
   --loss-type custom_loss
   --custom-loss-function-path combine_loss.combine_loss_function
   --use-kl-loss
   --kl-loss-coef 0.0
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

# ── Optimizer (CPU-offloaded Adam to save ~200 MB VRAM) ─────────────────────
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --fsdp-cpu-offload
)

# ── LoRA (rank 16 to fit 4 GB) ─────────────────────────────────────────────
LORA_ARGS=(
   --use-lora
   --lora-rank 16                     # ↓ from 64 — saves ~3× LoRA VRAM
   --lora-alpha 32                    # standard 2× rank
   --lora-target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
)

# ── SGLang inference engine ─────────────────────────────────────────────────
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${TP}"
   --sglang-tool-call-parser "${TOOL_CALL_PARSER}"
   --sglang-mem-fraction-static 0.90
   --sglang-context-length 4096       # match CONTEXT_LENGTH
   --sglang-reasoning-parser qwen3
)

# ── PRM (disabled — no VRAM for a second model) ────────────────────────────
PRM_ARGS=()

# ── Custom API server ──────────────────────────────────────────────────────
CUSTOM_ARGS=(
   --custom-generate-function-path openclaw_combine_api_server.generate
   --custom-rm-path openclaw_combine_api_server.reward_func
)

# ── WandB (off by default for local experiments) ────────────────────────────
USE_WANDB=${USE_WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-openclaw_rl}
WANDB_KEY_VALUE=${WANDB_KEY:-${WANDB_API_KEY:-}}
if [ "${USE_WANDB}" = "1" ] && [ -n "${WANDB_KEY_VALUE}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project ${WANDB_PROJECT}
    --wandb-group qwen2.5-1.5b-openclaw-lora-4gb
    --wandb-key ${WANDB_KEY_VALUE}
  )
else
  WANDB_ARGS=()
fi

export OPENCLAW_EVAL_MODE="${OPENCLAW_EVAL_MODE:-1}"

# ── Launch Ray + submit training job ────────────────────────────────────────
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${SCRIPT_DIR}:${SCRIPT_DIR}/../openclaw-opd:${SLIME_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"OPENCLAW_EVAL_MODE\": \"${OPENCLAW_EVAL_MODE}\",
    \"OPENCLAW_COMBINE_W_RL\": \"${OPENCLAW_COMBINE_W_RL}\",
    \"OPENCLAW_COMBINE_W_OPD\": \"${OPENCLAW_COMBINE_W_OPD}\",
    \"TRAIN_EPOCHS\": \"${TRAIN_EPOCHS}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --train-backend fsdp \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_GPUS}" \
   --rollout-num-gpus "${ROLLOUT_GPUS}" \
   --num-gpus-per-node "${NUM_GPUS}" \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${COMBINE_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${PRM_ARGS[@]} \
   ${LORA_ARGS[@]}
