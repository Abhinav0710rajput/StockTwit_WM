#!/bin/bash
# ============================================================================
# TwitWave — RSSM large model training
#
# Submit:  sbatch slurm/train_rssm_large.sh
# Resume:  same command — script detects last.pt automatically
# ============================================================================

#SBATCH --job-name=twitwave_rssm_large
#SBATCH --account=ds_ga_1003-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --chdir=/home/ar10026/StockTwit_WM
#SBATCH --output=logs/%j_train.out
#SBATCH --error=logs/%j_train.err

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
CFG="${CFG:-configs/rssm_large.yaml}"
DATA_DIR="${DATA_DIR:-data/processed_week}"
OUT_DIR="${OUT_DIR:-outputs/rssm_large}"
SEED="${SEED:-42}"
WANDB="${WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-twit_wave}"
FRESH_START="${FRESH_START:-false}"      # set FRESH_START=true to ignore existing checkpoint
RESET_PATIENCE="${RESET_PATIENCE:-false}" # set RESET_PATIENCE=true to reset early-stopping counter

# ── Directories ───────────────────────────────────────────────────────────────
# --chdir in SBATCH directives already sets CWD to the project root

mkdir -p logs "$OUT_DIR"

# ── Log job context ───────────────────────────────────────────────────────────
echo "========================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $(hostname)"
echo "Started      : $(date)"
echo "Config       : $CFG"
echo "Output dir   : $OUT_DIR"
echo "SLURM_RESTART: ${SLURM_RESTART_COUNT:-0}"
echo "========================================"

# ── Environment ───────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wm_ml

export PYTHONPATH="$PWD"

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# ── Resume detection ──────────────────────────────────────────────────────────
LAST_CKPT="$OUT_DIR/checkpoints/last.pt"
RESUME_FLAG=""
RESET_PATIENCE_FLAG=""
if [ "$FRESH_START" = "true" ]; then
    echo "FRESH_START=true — ignoring existing checkpoint"
elif [ -f "$LAST_CKPT" ]; then
    echo "Found checkpoint: $LAST_CKPT — resuming"
    RESUME_FLAG="--resume latest"
    if [ "$RESET_PATIENCE" = "true" ]; then
        echo "RESET_PATIENCE=true — resetting early-stopping counter"
        RESET_PATIENCE_FLAG="--reset_patience"
    fi
else
    echo "No checkpoint found — starting fresh"
fi

# ── Wandb flag ────────────────────────────────────────────────────────────────
WANDB_FLAG=""
if [ "$WANDB" = "true" ]; then
    WANDB_FLAG="--wandb --wandb_project $WANDB_PROJECT"
fi

# ── Trap: SIGTERM before preemption/timeout ───────────────────────────────────
_term() {
    echo "SIGTERM received at $(date) — job will be requeued, last.pt is safe"
}
trap _term SIGTERM

# ── Train ─────────────────────────────────────────────────────────────────────
python scripts/2_b_train_rssm.py \
    --cfg      "$CFG"      \
    --data_dir "$DATA_DIR" \
    --out_dir  "$OUT_DIR"  \
    --seed     "$SEED"     \
    $RESUME_FLAG           \
    $RESET_PATIENCE_FLAG   \
    $WANDB_FLAG

echo "========================================"
echo "Training complete: $(date)"
echo "========================================"
