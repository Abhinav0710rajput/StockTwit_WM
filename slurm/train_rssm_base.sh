#!/bin/bash
# ============================================================================
# TwitWave — RSSM base model training
#
# Submit:   sbatch slurm/train_rssm_base.sh
# Resume:   same command — script detects last.pt automatically
# Override: CFG=configs/rssm_large.yaml OUT=outputs/rssm_large sbatch slurm/train_rssm_base.sh
# ============================================================================

#SBATCH --job-name=twitwave_rssm
#SBATCH --account=ds_ga_1003-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%j_train.out
#SBATCH --error=logs/%j_train.err

set -euo pipefail

# ── Config (override via env vars before sbatch) ──────────────────────────────
CFG="${CFG:-configs/rssm_base.yaml}"
DATA_DIR="${DATA_DIR:-data/processed_week}"
OUT_DIR="${OUT_DIR:-outputs/rssm_base}"
SEED="${SEED:-42}"
WANDB="${WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-twit_wave}"

# ── Directories ───────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

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

export PYTHONPATH="$PROJECT_DIR"

# Confirm GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# ── Resume detection ──────────────────────────────────────────────────────────
LAST_CKPT="$OUT_DIR/checkpoints/last.pt"
RESUME_FLAG=""
if [ -f "$LAST_CKPT" ]; then
    echo "Found checkpoint: $LAST_CKPT — resuming"
    RESUME_FLAG="--resume latest"
else
    echo "No checkpoint found — starting fresh"
fi

# ── Wandb flag ────────────────────────────────────────────────────────────────
WANDB_FLAG=""
if [ "$WANDB" = "true" ]; then
    WANDB_FLAG="--wandb --wandb_project $WANDB_PROJECT"
fi

# ── Trap: SIGTERM is sent by SLURM before preemption/timeout ─────────────────
# last.pt is written after every epoch so the next requeue picks up from there.
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
    $WANDB_FLAG

echo "========================================"
echo "Training complete: $(date)"
echo "========================================"
