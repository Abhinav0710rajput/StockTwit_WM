#!/bin/bash
#SBATCH --job-name=twit_wave_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100-40gb
#SBATCH --output=logs/train_a100_%j.out
#SBATCH --error=logs/train_a100_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=CHANGE_ME@nyu.edu

# ─── NYU HPC: Greene cluster  ────────────────────────────────────────────────
# Node type: c12m85-a100-1 (A100 40GB SXM4, 12 cores, 85GB RAM)
# Submit from the project root: sbatch slurm/train_a100.sh
# Logs land in logs/ — create the directory first if it doesn't exist.

set -euo pipefail

mkdir -p logs outputs/rssm_base

echo "========================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start:      $(date)"
echo "Working dir: $(pwd)"
echo "========================================"

# ─── environment ─────────────────────────────────────────────────────────────
module purge
module load cuda/12.1
module load python/3.11

# Activate your conda / virtualenv (edit path as needed)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate twit_wave

# Verify GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ─── training ────────────────────────────────────────────────────────────────
CFG="${CFG:-configs/rssm_base.yaml}"
OUT_DIR="${OUT_DIR:-outputs/rssm_base}"
WANDB="${WANDB:-false}"

WANDB_FLAG=""
if [ "$WANDB" = "true" ]; then
    WANDB_FLAG="--wandb --wandb_project twit_wave --wandb_run a100_base_$SLURM_JOB_ID"
fi

python scripts/2_b_train_rssm.py \
    --cfg      "$CFG" \
    --data_dir data/processed \
    --out_dir  "$OUT_DIR" \
    --seed     42 \
    $WANDB_FLAG

echo "Training complete: $(date)"
echo "Checkpoint saved to: $OUT_DIR/best_model.pt"

# ─── optional: immediately run evaluation ─────────────────────────────────────
# Uncomment to chain prediction eval after training completes.
# python scripts/3_a_eval_prediction.py \
#     --model_dir "$OUT_DIR" \
#     --data_dir  data/processed \
#     --out_dir   outputs/eval/prediction \
#     --horizons  1 4 13
