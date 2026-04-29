#!/bin/bash
#SBATCH --job-name=twit_wave_l4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --gres=gpu:l4:1
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/train_l4_%j.out
#SBATCH --error=logs/train_l4_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=CHANGE_ME@nyu.edu

# ─── NYU HPC: Greene cluster  ────────────────────────────────────────────────
# Node type: g2-standard-12 (L4 24GB, 6 cores, 48GB RAM)
# L4 is slower than A100 — use rssm_small or rssm_base with batch_size=16.
# For rssm_base: set TRAIN_CFG to a modified train_base.yaml with batch_size=16.

set -euo pipefail

mkdir -p logs outputs/rssm_base_l4

echo "========================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "GPU:        L4 24GB"
echo "Start:      $(date)"
echo "========================================"

module purge
module load cuda/12.1
module load python/3.11

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate twit_wave

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ─── training ────────────────────────────────────────────────────────────────
CFG="${CFG:-configs/rssm_base.yaml}"
OUT_DIR="${OUT_DIR:-outputs/rssm_base_l4}"

python scripts/2_b_train_rssm.py \
    --cfg      "$CFG" \
    --data_dir data/processed \
    --out_dir  "$OUT_DIR" \
    --seed     42

echo "Training complete: $(date)"
