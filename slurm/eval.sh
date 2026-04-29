#!/bin/bash
#SBATCH --job-name=twit_wave_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=CHANGE_ME@nyu.edu

# ─── Full evaluation pipeline ────────────────────────────────────────────────
# Runs all five evaluation scripts sequentially.
# Set MODEL_DIR to the directory containing best_model.pt and norm_stats.json.
#
# Usage:
#   MODEL_DIR=outputs/rssm_base sbatch slurm/eval.sh

set -euo pipefail

mkdir -p logs

MODEL_DIR="${MODEL_DIR:-outputs/rssm_base}"
DATA_DIR="${DATA_DIR:-data/processed}"
EVAL_DIR="${EVAL_DIR:-outputs/eval}"
BASELINES_DIR="${BASELINES_DIR:-outputs/baselines}"

echo "========================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Model dir: $MODEL_DIR"
echo "Start:     $(date)"
echo "========================================"

module purge
module load cuda/12.1
module load python/3.11

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate twit_wave

# ── 3a: Prediction metrics ────────────────────────────────────────────────────
echo "[$(date)] Step 3a: Predictive evaluation"
python scripts/3_a_eval_prediction.py \
    --model_dir     "$MODEL_DIR" \
    --data_dir      "$DATA_DIR" \
    --baselines_dir "$BASELINES_DIR" \
    --out_dir       "$EVAL_DIR/prediction" \
    --horizons      1 4 13 \
    --splits        test1 test2

# ── 3b: KL analysis ───────────────────────────────────────────────────────────
echo "[$(date)] Step 3b: KL analysis"
python scripts/3_b_eval_kl.py \
    --model_dir  "$MODEL_DIR" \
    --data_dir   "$DATA_DIR" \
    --out_dir    "$EVAL_DIR/kl" \
    --splits     val test1 test2 \
    --spike_z    2.0

# ── 3c: Attention analysis ────────────────────────────────────────────────────
echo "[$(date)] Step 3c: Attention analysis"
python scripts/3_c_eval_attention.py \
    --model_dir "$MODEL_DIR" \
    --data_dir  "$DATA_DIR" \
    --out_dir   "$EVAL_DIR/attention" \
    --splits    test1 test2 \
    --top_n     50

# ── 3d: Latent clustering ──────────────────────────────────────────────────────
echo "[$(date)] Step 3d: Latent clustering"
python scripts/3_d_eval_latent.py \
    --model_dir  "$MODEL_DIR" \
    --data_dir   "$DATA_DIR" \
    --out_dir    "$EVAL_DIR/latent" \
    --splits     train val test1 test2 \
    --n_clusters 5

# ── 3e: Counterfactual probing ────────────────────────────────────────────────
echo "[$(date)] Step 3e: Counterfactual probing"
python scripts/3_e_eval_counterfactual.py \
    --model_dir "$MODEL_DIR" \
    --data_dir  "$DATA_DIR" \
    --out_dir   "$EVAL_DIR/counterfactual" \
    --run_all_experiments \
    --run_residual_corr

echo "========================================"
echo "All evaluation complete: $(date)"
echo "Results in: $EVAL_DIR"
echo "========================================"
