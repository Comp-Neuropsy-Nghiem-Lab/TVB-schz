#!/bin/bash
#SBATCH --job-name=classification
#SBATCH --output=logs/slurm/classification_%A_%a.out
#SBATCH --error=logs/slurm/classification_%A_%a.err
#SBATCH --array=0-5
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=cpu-galvani

# ── EDIT to control what runs ─────────────────────────────
CLASSIFIERS=(SGD RF)
EXPS=(normal shuffle_region shuffle_subj)
# ──────────────────────────────────────────────────────────

mkdir -p logs/slurm
source ~/.bashrc
conda activate $WORK/.conda/tvboptim || conda activate tvboptim

N_EXP=${#EXPS[@]}
C_IDX=$(( SLURM_ARRAY_TASK_ID / N_EXP ))
E_IDX=$(( SLURM_ARRAY_TASK_ID % N_EXP ))

CLASSIFIER=${CLASSIFIERS[$C_IDX]}
EXP=${EXPS[$E_IDX]}
echo "$SLURM_ARRAY_TASK_ID | $CLASSIFIER | $EXP"

srun python main.py \
    --task       classification \
    --classifier "$CLASSIFIER"  \
    --exp        "$EXP"
