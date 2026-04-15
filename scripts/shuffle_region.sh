#!/bin/bash
#SBATCH --job-name=ei_shuf_reg
#SBATCH --output=logs/slurm/shuf_reg_%A_%a.out
#SBATCH --error=logs/slurm/shuf_reg_%A_%a.err
#SBATCH --array=0-53%20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --partition=2080-galvani

# ── EDIT to control what runs ─────────────────────────────
# PARCELLATIONS=(83 129 234 463 1015)
PARCELLATIONS=(129)
WEIGHT_TYPES=(ADC gFA density number)
# ──────────────────────────────────────────────────────────

mkdir -p logs/slurm
source ~/.bashrc
conda activate $WORK/.conda/tvboptim || conda activate tvboptim

SUBJECTS=($(python - <<EOF
import numpy as np, os, yaml
cfg      = yaml.safe_load(open("configs/ei_tuning.yaml"))
data_dir = os.path.expandvars(cfg["data_dir"])
d        = np.load(os.path.join(data_dir, list(cfg["weight_types"].values())[0]), allow_pickle=True).item()
print(" ".join(sorted(d.keys())))
EOF
))

N_SUBJ=${#SUBJECTS[@]}
N_PARC=${#PARCELLATIONS[@]}
W_IDX=$(( SLURM_ARRAY_TASK_ID / (N_PARC * N_SUBJ) ))
R=$(( SLURM_ARRAY_TASK_ID % (N_PARC * N_SUBJ) ))
P_IDX=$(( R / N_SUBJ ))
S_IDX=$(( R % N_SUBJ ))

WEIGHT_TYPE=${WEIGHT_TYPES[$W_IDX]}
PARCELLATION=${PARCELLATIONS[$P_IDX]}
SUBJECT=${SUBJECTS[$S_IDX]}
echo "$SLURM_ARRAY_TASK_ID | $WEIGHT_TYPE | parc$PARCELLATION | $SUBJECT"

srun python main.py \
    --task ei_tuning\
    --exp shuffle_region \
    --weight_type  "$WEIGHT_TYPE"  \
    --parcellation "$PARCELLATION" \
    --subject      "$SUBJECT"      \
