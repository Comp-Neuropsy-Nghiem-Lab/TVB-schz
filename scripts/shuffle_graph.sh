#!/bin/bash
#SBATCH --job-name=ei_shuf_graph
#SBATCH --output=logs/slurm/shuf_graph_%A_%a.out
#SBATCH --error=logs/slurm/shuf_graph_%A_%a.err
#SBATCH --array=0-199%5nein    
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00 
#SBATCH --partition=2080-galvani

WEIGHT_TYPES=(ADC gFA density number)
PARCELLATIONS=(83)
SPARSITIES=(0.2 0.4 0.6 0.8 1.0)
N_SHUFFLES=10

# ──────────────────────────────────────────────────────────

mkdir -p logs/slurm
source ~/.bashrc
conda activate /home/buerger/bht335/.conda/envs/tvboptim || conda activate tvboptim

N_PARC=${#PARCELLATIONS[@]}
N_SPAR=${#SPARSITIES[@]}
N_WTYPE=${#WEIGHT_TYPES[@]}

I=$SLURM_ARRAY_TASK_ID

SHUF_IDX=$(( I % N_SHUFFLES ))
I=$(( I / N_SHUFFLES ))

SPAR_IDX=$(( I % N_SPAR ))
I=$(( I / N_SPAR ))

P_IDX=$(( I % N_PARC ))
I=$(( I / N_PARC ))

WTYPE_IDX=$(( I % N_WTYPE ))

WEIGHT_TYPE=${WEIGHT_TYPES[$WTYPE_IDX]}
PARCELLATION=${PARCELLATIONS[$P_IDX]}
SPARSITY=${SPARSITIES[$SPAR_IDX]}
GRAPH_IDX=$SHUF_IDX

echo "ID: $SLURM_ARRAY_TASK_ID | Type: $WEIGHT_TYPE | Sparsity: $SPARSITY | GraphIdx: $GRAPH_IDX"

export XLA_PYTHON_CLIENT_PREALLOCATE=false

srun python main.py \
    --task          ei_tuning \
    --exp           shuffle_graph \
    --weight_type   "$WEIGHT_TYPE" \
    --parcellation  "$PARCELLATION" \
    --extra         "sparsity=$SPARSITY,graph_idx=$GRAPH_IDX" \
    
