"""
utils/generate_shuffles.py

Run ONCE before submitting shuffle tasks.
Saves:
  $WORK/TVB-schz/shuffle/region_order/shuffle_parc{N}.npy  — one per parcellation
  $WORK/TVB-schz/shuffle/sub_order/keys_shuffled.npy        — subject order

Usage:
    python utils/generate_shuffles.py --config /configs/ei_tuning_config.yaml
"""

import argparse
import os
import numpy as np
from pathlib import Path
import yaml
import networkx as nx

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()

cfg         = yaml.safe_load(open(args.config))
data_dir    = Path(os.path.expandvars(cfg["data_dir"]))
shuffle_dir = Path(os.path.expandvars(cfg["shuffle_dir"]))
seed        = cfg.get("random_seed", 42)

region_dir = shuffle_dir / "region_order"
sub_dir    = shuffle_dir / "sub_order"
graph_dir = shuffle_dir / "graph_shuffles"
mean_sc_dir = shuffle_dir / "mean_SC"
region_dir.mkdir(parents=True, exist_ok=True)
sub_dir.mkdir(parents=True, exist_ok=True)
graph_dir.mkdir(parents=True, exist_ok=True)
mean_sc_dir.mkdir(parents=True, exist_ok=True)

# Region shuffle — one permutation per parcellation size
parcellations = [83, 129, 234, 463, 1015]
print("Region shuffles:")
for parc in parcellations:
    out = region_dir / f"shuffle_parc{parc}.npy"
    if out.exists():
        print(f"  parc{parc}: already exists, skipping")
        continue
    perm = np.arange(parc)
    np.random.default_rng(seed).shuffle(perm)
    np.save(str(out), perm)
    print(f"  parc{parc}: saved {out}")

# Subject shuffle
out = sub_dir / "keys_shuffled.npy"
if out.exists():
    print("\nSubject shuffle: already exists, skipping")
else:
    d    = np.load(data_dir / list(cfg["weight_types"].values())[0], allow_pickle=True).item()
    keys = sorted(d.keys())
    shuf = keys.copy()
    np.random.default_rng(seed).shuffle(shuf)
    np.save(str(out), np.array(shuf))
    print(f"\nSubject shuffle: {keys[:3]} → {shuf[:3]} ...")
    print(f"Saved {out}")

# randomly generated erdos-renyi graph shuffles for different sparsities
metrics = list(cfg["weight_types"].keys())  

mean_SC = {}
for metric in metrics:
    out = mean_sc_dir / f"{metric}_mean_SC.npy"
    if out.exists():
        print(f"  {metric}: already exists, skipping")
        mean_SC[metric] = np.load(str(out))
        continue
    
    sc_file = data_dir / "connectomes" / f"{metric}_allsubj_Hagmann.npy"
    sc_data = np.load(str(sc_file), allow_pickle=True).item()
    
    # stack all subject SC matrices: shape (n_subjects, n_regions, n_regions)
    matrices = np.array([sc_data[subj][83] for subj in sc_data if sc_data[subj] is not None])

    # average only over nonzero entries per element
    nonzero_sum   = np.sum(matrices, axis=0)
    nonzero_count = np.sum(matrices != 0, axis=0)
    mean_matrix   = np.where(nonzero_count > 0, nonzero_sum / nonzero_count, 0)

    np.save(str(out), mean_matrix)
    mean_SC[metric] = mean_matrix
    print(f"  {metric}: saved {out}  (shape {mean_matrix.shape})")    
    
# ── Erdős–Rényi graphs with different sparsities ─────────────────────────────
print("\nErdős–Rényi graphs:")
sparsity_levels = [0.2, 0.4, 0.6, 0.8, 1.0] 

for metric in metrics:
    n_regions = mean_SC[metric].shape[0]
    nonzero_mean = mean_SC[metric][mean_SC[metric] != 0].mean()

    for sparsity in sparsity_levels:
        p   = 1 - sparsity 
        out = graph_dir / f"{metric}_er_spars{int(sparsity * 100):03d}.npy"
        if out.exists():
            print(f"  {metric} sparsity={sparsity}: already exists, skipping")
            continue

        graphs = []
        for i in range(10):
            er = nx.erdos_renyi_graph(n_regions, p, seed=seed + i)
            matrix = nx.to_numpy_array(er)
            matrix[matrix != 0] = nonzero_mean
            graphs.append(matrix)

        np.save(str(out), np.array(graphs)) 
        print(f"  {metric} sparsity={sparsity:.1f}: saved {out}")


print("\nDone.")