"""
utils/generate_shuffles.py

Run ONCE before submitting shuffle tasks.
Saves:
  $WORK/TVB-schz/shuffle/region_order/shuffle_parc{N}.npy  — one per parcellation
  $WORK/TVB-schz/shuffle/sub_order/keys_shuffled.npy        — subject order

Usage:
    python utils/generate_shuffles.py --config configs/ei_tuning_config.yaml
"""

import argparse
import os
import numpy as np
from pathlib import Path
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()

cfg         = yaml.safe_load(open(args.config))
data_dir    = Path(os.path.expandvars(cfg["data_dir"]))
shuffle_dir = Path(os.path.expandvars(cfg["shuffle_dir"]))
seed        = cfg.get("random_seed", 42)

region_dir = shuffle_dir / "region_order"
sub_dir    = shuffle_dir / "sub_order"
region_dir.mkdir(parents=True, exist_ok=True)
sub_dir.mkdir(parents=True, exist_ok=True)

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
    np.save(str(out), shuf)
    print(f"\nSubject shuffle: {keys[:3]} → {shuf[:3]} ...")
    print(f"Saved {out}")

print("\nDone.")