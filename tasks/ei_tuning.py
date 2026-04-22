import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from tvboptim.utils import set_cache_path

from utils.dyn_model import EITuningDynamicsRunner
from utils.tools import expand_path, setup_ei_tuning_logger


class NormalEITuning:
    def __init__(self, config):
        self.cfg = config
        self.task = config["exp_name"]
        self.wtype = config["weight_type"]
        self.parc = config["parcellation"]
        self.pat = config["subject"]

        self.data_dir = expand_path(config["data_dir"])
        self.output_dir = expand_path(config["output_dir"])
        self.cache_dir = expand_path(config["cache_dir"])
        self.shuffle_dir = expand_path(config["shuffle_dir"])

        self.sim_cfg = config["simulation"]
        self.fic_cfg = config["fic"]
        self.eib_cfg = config["eib"]

        self.fc_subject = self.pat
        self.out_dir = self.output_dir / self.task / self.wtype / f"parc{self.parc}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.out_file = self.out_dir / f"{self.pat}.npy"

        self.log = setup_ei_tuning_logger(self.task, self.wtype, self.parc, self.pat)
        jax.config.update("jax_enable_x64", True)
        self.log.info(
            f"task={self.task} | weight={self.wtype} | "
            f"parcellation={self.parc} | subject={self.pat}"
        )
        self.log.info(f"JAX devices: {jax.devices()}")

        self.weights = None
        self.region_labels = None
        self.fc_target = None

    def load_data(self):
        self.log.info("Loading data...")
        fc_all = jnp.load(self.data_dir / self.cfg["fc_file"], allow_pickle=True).item()
        weights_all = jnp.load(self.data_dir / self.cfg["weight_types"][self.wtype], allow_pickle=True).item()
        lengths_all = jnp.load(self.data_dir / self.cfg["lengths_file"], allow_pickle=True).item()
        self.region_labels = jnp.load(self.data_dir / self.cfg["region_labels"][self.parc])

        self.weights = jnp.array(weights_all[self.pat][self.parc])
        _ = jnp.array(lengths_all[self.pat][self.parc])
        self.weights = self.weights / jnp.max(self.weights)

        fc_raw = jnp.array(fc_all[self.fc_subject][self.parc])
        self.fc_target = fc_raw - np.diag(np.diag(fc_raw))

    def save(self, snapshots, fc_correlations, fc_rmse_values):
        save_data = {k: snapshots[k][-1] for k in snapshots}
        save_data["fc_corr_history"] = np.array(fc_correlations)
        save_data["fc_rmse_history"] = np.array(fc_rmse_values)
        save_data["task"] = self.task
        save_data["subject"] = self.pat
        save_data["weight_type"] = self.wtype
        save_data["parcellation"] = self.parc
        np.save(str(self.out_file), save_data)
        self.log.info(f"Saved → {self.out_file}")

    def run(self):
        if self.out_file.exists():
            self.log.info(f"Already done, skipping: {self.out_file}")
            sys.exit(0)

        t_start = time.time()
        self.load_data()
        set_cache_path(str(self.cache_dir / self.task / self.wtype / f"parc{self.parc}" / self.pat))

        runner = EITuningDynamicsRunner(weights=self.weights, region_labels=self.region_labels, sim_cfg=self.sim_cfg)

        self.log.info(f"Running transient ({self.sim_cfg['t1_transient']/60000:.0f} min)...")
        result_init, model_short, state_short, bold_monitor = runner.run_transient()
        self.log.info(
            f"Transient done. S_e={result_init.data[-1,0,:].mean():.3f} "
            f"S_i={result_init.data[-1,1,:].mean():.3f}"
        )

        self.log.info(f"FIC tuning ({self.fic_cfg['n_steps']} steps)...")
        state_fic, bold_monitor_fic, bold_signal_fic = runner.run_fic(
            model_short=model_short,
            state_short=state_short,
            bold_monitor=bold_monitor,
            fic_cfg=self.fic_cfg,
            log=self.log,
        )

        self.log.info(f"EIB tuning ({self.eib_cfg['n_steps']} steps)...")
        snapshots, fc_correlations, fc_rmse_values = runner.run_eib(
            model_short=model_short,
            state_fic=state_fic,
            bold_monitor_fic=bold_monitor_fic,
            bold_signal_fic=bold_signal_fic,
            fc_target=self.fc_target,
            fic_cfg=self.fic_cfg,
            eib_cfg=self.eib_cfg,
            log=self.log,
        )

        elapsed = (time.time() - t_start) / 60
        self.log.info(
            f"Finished in {elapsed:.1f} min | "
            f"Final FC corr: {fc_correlations[-1]:.4f}"
        )
        self.save(snapshots, fc_correlations, fc_rmse_values)


class ShuffleRegionEITuning(NormalEITuning):
    def load_data(self):
        super().load_data()
        region_order_file = self.shuffle_dir / "region_order" / f"shuffle_parc{self.parc}.npy"
        if not region_order_file.exists():
            self.log.error(
                "Region shuffle file not found. Run: "
                "python utils/generate_shuffles.py --config configs/ei_tuning_config.yaml"
            )
            sys.exit(1)
        perm = np.load(str(region_order_file))
        self.weights = self.weights[perm, :][:, perm]
        self.log.info(f"Region shuffle applied (parc{self.parc})")


class ShuffleSubjectEITuning(NormalEITuning):
    def __init__(self, config):
        super().__init__(config)
        sub_order_file = self.shuffle_dir / "sub_order" / "keys_shuffled.npy"
        if not sub_order_file.exists():
            self.log.error(
                "Subject shuffle file not found. Run: "
                "python utils/generate_shuffles.py --config configs/ei_tuning_config.yaml"
            )
            sys.exit(1)

        weights_tmp = np.load(
            self.data_dir / self.cfg["weight_types"][self.wtype],
            allow_pickle=True,
        ).item()
        keys_original = sorted(weights_tmp.keys())
        keys_shuffled = np.load(str(sub_order_file), allow_pickle=True)
        keys_shuffled = (
            keys_shuffled.tolist()
            if keys_shuffled.ndim > 0
            else list(keys_shuffled.item())
        )
        self.fc_subject = keys_shuffled[keys_original.index(self.pat)]
        self.out_file = self.out_dir / f"{self.pat}_fc{self.fc_subject}.npy"
        self.log.info(f"SC: {self.pat}  FC target: {self.fc_subject}")

    def save(self, snapshots, fc_correlations, fc_rmse_values):
        save_data = {k: snapshots[k][-1] for k in snapshots}
        save_data["fc_corr_history"] = np.array(fc_correlations)
        save_data["fc_rmse_history"] = np.array(fc_rmse_values)
        save_data["task"] = self.task
        save_data["subject"] = self.pat
        save_data["weight_type"] = self.wtype
        save_data["parcellation"] = self.parc
        save_data["fc_subject"] = self.fc_subject
        np.save(str(self.out_file), save_data)
        self.log.info(f"Saved → {self.out_file}")

class ShuffleGraphEITuning(NormalEITuning):
    
    def __init__(self, config):
        super().__init__(config)
        self.graph_idx = int(config.get("graph_idx", config.get("extra", {}).get("graph_idx", 0)))
        self.sparsity = config.get("sparsity", config.get("extra", {}).get("sparsity", 0.4))
        self.pat = f"shuf_{self.graph_idx}"
        
        self.out_dir = self.out_dir / f"spar_{self.sparsity}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.out_file = self.out_dir / f"shuf_{self.pat}.npy"
        
    def load_data(self):
        self.log.info(f"Loading shuffled graph data (Idx: {self.graph_idx})...")
    
        sparsity_int = int(float(self.sparsity) * 100)
        file_name = f"{self.wtype}_er_spars{sparsity_int:03d}.npy"
        file_path = self.shuffle_dir / "graph_shuffles" / file_name
        
        all_graphs = np.load(str(file_path)) 
        self.weights = jnp.array(all_graphs[self.graph_idx])
        
        self.weights = self.weights / jnp.max(self.weights)
        self.region_labels = jnp.load(self.data_dir / self.cfg["region_labels"][self.parc])
        
        fc_all = jnp.load(self.data_dir / self.cfg["fc_file"], allow_pickle=True).item()
        fc_matrices = jnp.array([fc_all[subj][self.parc] for subj in fc_all if fc_all[subj] is not None])
        self.fc = jnp.mean(fc_matrices, axis=0)
        
        
normal = NormalEITuning
shuffle_region = ShuffleRegionEITuning
shuffle_subj = ShuffleSubjectEITuning
shuffle_graph = ShuffleGraphEITuning
