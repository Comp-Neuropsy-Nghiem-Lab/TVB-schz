"""
tasks/ei_tuning.py

Three tasks, controlled via --task argument:
  normal         — fit subject's own SC to own FC
  shuffle_region — shuffle SC region order, fit own FC
  shuffle_subj   — use subject's SC, fit a different subject's FC

One subject per SLURM job. JAX runs on GPU automatically.

Usage:
    python tasks/ei_tuning.py \
        --task normal \
        --weight_type ADC \
        --parcellation 83 \
        --subject ctrl0 \
        --config configs/ei_tuning_config.yaml
"""

import argparse
import os
import sys
import time
import copy
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml

# ── Parse args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--task",          required=True,
                    choices=["normal", "shuffle_region", "shuffle_subj"])
parser.add_argument("--weight_type",   required=True)
parser.add_argument("--parcellation",  required=True, type=int)
parser.add_argument("--subject",       required=True)
parser.add_argument("--config",        required=True)
args = parser.parse_args()

# ── Load config ───────────────────────────────────────────────────────────────
with open(args.config) as f:
    cfg = yaml.safe_load(f)

def expand(p):
    return Path(os.path.expandvars(str(p)))

data_dir    = expand(cfg["data_dir"])
output_dir  = expand(cfg["output_dir"])
cache_dir   = expand(cfg["cache_dir"])
shuffle_dir = expand(cfg["shuffle_dir"])

task  = args.task
wtype = args.weight_type
parc  = args.parcellation
pat   = args.subject

# ── Logging ───────────────────────────────────────────────────────────────────
log_dir = Path(f"logs/{task}/{wtype}/parc{parc}")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_dir / f"{pat}.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)
log.info(f"task={task} | weight={wtype} | parcellation={parc} | subject={pat}")

# ── JAX imports (after logging setup) ────────────────────────────────────────
import jax
import jax.numpy as jnp
import equinox as eqx

jax.config.update("jax_enable_x64", True)
log.info(f"JAX devices: {jax.devices()}")

from tvboptim.utils import set_cache_path
from tvboptim.experimental.network_dynamics import Network, prepare
from tvboptim.experimental.network_dynamics.graph import DenseGraph
from tvboptim.experimental.network_dynamics.solvers import Heun, BoundedSolver
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics
from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.coupling.base import InstantaneousCoupling
from tvboptim.observations.tvb_monitors.bold import Bold
from tvboptim.observations.observation import compute_fc, fc_corr, rmse

# ── Model ─────────────────────────────────────────────────────────────────────

class ReducedWongWangEIB(AbstractDynamics):
    STATE_NAMES     = ('S_e', 'S_i')
    INITIAL_STATE   = (0.001, 0.001)
    AUXILIARY_NAMES = ('H_e', 'H_i')
    DEFAULT_PARAMS  = Bunch(
        a_e=310.0, b_e=125.0, d_e=0.160, gamma_e=0.641/1000, tau_e=100.0,
        w_p=1.4,   W_e=1.0,
        a_i=615.0, b_i=177.0, d_i=0.087, gamma_i=1.0/1000,  tau_i=10.0, W_i=0.7,
        J_N=0.15,  J_i=1.0,
        I_o=0.382, I_ext=0.0,
        lamda=1.0,
    )
    COUPLING_INPUTS = {'coupling': 2}

    def dynamics(
        self,
        t: float,
        state: jnp.ndarray,
        params: Bunch,
        coupling: Bunch,
        external: Bunch,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        S_e, S_i = state[0], state[1]
        c_lre = params.J_N * coupling.coupling[0]
        c_ffi = params.J_N * coupling.coupling[1]

        J_N_S_e = params.J_N * S_e
        x_e_pre = (params.w_p * J_N_S_e - params.J_i * S_i
                   + params.W_e * params.I_o + c_lre + params.I_ext)
        x_e     = params.a_e * x_e_pre - params.b_e
        H_e     = x_e / (1.0 - jnp.exp(-params.d_e * x_e))
        dS_e_dt = -(S_e / params.tau_e) + (1.0 - S_e) * H_e * params.gamma_e

        x_i_pre = J_N_S_e - S_i + params.W_i * params.I_o + params.lamda * c_ffi
        x_i     = params.a_i * x_i_pre - params.b_i
        H_i     = x_i / (1.0 - jnp.exp(-params.d_i * x_i))
        dS_i_dt = -(S_i / params.tau_i) + H_i * params.gamma_i

        return jnp.array([dS_e_dt, dS_i_dt]), jnp.array([H_e, H_i])


class EIBLinearCoupling(InstantaneousCoupling):
    N_OUTPUT_STATES = 2
    DEFAULT_PARAMS  = Bunch(wLRE=1.0, wFFI=1.0)

    def pre(
        self,
        incoming_states: jnp.ndarray,
        local_states: jnp.ndarray,
        params: Bunch,
    ) -> jnp.ndarray:
        S_e = incoming_states[0]
        return jnp.stack([S_e * params.wLRE, S_e * params.wFFI], axis=0)

    def post(
        self,
        summed_inputs: jnp.ndarray,
        local_states: jnp.ndarray,
        params: Bunch,
    ) -> jnp.ndarray:
        return summed_inputs


# ── Update rules ──────────────────────────────────────────────────────────────

def FIC_update_rule(
    J_i: jnp.ndarray,
    raw_data: jnp.ndarray,
    eta_fic: float,
    target_fic: float,
) -> jnp.ndarray:
    mean_S_i = jnp.mean(raw_data[:, 1], axis=0)
    mean_S_e = jnp.mean(raw_data[:, 0], axis=0)
    return J_i + eta_fic * (mean_S_i * mean_S_e - target_fic * mean_S_i)


def EI_update_rule(
    wLRE: jnp.ndarray,
    wFFI: jnp.ndarray,
    fc_pred: jnp.ndarray,
    fc_target: jnp.ndarray,
    eta_eib: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    diff_FC = fc_target - fc_pred
    rmse_FC = rmse(fc_target, fc_pred, axis=1)[:, None]
    return (jnp.clip(wLRE + eta_eib * diff_FC * rmse_FC, 0, None),
            jnp.clip(wFFI - eta_eib * diff_FC * rmse_FC, 0, None))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    sim_cfg = cfg["simulation"]
    fic_cfg = cfg["fic"]
    eib_cfg = cfg["eib"]

    # Skip if already done
    out_dir = output_dir / task / wtype / f"parc{parc}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if task == "shuffle_subj":
        sub_order_file = shuffle_dir / "sub_order" / "keys_shuffled.npy"
        if not sub_order_file.exists():
            log.error("Subject shuffle file not found. Run: python utils/generate_shuffles.py --config configs/ei_tuning_config.yaml")
            sys.exit(1)
        weights_tmp   = np.load(data_dir / cfg["weight_types"][wtype], allow_pickle=True).item()
        keys_original = sorted(weights_tmp.keys())
        keys_shuffled = np.load(str(sub_order_file), allow_pickle=True).tolist()
        fc_subject    = keys_shuffled[keys_original.index(pat)]
        out_file = out_dir / f"{pat}_fc{fc_subject}.npy"
        log.info(f"SC: {pat}  FC target: {fc_subject}")
    else:
        out_file = out_dir / f"{pat}.npy"

    if out_file.exists():
        log.info(f"Already done, skipping: {out_file}")
        sys.exit(0)

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("Loading data...")
    fc_all        = jnp.load(data_dir / cfg["fc_file"],    allow_pickle=True).item()
    weights_all   = jnp.load(data_dir / cfg["weight_types"][wtype], allow_pickle=True).item()
    lengths_all   = jnp.load(data_dir / cfg["lengths_file"], allow_pickle=True).item()
    region_labels = jnp.load(data_dir / cfg["region_labels"][parc])

    weights = jnp.array(weights_all[pat][parc])
    lengths = jnp.array(lengths_all[pat][parc])

    # Task 2: shuffle SC region order
    if task == "shuffle_region":
        region_order_file = shuffle_dir / "region_order" / f"shuffle_parc{parc}.npy"
        if not region_order_file.exists():
            log.error("Region shuffle file not found. Run: python utils/generate_shuffles.py --config configs/ei_tuning_config.yaml")
            sys.exit(1)
        perm    = np.load(str(region_order_file))
        weights = weights[perm, :][:, perm]
        log.info(f"Region shuffle applied (parc{parc})")

    weights   = weights / jnp.max(weights)
    n_nodes   = weights.shape[0]
    fc_raw    = jnp.array(fc_all[fc_subject if task == "shuffle_subj" else pat][parc])
    fc_target = fc_raw - np.diag(np.diag(fc_raw))

    set_cache_path(str(cache_dir / task / wtype / f"parc{parc}" / pat))

    # ── Build network ─────────────────────────────────────────────────────────
    graph    = DenseGraph(weights, region_labels=region_labels)
    dynamics = ReducedWongWangEIB(J_i=jnp.ones((n_nodes,)))
    coupling = EIBLinearCoupling(incoming_states=["S_e"])
    coupling.params.wLRE = jnp.ones((n_nodes, n_nodes))
    coupling.params.wFFI = jnp.ones((n_nodes, n_nodes))
    network  = Network(
        dynamics=dynamics,
        coupling={'coupling': coupling},
        graph=graph,
        noise=AdditiveNoise(sigma=0.01, apply_to="S_e"),
    )

    dt      = sim_cfg["dt"]
    bold_TR = sim_cfg["bold_TR"]
    t1      = sim_cfg["t1_transient"]
    solver  = BoundedSolver(Heun(), low=0.0, high=1.0)

    # ── Initial transient ─────────────────────────────────────────────────────
    log.info(f"Running transient ({t1/60000:.0f} min)...")
    model, state = prepare(network, solver, t1=t1, dt=dt)
    result_init  = jax.block_until_ready(model(state))
    network.update_history(result_init)
    log.info(f"Transient done. S_e={result_init.data[-1,0,:].mean():.3f} "
             f"S_i={result_init.data[-1,1,:].mean():.3f}")

    model_short, state_short = prepare(network, solver, t1=bold_TR, dt=dt)
    bold_monitor = Bold(period=bold_TR, downsample_period=4.0,
                        voi=0, history=result_init)
    history_acc  = lambda tree: tree.history

    # ── FIC tuning ────────────────────────────────────────────────────────────
    log.info(f"FIC tuning ({fic_cfg['n_steps']} steps)...")
    state_fic        = copy.deepcopy(state_short)
    bold_monitor_fic = copy.deepcopy(bold_monitor)
    bold_signal_fic  = []
    key = jax.random.key(42)

    for i in range(fic_cfg["n_steps"]):
        raw_result  = model_short(state_fic)
        bold_result = bold_monitor_fic(raw_result)
        bold_signal_fic.append(bold_result.ys[0, 0, :])

        new_hist = jnp.roll(bold_monitor_fic.history, -raw_result.data.shape[0], axis=0)
        new_hist = new_hist.at[-raw_result.data.shape[0]:, :, :].set(raw_result.data[:, 0:1, :])
        bold_monitor_fic = eqx.tree_at(history_acc, bold_monitor_fic, new_hist)
        state_fic.initial_state.dynamics = raw_result.data[-1]
        key, subkey = jax.random.split(key, 2)
        state_fic._internal.noise_samples = jax.random.normal(
            key=subkey, shape=state_fic._internal.noise_samples.shape)
        state_fic.dynamics.J_i = FIC_update_rule(
            state_fic.dynamics.J_i, raw_result.data,
            eta_fic=fic_cfg["eta"], target_fic=fic_cfg["target"])

        if (i + 1) % 50 == 0:
            log.info(f"  FIC {i+1}/{fic_cfg['n_steps']} | "
                     f"S_e={jnp.mean(raw_result.data[:,0,:]):.4f} "
                     f"(target {fic_cfg['target']})")

    bold_signal_fic = jnp.array(bold_signal_fic)

    # ── EIB tuning ────────────────────────────────────────────────────────────
    n_eib         = eib_cfg["n_steps"]
    window_size   = eib_cfg["window_size"]
    snap_interval = eib_cfg["snapshot_interval"]

    log.info(f"EIB tuning ({n_eib} steps)...")
    state_ei        = copy.deepcopy(state_fic)
    bold_monitor_ei = copy.deepcopy(bold_monitor_fic)
    bold_signal     = bold_signal_fic[-window_size:].reshape((window_size, 1, n_nodes))
    fc_correlations, fc_rmse_values = [], []
    snapshots = {k: [] for k in ['iterations', 'bold_signal', 'raw_timeseries',
                                  'J_i', 'fc_pred', 'fc_corr', 'fc_rmse', 'wLRE', 'wFFI']}
    key = jax.random.key(43)

    for i in range(n_eib):
        raw_result  = model_short(state_ei)
        bold_result = bold_monitor_ei(raw_result)

        bold_signal = jnp.roll(bold_signal, -1, axis=0)
        bold_signal = bold_signal.at[-1, 0, :].set(bold_result.ys[0, 0, :])

        new_hist = jnp.roll(bold_monitor_ei.history, -raw_result.data.shape[0], axis=0)
        new_hist = new_hist.at[-raw_result.data.shape[0]:, :, :].set(raw_result.data[:, 0:1, :])
        bold_monitor_ei = eqx.tree_at(history_acc, bold_monitor_ei, new_hist)
        state_ei.initial_state.dynamics = raw_result.data[-1]
        key, subkey = jax.random.split(key, 2)
        state_ei._internal.noise_samples = jax.random.normal(
            key=subkey, shape=state_ei._internal.noise_samples.shape)

        state_ei.dynamics.J_i = FIC_update_rule(
            state_ei.dynamics.J_i, raw_result.data,
            eta_fic=fic_cfg["eta"], target_fic=fic_cfg["target"])

        fc_pred = compute_fc(bold_signal)
        wLRE_new, wFFI_new = EI_update_rule(
            state_ei.coupling.coupling.wLRE,
            state_ei.coupling.coupling.wFFI,
            fc_pred, fc_target,
            eta_eib=((i + 1) / n_eib) * eib_cfg["eta"])
        state_ei.coupling.coupling.wLRE = wLRE_new
        state_ei.coupling.coupling.wFFI = wFFI_new

        fc_corr_val = fc_corr(fc_pred, fc_target)
        fc_rmse_val = jnp.sqrt(jnp.mean((fc_pred - fc_target) ** 2))
        fc_correlations.append(fc_corr_val)
        fc_rmse_values.append(fc_rmse_val)

        if (i + 1) % snap_interval == 0:
            snapshots['iterations'].append(i + 1)
            snapshots['bold_signal'].append(np.array(bold_signal[:, 0, :]))
            snapshots['raw_timeseries'].append(np.array(raw_result.data[:, 0, :]))
            snapshots['J_i'].append(np.array(state_ei.dynamics.J_i.flatten()))
            snapshots['wLRE'].append(np.array(state_ei.coupling.coupling.wLRE))
            snapshots['wFFI'].append(np.array(state_ei.coupling.coupling.wFFI))
            snapshots['fc_pred'].append(np.array(fc_pred))
            snapshots['fc_corr'].append(float(fc_corr_val))
            snapshots['fc_rmse'].append(float(fc_rmse_val))
            log.info(f"  EIB {i+1}/{n_eib} | "
                     f"FC corr: {fc_corr_val:.4f} | RMSE: {fc_rmse_val:.4f}")

    elapsed = (time.time() - t_start) / 60
    log.info(f"Finished in {elapsed:.1f} min | "
             f"Final FC corr: {fc_correlations[-1]:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_data = {k: snapshots[k][-1] for k in snapshots}
    save_data["fc_corr_history"] = np.array(fc_correlations)
    save_data["fc_rmse_history"] = np.array(fc_rmse_values)
    save_data["task"]            = task
    save_data["subject"]         = pat
    save_data["weight_type"]     = wtype
    save_data["parcellation"]    = parc
    if task == "shuffle_subj":
        save_data["fc_subject"]  = fc_subject

    np.save(str(out_file), save_data)
    log.info(f"Saved → {out_file}")


if __name__ == "__main__":
    main()