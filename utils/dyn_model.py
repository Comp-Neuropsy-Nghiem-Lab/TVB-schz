from typing import Tuple

import copy

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from tvboptim.experimental.network_dynamics import Network, prepare
from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.coupling.base import InstantaneousCoupling
from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics
from tvboptim.experimental.network_dynamics.graph import DenseGraph
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
from tvboptim.experimental.network_dynamics.solvers import BoundedSolver, Heun
from tvboptim.observations.observation import compute_fc, fc_corr, rmse
from tvboptim.observations.tvb_monitors.bold import Bold


class ReducedWongWangEIB(AbstractDynamics):
    STATE_NAMES = ("S_e", "S_i")
    INITIAL_STATE = (0.001, 0.001)
    AUXILIARY_NAMES = ("H_e", "H_i")
    DEFAULT_PARAMS = Bunch(
        a_e=310.0,
        b_e=125.0,
        d_e=0.160,
        gamma_e=0.641 / 1000,
        tau_e=100.0,
        w_p=1.4,
        W_e=1.0,
        a_i=615.0,
        b_i=177.0,
        d_i=0.087,
        gamma_i=1.0 / 1000,
        tau_i=10.0,
        W_i=0.7,
        J_N=0.15,
        J_i=1.0,
        I_o=0.382,
        I_ext=0.0,
        lamda=1.0,
    )
    COUPLING_INPUTS = {"coupling": 2}

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
        x_e_pre = (
            params.w_p * J_N_S_e
            - params.J_i * S_i
            + params.W_e * params.I_o
            + c_lre
            + params.I_ext
        )
        x_e = params.a_e * x_e_pre - params.b_e
        H_e = x_e / (1.0 - jnp.exp(-params.d_e * x_e))
        dS_e_dt = -(S_e / params.tau_e) + (1.0 - S_e) * H_e * params.gamma_e

        x_i_pre = J_N_S_e - S_i + params.W_i * params.I_o + params.lamda * c_ffi
        x_i = params.a_i * x_i_pre - params.b_i
        H_i = x_i / (1.0 - jnp.exp(-params.d_i * x_i))
        dS_i_dt = -(S_i / params.tau_i) + H_i * params.gamma_i

        return jnp.array([dS_e_dt, dS_i_dt]), jnp.array([H_e, H_i])


class EIBLinearCoupling(InstantaneousCoupling):
    N_OUTPUT_STATES = 2
    DEFAULT_PARAMS = Bunch(wLRE=1.0, wFFI=1.0)

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


def fic_update_rule(
    J_i: jnp.ndarray,
    raw_data: jnp.ndarray,
    eta_fic: float,
    target_fic: float,
) -> jnp.ndarray:
    mean_S_i = jnp.mean(raw_data[:, 1], axis=0)
    mean_S_e = jnp.mean(raw_data[:, 0], axis=0)
    return J_i + eta_fic * (mean_S_i * mean_S_e - target_fic * mean_S_i)


def eib_update_rule(
    wLRE: jnp.ndarray,
    wFFI: jnp.ndarray,
    fc_pred: jnp.ndarray,
    fc_target: jnp.ndarray,
    eta_eib: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    diff_FC = fc_target - fc_pred
    rmse_FC = rmse(fc_target, fc_pred, axis=1)[:, None]
    return (
        jnp.clip(wLRE + eta_eib * diff_FC * rmse_FC, 0, None),
        jnp.clip(wFFI - eta_eib * diff_FC * rmse_FC, 0, None),
    )


class EITuningDynamicsRunner:
    def __init__(self, weights, region_labels, sim_cfg):
        self.weights = weights
        self.region_labels = region_labels
        self.sim_cfg = sim_cfg
        self.n_nodes = weights.shape[0]
        self.dt = sim_cfg["dt"]
        self.bold_TR = sim_cfg["bold_TR"]
        self.t1 = sim_cfg["t1_transient"]
        self.solver = BoundedSolver(Heun(), low=0.0, high=1.0)

        graph = DenseGraph(weights, region_labels=region_labels)
        dynamics = ReducedWongWangEIB(J_i=jnp.ones((self.n_nodes,)))
        coupling = EIBLinearCoupling(incoming_states=["S_e"])
        coupling.params.wLRE = jnp.ones((self.n_nodes, self.n_nodes))
        coupling.params.wFFI = jnp.ones((self.n_nodes, self.n_nodes))

        self.network = Network(
            dynamics=dynamics,
            coupling={"coupling": coupling},
            graph=graph,
            noise=AdditiveNoise(sigma=0.01, apply_to="S_e"),
        )
        self.history_acc = lambda tree: tree.history

    def run_transient(self):
        model, state = prepare(self.network, self.solver, t1=self.t1, dt=self.dt)
        result_init = jax.block_until_ready(model(state))
        self.network.update_history(result_init)

        model_short, state_short = prepare(
            self.network,
            self.solver,
            t1=self.bold_TR,
            dt=self.dt,
        )
        bold_monitor = Bold(
            period=self.bold_TR,
            downsample_period=4.0,
            voi=0,
            history=result_init,
        )
        return result_init, model_short, state_short, bold_monitor

    def update_state_and_monitor(self, state, bold_monitor, raw_result, key):
        new_hist = jnp.roll(
            bold_monitor.history,
            -raw_result.data.shape[0],
            axis=0,
        )
        new_hist = new_hist.at[-raw_result.data.shape[0] :, :, :].set(
            raw_result.data[:, 0:1, :]
        )
        bold_monitor = eqx.tree_at(self.history_acc, bold_monitor, new_hist)
        state.initial_state.dynamics = raw_result.data[-1]
        key, subkey = jax.random.split(key, 2)
        state._internal.noise_samples = jax.random.normal(
            key=subkey,
            shape=state._internal.noise_samples.shape,
        )
        return state, bold_monitor, key

    def run_fic(self, model_short, state_short, bold_monitor, fic_cfg, log):
        state_fic = copy.deepcopy(state_short)
        bold_monitor_fic = copy.deepcopy(bold_monitor)
        bold_signal_fic = []
        key = jax.random.key(42)

        for i in range(fic_cfg["n_steps"]):
            raw_result = model_short(state_fic)
            bold_result = bold_monitor_fic(raw_result)
            bold_signal_fic.append(bold_result.ys[0, 0, :])

            state_fic, bold_monitor_fic, key = self.update_state_and_monitor(
                state_fic,
                bold_monitor_fic,
                raw_result,
                key,
            )
            state_fic.dynamics.J_i = fic_update_rule(
                state_fic.dynamics.J_i,
                raw_result.data,
                eta_fic=fic_cfg["eta"],
                target_fic=fic_cfg["target"],
            )

            if (i + 1) % 50 == 0:
                log.info(
                    f"  FIC {i+1}/{fic_cfg['n_steps']} | "
                    f"S_e={jnp.mean(raw_result.data[:,0,:]):.4f} "
                    f"(target {fic_cfg['target']})"
                )

        return state_fic, bold_monitor_fic, jnp.array(bold_signal_fic)

    def run_eib(
        self,
        model_short,
        state_fic,
        bold_monitor_fic,
        bold_signal_fic,
        fc_target,
        fic_cfg,
        eib_cfg,
        log,
    ):
        n_eib = eib_cfg["n_steps"]
        window_size = eib_cfg["window_size"]
        snap_interval = eib_cfg["snapshot_interval"]

        state_ei = copy.deepcopy(state_fic)
        bold_monitor_ei = copy.deepcopy(bold_monitor_fic)
        bold_signal = bold_signal_fic[-window_size:].reshape((window_size, 1, self.n_nodes))
        fc_correlations, fc_rmse_values = [], []
        snapshots = {
            k: []
            for k in [
                "iterations",
                "bold_signal",
                "raw_timeseries",
                "J_i",
                "fc_pred",
                "fc_corr",
                "fc_rmse",
                "wLRE",
                "wFFI",
            ]
        }
        key = jax.random.key(43)

        for i in range(n_eib):
            raw_result = model_short(state_ei)
            bold_result = bold_monitor_ei(raw_result)

            bold_signal = jnp.roll(bold_signal, -1, axis=0)
            bold_signal = bold_signal.at[-1, 0, :].set(bold_result.ys[0, 0, :])

            state_ei, bold_monitor_ei, key = self.update_state_and_monitor(
                state_ei,
                bold_monitor_ei,
                raw_result,
                key,
            )
            state_ei.dynamics.J_i = fic_update_rule(
                state_ei.dynamics.J_i,
                raw_result.data,
                eta_fic=fic_cfg["eta"],
                target_fic=fic_cfg["target"],
            )

            fc_pred = compute_fc(bold_signal)
            wLRE_new, wFFI_new = eib_update_rule(
                state_ei.coupling.coupling.wLRE,
                state_ei.coupling.coupling.wFFI,
                fc_pred,
                fc_target,
                eta_eib=((i + 1) / n_eib) * eib_cfg["eta"],
            )
            state_ei.coupling.coupling.wLRE = wLRE_new
            state_ei.coupling.coupling.wFFI = wFFI_new

            fc_corr_val = fc_corr(fc_pred, fc_target)
            fc_rmse_val = jnp.sqrt(jnp.mean((fc_pred - fc_target) ** 2))
            fc_correlations.append(fc_corr_val)
            fc_rmse_values.append(fc_rmse_val)

            if (i + 1) % snap_interval == 0:
                snapshots["iterations"].append(i + 1)
                snapshots["bold_signal"].append(np.array(bold_signal[:, 0, :]))
                snapshots["raw_timeseries"].append(np.array(raw_result.data[:, 0, :]))
                snapshots["J_i"].append(np.array(state_ei.dynamics.J_i.flatten()))
                snapshots["wLRE"].append(np.array(state_ei.coupling.coupling.wLRE))
                snapshots["wFFI"].append(np.array(state_ei.coupling.coupling.wFFI))
                snapshots["fc_pred"].append(np.array(fc_pred))
                snapshots["fc_corr"].append(float(fc_corr_val))
                snapshots["fc_rmse"].append(float(fc_rmse_val))
                log.info(
                    f"  EIB {i+1}/{n_eib} | "
                    f"FC corr: {fc_corr_val:.4f} | RMSE: {fc_rmse_val:.4f}"
                )

        return snapshots, fc_correlations, fc_rmse_values
