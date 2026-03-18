# Set up environment
import os
import time
cpu = True
if cpu:
    N = 8
    os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={N}'

# Import all required libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import jax
import jax.numpy as jnp
import copy
import optax
from scipy import io
import equinox as eqx

# Jax enable x64
jax.config.update("jax_enable_x64", True)

# Import from tvboptim
from tvboptim.types import Parameter, BoundedParameter
from tvboptim.types.stateutils import show_parameters
from tvboptim.utils import set_cache_path, cache
from tvboptim.optim.optax import OptaxOptimizer
from tvboptim.optim.callbacks import MultiCallback, DefaultPrintCallback, SavingLossCallback

# Network dynamics imports
from tvboptim.experimental.network_dynamics import Network, solve, prepare
from tvboptim.experimental.network_dynamics.dynamics.tvb import ReducedWongWang
from tvboptim.experimental.network_dynamics.coupling import LinearCoupling, FastLinearCoupling
from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph, DenseGraph
from tvboptim.experimental.network_dynamics.solvers import Heun, BoundedSolver
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
from tvboptim.data import load_structural_connectivity, load_functional_connectivity

# BOLD monitoring
from tvboptim.observations.tvb_monitors.bold import Bold

# Observation functions
from tvboptim.observations.observation import compute_fc, fc_corr, rmse

# Caching utilities
from tvboptim.utils import set_cache_path, cache

# Set cache path for tvboptim
set_cache_path("./ei_tuning")


# Load structural connectivity with region labels
FC_allsubj = jnp.load('FC_allsubj_allparcellations_Hagmann.npy', allow_pickle=True).item()
# weights_allsubj = jnp.load('ADC_allsubj_Hagmann.npy', allow_pickle=True).item()
# weights_allsubj = jnp.load('gFA_allsubj_Hagmann.npy', allow_pickle=True).item()
# weights_allsubj = jnp.load('denasity_allsubj_Hagmann.npy', allow_pickle=True).item()
weights_allsubj = jnp.load('numb_allsubj_Hagmann.npy', allow_pickle=True).item()
len_allsubj = jnp.load('len_allsubj_Hagmann.npy', allow_pickle=True).item()

print(weights_allsubj.keys())
# Loop over all parcellations
# parcellations = [83, 463, 1015, 129, 234]


# Loop over patients
# for pat in weights_allsubj:
ctrl_keys = [k for k in weights_allsubj.keys() if k.startswith('ctrl')]
for pat in ctrl_keys[5:10]: 

    weights = jnp.array(weights_allsubj[pat][129])
    lengths = jnp.array(len_allsubj[pat][129])

    region_labels = jnp.load('reg_labels_Hagmann129.npy')

    # Normalize weights to [0, 1] range
    weights = weights / jnp.max(weights)
    n_nodes = weights.shape[0]

    # Delays
    speed = 3.0
    delays = lengths / speed

    # Load empirical functional connectivity as optimization target
    target_fcdiag1 = jnp.array(FC_allsubj[pat][129])
    fc_target = target_fcdiag1 - np.diag(np.diag(target_fcdiag1))


    # Define consistent color palette derived from cividis
    import matplotlib.colors as mcolors
    cividis_cmap = plt.cm.cividis
    cividis_colors = cividis_cmap(np.linspace(0, 1, 256))
    accent_blue = cividis_cmap(0.3)  # Dark blue from cividis
    accent_gold = cividis_cmap(0.85)  # Gold/yellow from cividis
    accent_mid = cividis_cmap(0.6)   # Mid-tone

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8.1, 4))

    # Structural weights - use cividis
    im1 = ax1.imshow(weights, cmap='cividis', vmin=0, vmax=1)
    ax1.set_title('Structural Weights')
    ax1.set_xlabel('Region')
    ax1.set_ylabel('Region')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    # Delays - use cividis
    im2 = ax2.imshow(delays, cmap='cividis')
    ax2.set_title('Transmission Delays (ms)')
    ax2.set_xlabel('Region')
    ax2.set_ylabel('Region')
    plt.colorbar(im2, ax=ax2, fraction=0.046, label='ms')

    # Target FC - use cividis
    im3 = ax3.imshow(fc_target, vmin=0, vmax=1.0, cmap='cividis')
    ax3.set_title('Target Functional Connectivity')
    ax3.set_xlabel('Region')
    ax3.set_ylabel('Region')
    plt.colorbar(im3, ax=ax3, label='Correlation', fraction=0.046)

    # plt.tight_layout()
    # plt.show()


    from typing import Tuple
    from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics
    from tvboptim.experimental.network_dynamics.core.bunch import Bunch

    class ReducedWongWangEIB(AbstractDynamics):
        """Two-population Reduced Wong-Wang model with E-I balance support"""

        STATE_NAMES = ('S_e', 'S_i')
        INITIAL_STATE = (0.001, 0.001)
        AUXILIARY_NAMES = ('H_e', 'H_i')

        DEFAULT_PARAMS = Bunch(
            # Excitatory population parameters
            a_e=310.0,         # Input gain parameter
            b_e=125.0,         # Input shift parameter [Hz]
            d_e=0.160,         # Input scaling parameter [s]
            gamma_e=0.641/1000,  # Kinetic parameter
            tau_e=100.0,       # NMDA decay time constant [ms]
            w_p=1.4,           # Excitatory recurrence weight
            W_e=1.0,           # External input scaling weight

            # Inhibitory population parameters
            a_i=615.0,         # Input gain parameter
            b_i=177.0,         # Input shift parameter [Hz]
            d_i=0.087,         # Input scaling parameter [s]
            gamma_i=1.0/1000,  # Kinetic parameter
            tau_i=10.0,        # NMDA decay time constant [ms]
            W_i=0.7,           # External input scaling weight

            # Synaptic weights
            J_N=0.15,          # NMDA current [nA]
            J_i=1.0,           # Inhibitory synaptic weight

            # External inputs
            I_o=0.382,         # Background input current
            I_ext=0.0,         # External stimulation current

            # Coupling parameters
            lamda=1.0,         # Lambda: inhibitory coupling scaling
        )

        COUPLING_INPUTS = {
            'coupling': 2,  # Long-range excitation and Feedforward inhibition
        }

        def dynamics(
            self,
            t: float,
            state: jnp.ndarray,
            params: Bunch,
            coupling: Bunch,
            external: Bunch
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Compute two-population Wong-Wang dynamics with dual coupling."""

            # Unpack state variables
            S_e = state[0]  # Excitatory synaptic gating
            S_i = state[1]  # Inhibitory synaptic gating

            # Unpack coupling inputs
            c_lre = params.J_N * coupling.coupling[0]  # Long-range excitation
            c_ffi = params.J_N * coupling.coupling[1]  # Feedforward inhibition

            # Excitatory population input
            J_N_S_e = params.J_N * S_e
            x_e_pre = (params.w_p * J_N_S_e - params.J_i * S_i +
                    params.W_e * params.I_o + c_lre + params.I_ext)

            # Excitatory transfer function
            x_e = params.a_e * x_e_pre - params.b_e
            H_e = x_e / (1.0 - jnp.exp(-params.d_e * x_e))

            # Excitatory dynamics
            dS_e_dt = -(S_e / params.tau_e) + (1.0 - S_e) * H_e * params.gamma_e

            # Inhibitory population input
            x_i_pre = J_N_S_e - S_i + params.W_i * params.I_o + params.lamda * c_ffi

            # Inhibitory transfer function
            x_i = params.a_i * x_i_pre - params.b_i
            H_i = x_i / (1.0 - jnp.exp(-params.d_i * x_i))

            # Inhibitory dynamics
            dS_i_dt = -(S_i / params.tau_i) + H_i * params.gamma_i

            # Package results
            derivatives = jnp.array([dS_e_dt, dS_i_dt])
            auxiliaries = jnp.array([H_e, H_i])

            return derivatives, auxiliaries
        
        
    from tvboptim.experimental.network_dynamics.coupling.base import InstantaneousCoupling

    class EIBLinearCoupling(InstantaneousCoupling):
        """EIB Linear coupling with separate excitatory and inhibitory weight matrices.

        This coupling produces two outputs:
            c_lre: Long-range excitation (wLRE * S_e)
            c_ffi: Feedforward inhibition (wFFI * S_e)

        Both couplings are driven by the excitatory activity (S_e) from other regions.
        """

        N_OUTPUT_STATES = 2  # Produces two coupling outputs

        DEFAULT_PARAMS = Bunch(
            wLRE = 1.0,  # Long-range excitation weight matrix
            wFFI = 1.0,  # Feedforward inhibition weight matrix
        )

        def pre(
            self,
            incoming_states: jnp.ndarray,
            local_states: jnp.ndarray,
            params: Bunch
        ) -> jnp.ndarray:
            """Pre-synaptic transformation: multiply S_e with wLRE and wFFI."""
            # incoming_states[0] is S_e from all source nodes
            S_e = incoming_states[0]  # [n_target, n_source]
            # Apply weights: element-wise multiply S_e with each weight matrix
            # params.wLRE and params.wFFI have shape [n_nodes, n_nodes]
            c_lre = S_e * params.wLRE  # [n_target, n_source]
            c_ffi = S_e * params.wFFI  # [n_target, n_source]

            # Stack into [2, n_target, n_source]
            return jnp.stack([c_lre, c_ffi], axis=0)

        def post(
            self,
            summed_inputs: jnp.ndarray,
            local_states: jnp.ndarray,
            params: Bunch
        ) -> jnp.ndarray:
            """Post-synaptic transformation: pass through without scaling."""
            return summed_inputs
        
        
    # Create network components
    graph = DenseGraph(weights, region_labels=region_labels)
    dynamics = ReducedWongWangEIB(J_i = jnp.ones((n_nodes)))

    # Initialize EIB coupling with dual weight matrices
    # wLRE and wFFI start as copies of structural connectivity
    coupling = EIBLinearCoupling(incoming_states=["S_e"])

    # Set the weight matrices to the proper shape based on structural connectivity
    # Both start as scaled versions of structural connectivity
    coupling.params.wLRE = jnp.ones((n_nodes, n_nodes)) #+ 0.8*fc_target  # [n_nodes, n_nodes]
    coupling.params.wFFI = jnp.ones((n_nodes, n_nodes)) #- 0.8*fc_target  # [n_nodes, n_nodes]

    # Small noise to break symmetry
    noise = AdditiveNoise(sigma=0.01, apply_to="S_e")

    # Assemble the network
    network = Network(
        dynamics=dynamics,
        coupling={'coupling': coupling},  # Both use same coupling but produce different outputs
        graph=graph,
        noise=noise
    )

    print(f"Network created with {n_nodes} nodes")

    # Prepare simulation: compile model and initialize state
    t1 = 5 * 60_000   # Simulation duration (ms) - 1 minute for initial transient
    dt = 4.0      # Integration timestep (ms) matching original script
    solver = BoundedSolver(Heun(), low=0.0, high=1.0)
    model, state = prepare(network, solver, t1=t1, dt=dt)

    # Run initial transient to reach quasi-stationary state
    print("Running initial transient simulation...")
    result_init = jax.block_until_ready(model(state))

    # Update network with final state as new initial conditions
    network.update_history(result_init)

    # Prepare for shorter simulations used in EI tuning
    bold_TR = 720.0
    model_short, state_short = prepare(network, solver, t1=bold_TR, dt=dt)

    print(f"Initial simulation complete. Final S_e mean: {result_init.data[-1, 0, :].mean():.3f}")
    print(f"Initial simulation complete. Final S_i mean: {result_init.data[-1, 1, :].mean():.3f}")

    # Create BOLD monitor - we'll monitor S_e (first state variable)
    # The BOLD period is 720ms (TR) as in the original script
    bold_monitor = Bold(
        period=bold_TR,           # BOLD sampling period (TR = 720 ms)
        downsample_period=4.0,  # Intermediate downsampling matches dt
        voi=0,                  # Monitor first state variable (S_e)
        history=result_init     # Use initial state as warm start for BOLD history
    )

    print("BOLD monitor initialized")


    # Will be populated after initial simulation completes
    model_eval, state_eval, _state = None, None, None

    def setup_eval_model():
        """Setup evaluation model for FC computation (called after initial simulation)."""
        global model_eval, state_eval, _state
        model_eval, state_eval = prepare(network, Heun(), t1=t1, dt=dt)
        _state = copy.deepcopy(state_eval)

    def eval_fc(J_i, wLRE, wFFI):
        setup_eval_model()
        """Evaluate FC for given parameters using a long simulation."""
        _state.dynamics.J_i = J_i
        _state.coupling.coupling.wLRE = wLRE
        _state.coupling.coupling.wFFI = wFFI

        # Run simulation
        raw_result = model_eval(_state)

        # Compute BOLD
        bold_signal = bold_monitor(raw_result)

        # Compute FC (skip initial transient)
        fc = compute_fc(bold_signal, skip_t=20)
        return fc

    print("Utility functions defined")


    def FIC_update_rule(J_i, raw_data, eta_fic=0.1, target_fic=0.25):
        """Update J_i using FIC algorithm to maintain E-I balance."""
        # Compute mean activity over the simulation window
        mean_S_i = jnp.mean(raw_data[:, 1], axis=0)  # Mean S_i over time [n_nodes]
        mean_S_e = jnp.mean(raw_data[:, 0], axis=0)  # Mean S_e over time [n_nodes]

        # FIC update rule: increase J_i if E activity is too high
        # When mean_S_e > target_fic, d_J_i is positive, increasing inhibition
        d_J_i = eta_fic * (mean_S_i * mean_S_e - target_fic * mean_S_i)
        J_i_new = J_i + d_J_i

        return J_i_new

    print("FIC update function defined")


    # FIC tuning parameters
    eta_fic = 0.5  # Learning rate for FIC
    target_fic = 0.25  # Target excitatory activity level
    n_fic_steps = 200  # Number of FIC iterations

    @cache("fic_tuning", redo=True)
    def run_fic_tuning():
        """Run FIC tuning loop with caching."""
        # Create a copy of the short simulation state for FIC tuning
        state_fic = copy.deepcopy(state_short)
        bold_monitor_fic = copy.deepcopy(bold_monitor)

        # Store initial state for comparison
        raw_result_pre_fic = model_short(state_fic)

        # Setup for tracking BOLD signal during FIC
        history_accessor = lambda tree: tree.history
        bold_signal_fic = []
        mean_S_e_history = []

        # Random key for noise updates
        key = jax.random.key(42)

        print("Starting FIC tuning...")

        # FIC tuning loop
        for i in range(n_fic_steps):
            # Simulate one time step
            raw_result = model_short(state_fic)

            # Compute BOLD signal for this step
            bold_result = bold_monitor_fic(raw_result)
            bold_signal_fic.append(bold_result.ys[0, 0, :])

            # Track mean excitatory activity
            mean_S_e = jnp.mean(raw_result.data[:, 0, :])
            mean_S_e_history.append(mean_S_e)

            # Update BOLD monitor history for next iteration
            new_history = jnp.roll(bold_monitor_fic.history, -raw_result.data.shape[0], axis=0)
            new_history = new_history.at[-raw_result.data.shape[0]:, :, :].set(raw_result.data[:, 0:1, :])
            bold_monitor_fic = eqx.tree_at(history_accessor, bold_monitor_fic, new_history)

            # Update initial conditions for next iteration
            state_fic.initial_state.dynamics = raw_result.data[-1]

            # Update noise realization
            key, subkey = jax.random.split(key, 2)
            state_fic._internal.noise_samples = jax.random.normal(key=subkey, shape=state_fic._internal.noise_samples.shape)

            # Apply FIC update rule
            state_fic.dynamics.J_i = FIC_update_rule(state_fic.dynamics.J_i, raw_result.data, eta_fic=eta_fic, target_fic=target_fic)

            if (i + 1) % 50 == 0:
                print(f"  Step {i+1}/{n_fic_steps}, Mean S_e: {mean_S_e:.4f}, Target: {target_fic:.4f}")

        # Final simulation after FIC
        raw_result_post_fic = model_short(state_fic)

        # Convert lists to arrays
        bold_signal_fic = jnp.array(bold_signal_fic)
        mean_S_e_history = jnp.array(mean_S_e_history)

        print(f"FIC tuning complete!")
        print(f"Final mean S_e: {mean_S_e_history[-1]:.4f} (target: {target_fic:.4f})")

        return {
            'state_fic': state_fic,
            'bold_monitor_fic': bold_monitor_fic,
            'raw_result_pre_fic': raw_result_pre_fic,
            'raw_result_post_fic': raw_result_post_fic,
            'bold_signal_fic': bold_signal_fic,
            'mean_S_e_history': mean_S_e_history
        }

    # Run FIC tuning (cached)
    fic_results = run_fic_tuning()
    state_fic = fic_results['state_fic']
    bold_monitor_fic = fic_results['bold_monitor_fic']
    raw_result_pre_fic = fic_results['raw_result_pre_fic']
    raw_result_post_fic = fic_results['raw_result_post_fic']
    bold_signal_fic = fic_results['bold_signal_fic']
    mean_S_e_history = fic_results['mean_S_e_history']


    fig = plt.figure(figsize=(8.1, 7))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Use cividis-derived colors for consistency
    target_color = accent_gold
    trace_color = accent_blue
    convergence_color = accent_mid

    # Top left: Pre-FIC timeseries
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(raw_result_pre_fic.data[:, 0, :], alpha=0.6, linewidth=0.8, color=trace_color)
    ax1.axhline(target_fic, color=target_color, linestyle='--', linewidth=2, label=f'Target ({target_fic})')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('S_e (Excitatory activity)')
    ax1.set_title('Before FIC')
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Top right: Post-FIC timeseries
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(raw_result_post_fic.data[:, 0, :], alpha=0.6, linewidth=0.8, color=trace_color)
    ax2.axhline(target_fic, color=target_color, linestyle='--', linewidth=2, label=f'Target ({target_fic})')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('S_e (Excitatory activity)')
    ax2.set_title('After FIC')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Bottom left: Convergence
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(mean_S_e_history, linewidth=2, label='Mean S_e', color=accent_blue)
    ax3.axhline(target_fic, color=target_color, linestyle='--', linewidth=2, label=f'Target ({target_fic})')
    ax3.set_xlabel('FIC iteration')
    ax3.set_ylabel('Mean S_e')
    ax3.set_title('FIC Convergence')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Bottom right: BOLD signal evolution with mean overlay
    ax4 = fig.add_subplot(gs[1, 1])
    # Plot all regions with light colors from cividis
    n_regions = bold_signal_fic.shape[1]
    colors_bold = cividis_cmap(np.linspace(0.2, 0.9, n_regions))
    for i in range(n_regions):
        ax4.plot(bold_signal_fic[:, i], alpha=0.3, linewidth=0.8, color=colors_bold[i])
    # Overlay mean BOLD signal in darker color
    mean_bold = np.mean(bold_signal_fic, axis=1)
    ax4.plot(mean_bold, color=accent_blue, linewidth=2.5, label='Mean', alpha=0.9)
    ax4.set_xlabel('BOLD time point (TR)')
    ax4.set_ylabel('BOLD signal')
    ax4.set_title('BOLD Signal Evolution (all regions + mean)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # plt.tight_layout()
    # plt.show()


    def EI_update_rule(wLRE, wFFI, fc_pred, fc_target, eta_eib=0.02):
        """Update wLRE and wFFI using EIB algorithm"""
        # Compute FC difference (positive means FC is too low, need more coupling)
        diff_FC = fc_target - fc_pred

        # Compute row-wise RMSE to weight updates by overall error magnitude
        rmse_FC = rmse(fc_target, fc_pred, axis=1)[:, None]  # [n_nodes, 1]

        # Update rules:
        # - Increase wLRE when FC is too low (strengthen excitation)
        # - Decrease wFFI when FC is too low (reduce inhibition)
        # Note: opposite signs ensure coordinated adjustment
        wLRE_new = jnp.clip(wLRE + eta_eib * diff_FC * rmse_FC, 0, None)
        wFFI_new = jnp.clip(wFFI - eta_eib * diff_FC * rmse_FC, 0, None)

        return wLRE_new, wFFI_new

    # print("EIB update function defined")



    # Combined FIC+EIB tuning parameters
    eta_fic = 0.1  # FIC learning rate
    eta_eib = 0.005  # EIB learning rate (smaller than FIC)
    window_size = 150  # Number of BOLD TRs for FC calculation
    n_eib_steps = 2000  # Total number of iterations
    snapshot_interval = 50  # Collect snapshots every N iterations

    @cache("eib_tuning", redo=True)
    def run_eib_tuning():
        """Run combined FIC+EIB tuning loop with caching."""
        # Initialize state for combined tuning
        state_ei = copy.deepcopy(state_fic)
        bold_monitor_ei = copy.deepcopy(bold_monitor_fic)
        history_accessor = lambda tree: tree.history

        # BOLD signal sliding window
        bold_signal = bold_signal_fic[-window_size:].reshape((window_size, 1, n_nodes))

        # Track metrics during tuning
        fc_correlations = []
        fc_rmse_values = []

        # Random key for noise
        key = jax.random.key(43)

        # Store initial state for comparison
        raw_result_pre_eib = model_short(state_ei)

        # Data collection for animation
        snapshots = {
            'iterations': [],
            'bold_signal': [],
            'raw_timeseries': [],
            'J_i': [],
            'fc_pred': [],
            'fc_corr': [],
            'fc_rmse': [],
            'wLRE': [],
            'wFFI': [],
        }

        # print("Starting combined FIC+EIB tuning...")

        # Combined FIC+EIB tuning loop
        for i in range(n_eib_steps):
            # 1. Simulate neural dynamics for one BOLD period
            raw_result = model_short(state_ei)

            # 2. Compute BOLD signal
            bold_result = bold_monitor_ei(raw_result)

            # 3. Update BOLD signal sliding window (rolling buffer)
            bold_signal = jnp.roll(bold_signal, -1, axis=0)
            bold_signal = bold_signal.at[-1, 0, :].set(bold_result.ys[0, 0, :])

            # 4. Update BOLD monitor history for hemodynamic state continuity
            new_history = jnp.roll(bold_monitor_ei.history, -raw_result.data.shape[0], axis=0)
            new_history = new_history.at[-raw_result.data.shape[0]:, :, :].set(raw_result.data[:, 0:1, :])
            bold_monitor_ei = eqx.tree_at(history_accessor, bold_monitor_ei, new_history)

            # 5. Update initial conditions for next simulation
            state_ei.initial_state.dynamics = raw_result.data[-1]

            # 6. Update noise realization
            key, subkey = jax.random.split(key, 2)
            state_ei._internal.noise_samples = jax.random.normal(key=subkey, shape=state_ei._internal.noise_samples.shape)

            # 7. Apply FIC update (every iteration)
            state_ei.dynamics.J_i = FIC_update_rule(
                state_ei.dynamics.J_i,
                raw_result.data,
                eta_fic=eta_fic,
                target_fic=target_fic
            )

            # 8. Apply EIB update
            # Compute FC from BOLD signal window
            fc_pred = compute_fc(bold_signal)

            # Update wLRE and wFFI using EIB rule
            wLRE_new, wFFI_new = EI_update_rule(
                state_ei.coupling.coupling.wLRE,
                state_ei.coupling.coupling.wFFI,
                fc_pred,
                fc_target,
                eta_eib=((i+1)/n_eib_steps) * eta_eib
            )
            state_ei.coupling.coupling.wLRE = wLRE_new
            state_ei.coupling.coupling.wFFI = wFFI_new

            # Track FC quality metrics
            fc_corr_val = fc_corr(fc_pred, fc_target)
            fc_rmse_val = jnp.sqrt(jnp.mean((fc_pred - fc_target)**2))
            fc_correlations.append(fc_corr_val)
            fc_rmse_values.append(fc_rmse_val)

            # Collect snapshots for animation every N iterations
            if (i + 1) % snapshot_interval == 0:
                snapshots['iterations'].append(i + 1)
                snapshots['bold_signal'].append(np.array(bold_signal[:, 0, :]))  # [window_size, n_nodes]
                snapshots['raw_timeseries'].append(np.array(raw_result.data[:, 0, :]))  # [time_steps, n_nodes]
                snapshots['J_i'].append(np.array(state_ei.dynamics.J_i.flatten()))  # [n_nodes]
                snapshots['wLRE'].append(np.array(state_ei.coupling.coupling.wLRE))  # [n_nodes, n_nodes]
                snapshots['wFFI'].append(np.array(state_ei.coupling.coupling.wFFI))  # [n_nodes, n_nodes]
                snapshots['fc_pred'].append(np.array(fc_pred))
                snapshots['fc_corr'].append(float(fc_corr_val))
                snapshots['fc_rmse'].append(float(fc_rmse_val))

                # Print progress
                print(f"  Step {i+1}/{n_eib_steps}, FC corr: {fc_corr_val:.4f}, FC RMSE: {fc_rmse_val:.4f}")

        # Store final state for comparison
        raw_result_post_eib = model_short(state_ei)

        # Convert metrics to arrays
        fc_correlations = jnp.array(fc_correlations)
        fc_rmse_values = jnp.array(fc_rmse_values)

        # print("Combined FIC+EIB tuning complete!")
        # print(f"Collected {len(snapshots['iterations'])} snapshots for animation")

        return {
            'state_ei': state_ei,
            'raw_result_pre_eib': raw_result_pre_eib,
            'raw_result_post_eib': raw_result_post_eib,
            'fc_correlations': fc_correlations,
            'fc_rmse_values': fc_rmse_values,
            'snapshots': snapshots
        }

    # Run EIB tuning (cached)
    eib_results = run_eib_tuning()
    state_ei = eib_results['state_ei']
    raw_result_pre_eib = eib_results['raw_result_pre_eib']
    raw_result_post_eib = eib_results['raw_result_post_eib']
    fc_correlations = eib_results['fc_correlations']
    fc_rmse_values = eib_results['fc_rmse_values']
    snapshots = eib_results['snapshots']


    print(fc_correlations[-1])
    print(fc_rmse_values[-1])
    data_to_save = {}
    for key in snapshots.keys():
        data_to_save[key] = snapshots[key][-1]
    data_to_save["fc_corr"] = fc_correlations[-1]
    data_to_save["fc_rmse"] = fc_rmse_values[-1]
    np.save('numb_param_129_'+ pat, data_to_save)