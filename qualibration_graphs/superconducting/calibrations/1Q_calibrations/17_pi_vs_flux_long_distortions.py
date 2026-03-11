# %%
from __future__ import annotations

import warnings
from dataclasses import asdict

import matplotlib.pyplot as plt
plt.style.use("sans_style_ppt")
import numpy as np
import xarray as xr
from calibration_utils.pi_flux import Parameters, fit_raw_data, log_fitted_results, plot_fit, process_raw_dataset
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from qualibration_libs.core import tracked_updates
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

description = """
Long cryoscope (π vs flux) calibration.

This protocol measures the effective flux-line step response per qubit by sweeping the XY-drive detuning and the Z-flux pulse duration, then extracting the instantaneous qubit frequency versus time.
It then Processes and fits the extracted flux response to model it as a sum of decaying exponentials and converts to usable filters.

Workflow:
For each qubit, sweep detuning over the configured span and flux-pulse duration over the configured time axis; play a constant Z pulse with amplitude `flux_amp`, then a chosen XY operation (default π), and measure I/Q or state.
Analysis: convert raw data to volts and extract the center frequency vs detuning at each time; derive the flux response using each qubit’s `freq_vs_flux_01_quad_term`; fit a sum of exponentials and determine the best components and DC term.
State update (optional): convert the fitted sum-of-exponentials to a cascade representation and write it to the state.json.


Prerequisites
- A valid rotation angle and threshold if using state discrimination
- Calibrated XYZ delay
- A calibrated pi-pulse
- Each qubit must have a known `freq_vs_flux_01_quad_term` stored in the state (obtained via (09)Ramsey vs flux calibration).

Outputs and state updates
- Results: processed dataset, fit results, and figures are saved under `node.results`.
- If `update_state=True` and fits succeed, the script updates `state.json` per qubit at `z.opx_output.exponential_filter` with the cascade coefficients `(A_c, tau_c)` derived from the fit.
REMINDER: Adding digital filters will add a global delay --> need to recalibrate IQ blobs (rotation_angle & ge_threshold) and (15)XYZ_delay. It is also worth looking at (09) Ramsey vs Flux as well
"""

node = QualibrationNode[Parameters, Quam](
    name="17_pi_vs_flux_long_distortions",
    description=description,
    parameters=Parameters(),
)


# %% {Custom_param}
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # Default to the dataset currently under inspection unless the GUI already set another id.
    #if node.parameters.load_data_id is None:
    node.parameters.load_data_id = 1781
    node.parameters.update_state = True
    pass


# Instantiate machine
node.machine = stored_machine = Quam.load()

# store fitting fractions set from GUI
loaded_fractions = node.parameters.fitting_base_fractions
stored_gui_update_flag = node.parameters.update_state_from_GUI
stored_update_state = node.parameters.update_state


# %% {Create_qua_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    operation_name = node.parameters.operation

    # Ensure operation exists and default to x180 if not
    for qubit in qubits:
        if hasattr(qubit.xy.operations, operation_name):
            continue
        warnings.warn(f"Qubit {qubit.name} has no operation '{operation_name}', defaulting to 'x180'")
        operation_name = "x180"

    operation_amp_scale = node.parameters.operation_amplitude_factor or 1.0

    # Frequency sweep
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span // 2, span // 2, step, dtype=np.int32)

    # Time sweep linear of log scale
    if node.parameters.time_axis == "linear":
        times = np.arange(
            node.parameters.min_wait_time_in_ns // 4,
            node.parameters.duration_in_ns // 4,
            max(node.parameters.time_step_in_ns, 4) // 4,
            dtype=np.int32,
        )
    else:
        times = np.logspace(
            np.log10(max(node.parameters.min_wait_time_in_ns // 4, 1)),
            np.log10(max(node.parameters.duration_in_ns // 4, 2)),
            max(node.parameters.time_step_num, 3),
            dtype=np.int32,
        )
        times = np.unique(times)

    flux_amps = [np.sqrt(-node.parameters.detuning_in_mhz * 1e6 / q.freq_vs_flux_01_quad_term) for q in qubits]

    # Sweep axes for data fetcher
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "qubit frequency", "units": "Hz"}),
        "time": xr.DataArray(4 * times, attrs={"long_name": "Flux pulse duration", "units": "ns"}),
    }

    # Track LO updates
    tracked_qubits = []
    if_update = []

    for q in qubits:
        # Decide if updating the LO is needed depending on the detuning request
        if (
            q.xy.intermediate_frequency
            - node.parameters.detuning_in_mhz * 1e6
            - node.parameters.frequency_span_in_mhz * 1e6 / 2
        ) < -400e6:
            node.parameters.reset_type = "thermal"  # Active reset will not work if the lo is changed
            warnings.warn(
                "Qubit LO has been changed to reach desired detuning, active reset will not work. Reset type changed to thermal."
            )
            if_update.append(0)
            # track the LO and IF changes to revert later
            with tracked_updates(q, auto_revert=False, dont_assign_to_none=False) as q_upd:
                rf_frequency = q_upd.xy.intermediate_frequency + q_upd.xy.opx_output.upconverter_frequency
                lo_frequency = q_upd.xy.opx_output.upconverter_frequency - node.parameters.detuning_in_mhz * 1e6
                if (q_upd.xy.opx_output.band == 3) and (lo_frequency < 6.5e9):
                    raise ValueError("Requested detuning is too large for the given MW FEM band")
                elif (q_upd.xy.opx_output.band == 2) and (lo_frequency < 4.5e9):
                    raise ValueError("Requested detuning is too large for the given MW FEM band")
                print(f"Updating {q_upd.name} LO to {lo_frequency}")
                q_upd.xy.opx_output.upconverter_frequency = lo_frequency
                q_upd.xy.RF_frequency -= node.parameters.detuning_in_mhz * 1e6
                tracked_qubits.append(q_upd)
        else:
            if_update.append(int(node.parameters.detuning_in_mhz))
            #if_update.append(int(node.parameters.detuning_in_mhz * 1e6))

    node.namespace["if_update"] = if_update
    node.namespace["tracked_qubits"] = tracked_qubits

    with program() as qua_prog:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        df = declare(int)
        t_delay = declare(int)

        for multiplexed_qubits in qubits.batch():
            # Place qubits to their respective flux point
            for qubit in multiplexed_qubits.values():
                if qubit.z is not None:
                    node.machine.initialize_qpu(target=qubit)
            align()
            # Averaging loop
            with for_(n, 0, n < node.parameters.num_shots, n + 1):
                save(n, n_st)
                # Qubit spectroscopy frequency loop
                with for_(*from_array(df, dfs)):
                    # Time delay loop
                    with for_each_(t_delay, times):
                        # Reset the qubits
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                            # Extra wait to ensure long distortions have fully decayed between experiment repetitions
                            qubit.wait(times.max())
                        align()

                        for i, qubit in multiplexed_qubits.items():
                            # Step the qubit spectroscopy tone frequency
                            qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency - if_update[i])
                            qubit.align()
                            # Play the flux pulse
                            qubit.z.play(
                                "const",
                                amplitude_scale=flux_amps[i] / qubit.z.operations["const"].amplitude,
                                duration=t_delay + 200,
                            )
                            # Wait for a variable time
                            qubit.xy.wait(t_delay)
                            # Play the qubit spectroscopy pulse
                            qubit.xy.play(operation_name, amplitude_scale=operation_amp_scale)
                            qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                            qubit.align()
                            qubit.wait(200)

                        for i, qubit in multiplexed_qubits.items():
                            if node.parameters.use_state_discrimination:
                                qubit.readout_state(state[i])
                                save(state[i], state_st[i])
                            else:
                                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])
                        align()

        with stream_processing():
            n_st.save("n")
            for i, _ in enumerate(qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(times)).buffer(len(dfs)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(times)).buffer(len(dfs)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(times)).buffer(len(dfs)).average().save(f"Q{i + 1}")

    node.namespace["qua_program"] = qua_prog


# %% {Simulate_qua_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}
    plt.show()


# %% {Execute_qua_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(data_fetcher.get("n", 0), node.parameters.num_shots, start_time=data_fetcher.t_start)
        node.log(job.execution_report())
    node.results["ds_raw"] = dataset


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    load_id = node.parameters.load_data_id
    node.load_from_id(load_id)
    node.parameters.load_data_id = load_id
    node.namespace["qubits"] = get_qubits(node)

    # Overwrite the loaded node parrameters with the ones defined from the GUI
    node.parameters.fitting_base_fractions = loaded_fractions
    node.parameters.update_state_from_GUI = stored_gui_update_flag
    node.parameters.update_state = stored_update_state
    if node.parameters.update_state_from_GUI or node.parameters.update_state:
        node.machine = stored_machine
    if node.parameters.update_state_from_GUI:
        node.parameters.update_state = True
        print("State update from GUI is enabled")


# %% {Process_raw}
@node.run_action(skip_if=node.parameters.simulate)
def process_raw(node: QualibrationNode[Parameters, Quam]):
    ds_raw = node.results["ds_raw"]
    ds_proc = process_raw_dataset(ds_raw, node)
    node.results["ds_proc_input"] = ds_proc


# %% {Analyze_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyze_data(node: QualibrationNode[Parameters, Quam]):
    ds_in = node.results["ds_proc_input"]
    ds, fit_results = fit_raw_data(ds_in, node)

    node.results["ds_proc"] = ds
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(fit_results, log_callable=node.log)


# %% {Plot}
@node.run_action(skip_if=node.parameters.simulate)
def plot_results(node: QualibrationNode[Parameters, Quam]):
    if "ds_proc" not in node.results:
        return
    ds = node.results["ds_proc"]
    qubits = node.namespace.get("qubits", get_qubits(node))
    fig = plot_fit(ds, qubits, node.results["fit_results"])
    plt.show()
    node.results["fitted_data"] = fig


# %% {Plot_raw_spectrogram}
@node.run_action(skip_if=node.parameters.simulate)
def plot_raw_spectrogram(node: QualibrationNode[Parameters, Quam]):
    """2-D spectrogram of IQ amplitude (or state) vs detuning and time.
    Lets you see the qubit peak drifting as the flux-pulse distortion decays."""
    if "ds_proc_input" not in node.results:
        return
    ds = node.results["ds_proc_input"]
    qubits = node.namespace.get("qubits", get_qubits(node))
    data_var = "state" if "state" in ds.data_vars else "IQ_abs"
    num_qubits = len(qubits)

    fig, axes = plt.subplots(1, num_qubits, figsize=(6 * num_qubits, 4), squeeze=False)
    for idx, q in enumerate(qubits):
        ax = axes[0, idx]
        data = ds[data_var].sel(qubit=q.name)  # dims: (detuning, time)
        im = ax.pcolormesh(
            ds["time"].values / 1e3,
            ds["detuning"].values / 1e6,
            data.values,
            cmap="viridis",
            shading="auto",
        )
        plt.colorbar(im, ax=ax, label=data_var)
        ax.set_xlabel("Time (µs)")
        ax.set_ylabel("Detuning (MHz)")
        ax.set_title(f"{q.name} — Raw spectrogram")
    fig.tight_layout()
    plt.show()
    node.results["fig_raw_spectrogram"] = fig


# %% {Plot_center_freqs}
@node.run_action(skip_if=node.parameters.simulate)
def plot_center_freqs(node: QualibrationNode[Parameters, Quam]):
    """Intermediate diagnostic: extracted center frequencies and flux response before fitting.
sian-fitted peak positions overlaid (red dots).
    Row 1 — extracted qubit center-frequency offset vs time.
    Row 2 — flux response (converted from center_freqs) vs time on both linear and log scales."""
    if "ds_proc" not in node.results or "ds_proc_input" not in node.results:
        return
    ds_in = node.results["ds_proc_input"]
    ds = node.results["ds_proc"]
    qubits = node.namespace.get("qubits", get_qubits(node))
    num_qubits = len(qubits)
    data_var = "state" if "state" in ds_in.data_vars else "IQ_abs"

    fig, axes = plt.subplots(3, num_qubits, figsize=(7 * num_qubits, 12), squeeze=False)
    for idx, q in enumerate(qubits):
        times_ns = ds_in["time"].values
        dfs_mhz = ds_in["detuning"].values / 1e6
        center_freqs = ds["center_freqs"].sel(qubit=q.name).values
        flux_response = ds["flux_response"].sel(qubit=q.name).values

        # Row 0: spectrogram + extracted peak overlay
        ax0 = axes[0, idx]
        spectro = ds_in[data_var].sel(qubit=q.name)  # (detuning, time)
        im = ax0.pcolormesh(times_ns / 1e3, dfs_mhz, spectro.values, cmap="viridis", shading="auto")
        plt.colorbar(im, ax=ax0, label=data_var)
        # center_freqs stored with a -detuning_in_mhz*1e6 offset; undo it to align with dfs axis
        peak_mhz = (center_freqs + node.parameters.detuning_in_mhz * 1e6) / 1e6
        in_range = (peak_mhz >= dfs_mhz.min()) & (peak_mhz <= dfs_mhz.max())
        ax0.plot(times_ns[in_range] / 1e3, peak_mhz[in_range], "r.", ms=3, label="extracted peak")
        ax0.plot(times_ns[~in_range] / 1e3, np.full((~in_range).sum(), np.nan), "rx", ms=5, label="out-of-range")
        ax0.set_xlabel("Time (µs)")
        ax0.set_ylabel("Detuning (MHz)")
        ax0.set_title(f"{q.name} — Spectrogram + peak")
        ax0.legend(loc="lower right", fontsize=8)

        # Row 1: center frequency vs time
        ax1 = axes[1, idx]
        ax1.plot(times_ns / 1e3, center_freqs / 1e6, ".--", lw=1)
        ax1.set_xlabel("Time (µs)")
        ax1.set_ylabel("Center freq. offset (MHz)")
        ax1.set_title(f"{q.name} — Extracted center frequency")
        ax1.grid(True)

        # Row 2: flux response (linear + log-x)
        ax2 = axes[2, idx]
        ax2.plot(times_ns, flux_response, ".--", lw=1, label="data")
        ax2.set_xlabel("Time (ns)")
        ax2.set_ylabel("Flux response (a.u.)")
        ax2.set_title(f"{q.name} — Flux response before fit")
        ax2.grid(True)
        # inset with log-time axis
        ax2_log = ax2.inset_axes([0.55, 0.05, 0.42, 0.45])
        ax2_log.plot(times_ns, flux_response, ".--", lw=0.8)
        ax2_log.set_xscale("log")
        ax2_log.set_xlabel("Time (ns)", fontsize=7)
        ax2_log.tick_params(labelsize=7)
        ax2_log.set_title("log scale", fontsize=7)
        ax2_log.grid(True)

    fig.tight_layout()
    plt.show()
    node.results["fig_center_freqs"] = fig


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    if not node.parameters.update_state:
        return
    qubits = node.namespace["qubits"]

    with node.record_state_updates():
        for q in qubits:
            res = node.results["fit_results"][q.name]
            # Support dict or dataclass
            fit_success = res["fit_successful"]
            if not fit_success:
                continue
            best_a_dc = res["a_dc"]
            components = res["a_tau_tuple"]
            A_list = [amp / best_a_dc for amp, _ in components]
            tau_list = [tau for _, tau in components]
            node.machine.qubits[q.name].z.opx_output.exponential_filter = list(zip(A_list, tau_list))
            print(f"Updated {q.name} filter to: {node.machine.qubits[q.name].z.opx_output.exponential_filter}")


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):

    for qubit in node.namespace.get("tracked_qubits", []):
        try:
            qubit.revert_changes()
        except Exception:
            pass
    node.save()


# %%
