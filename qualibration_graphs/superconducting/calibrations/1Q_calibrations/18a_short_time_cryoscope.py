# %%
"""
        SHORT-TIME CRYOSCOPE

This node is designed for performing a short-time cryoscope calibration experiment on a specified qubit.
The cryoscope protocol is used to characterize and analyze the flux distortion at short time scales
(typically less than 100 ns).
"""

# %% {Imports}
import numpy as np
import xarray as xr
from calibration_utils.cryoscope import (
    ShortTimeCryoscopeParameters,
    analyze_and_plot_inverse_fir,
    baked_waveform,
    conv_causal,
    expdecay,
    process_raw_dataset_short_time,
    resample_to_target_rate,
)
from calibration_utils.node_utils import get_node_id_label
from qm.qua import *
from qm import SimulationConfig
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import lfilter

# %% {Node_parameters}
description = """
SHORT-TIME CRYOSCOPE
Characterises flux-line distortions at short time scales (< ~100 ns) using 1 ns-resolution
baked flux pulses in a Ramsey-style sequence.  The measured flux step response is then used
to extract either:
  - A set of FIR pre-distortion taps (iir_or_fir = 'fir'), or
  - A single IIR exponential correction (iir_or_fir = 'iir').

Prerequisites:
    - Resonator spectroscopy performed.
    - Qubit gates (x90) calibrated and saved to QuAM.

Next steps:
    - After a successful run the FIR/IIR filter is automatically written back to
      qubit.z.opx_output when update_state = True.
    - WARNING: digital filters add a global delay – recalibrate IQ blobs afterwards.
"""

node = QualibrationNode[ShortTimeCryoscopeParameters, Quam](
    name="18a_short_time_cryoscope",
    description=description,
    parameters=ShortTimeCryoscopeParameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[ShortTimeCryoscopeParameters, Quam]):
    """Allow local parameter override when running directly in a Python IDE."""
    node.parameters.qubits = ["q3"]
    node.parameters.num_shots = 2500
    pass


# Instantiate QuAM from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[ShortTimeCryoscopeParameters, Quam]):
    """Create baked waveforms and the QUA program for the short-time cryoscope."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    assert len(qubits) == 1, "This node only supports one qubit at a time."
    qubit = qubits[0]

    n_avg = node.parameters.num_shots
    cryoscope_len = node.parameters.cryoscope_len
    assert cryoscope_len % 16 == 0, "cryoscope_len must be a multiple of 16 ns"

    cryoscope_baking_len = cryoscope_len if node.parameters.only_baked_waveforms else 16
    amplitude = float(
        np.sqrt(-1e6 * node.parameters.detuning_target_in_MHz / qubit.freq_vs_flux_01_quad_term)
    )

    cryoscope_time = np.arange(0, cryoscope_len, 1)  # ns
    frames = np.linspace(0, 1, node.parameters.num_frames)

    baked_config = node.machine.generate_config()

    # Build 1 ns-resolution baked waveforms for the first cryoscope_baking_len ns
    baked_signals = baked_waveform(baked_config, amplitude, qubit, max_length=cryoscope_baking_len)

    node.namespace["baked_config"] = baked_config
    node.namespace["amplitude"] = amplitude
    node.namespace["frames"] = frames
    node.namespace["cryoscope_baking_len"] = cryoscope_baking_len

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "time": xr.DataArray(cryoscope_time, attrs={"long_name": "Cryoscope pulse duration", "units": "ns"}),
        "frame": xr.DataArray(frames, attrs={"long_name": "Frame rotation index"}),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=1)
        state = declare(int)
        state_st = declare_stream()
        virtual_detuning_phase = declare(fixed)
        idx = declare(int)
        frame = declare(fixed)
        t = declare(int)

        node.machine.initialize_qpu(flux_point=node.parameters.flux_point, target=qubit)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)

            # ---------- baked segment (ns 0 … cryoscope_baking_len-1) ----------
            with for_(idx, 0, idx < cryoscope_baking_len, idx + 1):
                assign(
                    virtual_detuning_phase,
                    Cast.mul_fixed_by_int(node.parameters.detuning_target_in_MHz * 1e-3, idx),
                )
                with for_each_(frame, frames.tolist()):
                    qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                    align()
                    qubit.xy.play("x90")
                    align()
                    wait(4)
                    align()
                    with switch_(idx):
                        for j in range(cryoscope_baking_len):
                            with case_(j):
                                baked_signals[j].run()
                    qubit.xy.wait((cryoscope_baking_len + 160) // 4)
                    qubit.xy.frame_rotation_2pi(-1 * virtual_detuning_phase)
                    qubit.xy.frame_rotation_2pi(frame)
                    qubit.xy.play("x90")
                    align()
                    qubit.resonator.measure("readout", qua_vars=(I[0], Q[0]))
                    assign(state, Cast.to_int(I[0] > qubit.resonator.operations["readout"].threshold))
                    save(state, state_st)

            # ---------- non-baked extension (ns cryoscope_baking_len … cryoscope_len-1) ----------
            if not node.parameters.only_baked_waveforms:
                with for_(t, 4, t < cryoscope_len // 4, t + 4):
                    with for_(idx, 0, idx < 16, idx + 1):
                        assign(
                            virtual_detuning_phase,
                            Cast.mul_fixed_by_int(node.parameters.detuning_target_in_MHz * 1e-3, idx + t * 4),
                        )
                        with for_each_(frame, frames.tolist()):
                            qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                            align()
                            qubit.xy.play("x90")
                            align()
                            wait(4)
                            align()
                            with switch_(idx):
                                for j in range(16):
                                    with case_(j):
                                        baked_signals[j].run()
                                        qubit.z.play(
                                            "const",
                                            duration=t,
                                            amplitude_scale=amplitude / qubit.z.operations["const"].amplitude,
                                        )
                            qubit.xy.wait((cryoscope_len + 160) // 4)
                            qubit.xy.frame_rotation_2pi(-1 * virtual_detuning_phase)
                            qubit.xy.frame_rotation_2pi(frame)
                            qubit.xy.play("x90")
                            align()
                            qubit.resonator.measure("readout", qua_vars=(I[0], Q[0]))
                            assign(state, Cast.to_int(I[0] > qubit.resonator.operations["readout"].threshold))
                            save(state, state_st)

        with stream_processing():
            n_st.save("n")
            state_st.buffer(len(frames)).buffer(cryoscope_len).average().save("state1")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[ShortTimeCryoscopeParameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.namespace["baked_config"]
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[ShortTimeCryoscopeParameters, Quam]):
    """Execute the QUA program and fetch the raw dataset."""
    qmm = node.machine.connect()
    config = node.namespace["baked_config"]
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher["n"],
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        node.log(job.execution_report())
    node.results["ds_raw"] = dataset


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[ShortTimeCryoscopeParameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[ShortTimeCryoscopeParameters, Quam]):
    """Extract phase → frequency → flux and compute the correction filter."""
    ds = node.results["ds_raw"]
    qubits = node.namespace["qubits"]
    qubit = qubits[0]

    # --- Phase / frequency / flux extraction ---
    ds = process_raw_dataset_short_time(ds, node)
    node.results["ds_raw"] = ds

    da = ds.flux.sel(qubit=qubit.name)

    # --- Plot phase, frequency, and flux ---
    title_suffix = f"\n{get_node_id_label(node)}"

    for var, ylabel, key in [
        ("state", "State", "figure_state"),
        ("phase", "Phase (rad)", "figure_phase"),
        ("frequencies", "Frequency (MHz)", "figure_freq"),
        ("flux", "Flux (V)", "figure_flux"),
    ]:
        da_plot = getattr(ds, var)
        if var == "state":
            da_plot = da_plot.sel(frame=0) if "frame" in da_plot.dims else da_plot
        fig, ax = plt.subplots()
        da_plot.sel(qubit=qubit.name).plot(ax=ax)
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs time{title_suffix}")
        node.results[key] = fig
        plt.show()

    # --- Filter extraction ---
    if node.parameters.iir_or_fir == "fir":
        existing_filter_length = (
            len(qubit.z.opx_output.feedforward_filter)
            if qubit.z.opx_output.feedforward_filter is not None
            else None
        )
        M = existing_filter_length if existing_filter_length is not None else node.parameters.num_inverse_firs

        normalized_response_raw = da.values / da.values[-10:].mean()
        normalized_response_2gsps = resample_to_target_rate(normalized_response_raw, 1.0, 0.5)
        time_2gsps = np.arange(len(normalized_response_2gsps)) * 0.5 + 0.5

        h_fir, inv_fir, best_reconstructed_response, fig_fir_fit, fig_inv_fir = analyze_and_plot_inverse_fir(
            response=normalized_response_2gsps,
            time=time_2gsps,
            Ts=0.5,
            L_values=node.parameters.num_forward_firs_values,
            lam1_values=node.parameters.lam1_values,
            lam2_values=node.parameters.lam2_values,
            M=M,
            sigma_ns=node.parameters.sigma_ns,
            lam_smooth=node.parameters.lam_smooth,
            method=node.parameters.method,
            verbose=True,
        )

        ideal_response = np.ones(len(da.values))
        predistorted_response = lfilter(inv_fir, 1, ideal_response)
        corrected_response = lfilter(h_fir, 1, predistorted_response)

        node.namespace["h_fir"] = h_fir
        node.namespace["inv_fir"] = inv_fir
        node.namespace["corrected_response"] = corrected_response
        node.namespace["best_reconstructed_response"] = best_reconstructed_response

        node.results["figure_fir_fit"] = fig_fir_fit
        node.results["figure_inv_fir"] = fig_inv_fir

        node.results["fit_results"] = {
            qubit.name: {
                "inverse_fir": inv_fir.tolist(),
                "forward_fir": h_fir.tolist(),
                "corrected_response": corrected_response.tolist(),
                "best_reconstructed_response": best_reconstructed_response.tolist(),
            }
        }

    elif node.parameters.iir_or_fir == "iir":
        first_vals = da.sel(time=slice(0, 1)).mean().values
        final_vals = da.isel(time=slice(-20, None)).mean().values
        exponential_fit_time_interval = [2, node.parameters.cryoscope_len - 1]
        start_index, end_index = exponential_fit_time_interval

        try:
            p0 = [final_vals, -1 + first_vals / final_vals, 10]
            fit, _ = curve_fit(
                expdecay,
                da.time[start_index:end_index],
                da[start_index:end_index],
                p0=p0,
                maxfev=10000,
                ftol=1e-8,
            )
        except RuntimeError:
            fit = p0
            node.log("Single exponential fit failed; using initial guess.")

        exponential_filter = [(round(fit[1], 6), round(fit[2], 6))]
        FIR_1exp = [1 / (1 + fit[1]), -np.exp(-1 / fit[2]) / (1 + fit[1])]
        IIR_1exp = [1, -np.exp(-1 / fit[2])]
        filtered_response = lfilter(FIR_1exp, IIR_1exp, da.values)

        node.namespace["exponential_filter"] = exponential_filter
        node.namespace["filtered_response"] = filtered_response

        node.results["fit_results"] = {
            qubit.name: {
                "filtered_response_1exp": filtered_response.tolist(),
                "exponential_filter": exponential_filter,
            }
        }

    # --- Final summary plot ---
    fig, ax = plt.subplots()
    normalized_da = da.values / da.values[-10:].mean()
    ax.plot(ds.time, normalized_da, label="data")
    if node.parameters.iir_or_fir == "iir":
        filtered = node.namespace["filtered_response"]
        ax.plot(ds.time, filtered / filtered[-10:].mean(), "--", label="predicted corrected response")
    else:
        corr = node.namespace["corrected_response"]
        ax.plot(ds.time, corr / corr[-10:].mean(), "--", label="predicted corrected response")
    ax.axhline(1.001, color="k", linewidth=0.8)
    ax.axhline(0.999, color="k", linewidth=0.8)
    ax.set_ylim([0.95, 1.05])
    ax.legend()
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Normalised amplitude")
    ax.set_title(f"Final results – {qubit.name}{title_suffix}")
    node.results["figure_final"] = fig
    plt.show()

    node.outcomes = {qubit.name: "successful"}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[ShortTimeCryoscopeParameters, Quam]):
    """Write the fitted filter coefficients back to QuAM."""
    if not node.parameters.update_state:
        return
    qubits = node.namespace["qubits"]
    with node.record_state_updates():
        for qubit in qubits:
            if node.outcomes.get(qubit.name) != "successful":
                continue
            if node.parameters.iir_or_fir == "fir":
                inv_fir = np.array(node.results["fit_results"][qubit.name]["inverse_fir"])
                if qubit.z.opx_output.feedforward_filter is None:
                    fir_list = inv_fir.tolist()
                else:
                    inv_fir_old = np.array(qubit.z.opx_output.feedforward_filter)
                    fir_list = conv_causal(inv_fir, inv_fir_old, N=len(inv_fir_old)).tolist()
                qubit.z.opx_output.feedforward_filter = fir_list
            else:
                exponential_filter = node.results["fit_results"][qubit.name]["exponential_filter"]
                qubit.z.opx_output.exponential_filter = [
                    *qubit.z.opx_output.exponential_filter,
                    *exponential_filter,
                ]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[ShortTimeCryoscopeParameters, Quam]):
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = node.machine
    node.save()
# %%
