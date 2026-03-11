from typing import List, Literal

from qualang_tools.bakery import baking
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


def baked_waveform(config, waveform_amp: float, qubit, max_length: int = 16):
    """Create baked pulse segments with 1ns granularity up to ``max_length`` ns.

    This mirrors the previous inline implementation inside ``12b_cryoscope.py`` and is
    extracted here so it can be shared / unit tested. Each index ``i`` (1..max_length)
    produces a baking object that plays a constant waveform of ``i`` ns with amplitude
    ``waveform_amp`` on the qubit flux line.

    Parameters
    ----------
    config : dict
        Configuration dictionary (typically produced by ``machine.generate_config()``)
        that the baking context mutates.
    waveform_amp : float
        The absolute amplitude to use for the flux pulse.
    qubit : Any
        QUAM qubit object containing the ``z`` element name.
    max_length : int, optional
        Maximum pulse length in ns to bake (default 16 to keep within baking memory limits).

    Returns
    -------
    list
        A list of baking objects; element ``i-1`` corresponds to a pulse of length ``i`` ns.
    """
    pulse_segments = []
    # Create the base waveform (1ns resolution). Represent as list of samples.
    waveform = [waveform_amp] * max_length
    for i in range(1, max_length + 1):  # inclusive
        with baking(config, padding_method="right") as b:
            wf = waveform[:i]
            b.add_op(f"flux_pulse{i}", qubit.z.name, wf)
            b.play(f"flux_pulse{i}", qubit.z.name)
        pulse_segments.append(b)
    return pulse_segments


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 5000
    """Number of averages to perform. Default is 50."""
    detuning_target_in_MHz: int = 300
    """Target detuning from sweetspot for the cryoscope pulse in MHz. Default is 350."""
    cryoscope_len: int = 240
    """Length of the cryoscope operation in microseconds. Default is 240."""
    num_frames: int = 17
    """Number of frames to use in the cryoscope experiment. Default is 17."""
    exponential_fit_time_fractions: List[float] = [0.5, 0.01]
    """List of time fractions for the exponential fit. Default is [0.5, 0.01]."""
    update_state_from_GUI: bool = False
    """Whether to update the state from the GUI. Default is False."""
    update_state: bool = False
    """Whether to update the state. Default is False."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass


class ShortTimeCryoscopeNodeSpecificParameters(RunnableParameters):
    """Extra parameters for the short-time cryoscope (18a) node.

    These augment :class:`CommonNodeParameters` and
    :class:`QubitsExperimentNodeParameters` with settings that differ from (or
    are not present in) the standard cryoscope :class:`NodeSpecificParameters`.
    """

    num_shots: int = 2500
    """Number of averages."""
    detuning_target_in_MHz: float = 800.0
    """Frequency detuning used for the cryoscope flux pulse (MHz)."""
    cryoscope_len: int = 64
    """Cryoscope pulse duration (ns). Must be a multiple of 16."""
    only_baked_waveforms: bool = True
    """If True, use only baked waveforms (recommended for cryoscope_len ≤ ~120 ns)."""
    num_frames: int = 17
    """Number of phase frames used to reconstruct the qubit phase."""
    flux_point: Literal["joint", "independent"] = "joint"
    """Whether to bias all qubits jointly or independently."""
    iir_or_fir: Literal["iir", "fir"] = "fir"
    """Use an IIR (exponential) or FIR-based correction filter."""
    num_forward_firs_values: List[int] = [16, 20, 24, 28, 32, 40, 48]
    """FIR lengths to search over during the forward-FIR grid search."""
    lam1_values: List[float] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    """Identity regularisation values for the FIR grid search."""
    lam2_values: List[float] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    """Exponential-tail regularisation values for the FIR grid search."""
    num_inverse_firs: int = 48
    """Length M of the inverse FIR filter. 0 means match best forward FIR length."""
    method: Literal["optimization", "analytical"] = "optimization"
    """Inversion method passed to :func:`invert_fir`."""
    sigma_ns: float = 0.5
    """Target Gaussian sigma (ns) for the optimisation-based inversion."""
    lam_smooth: float = 5e-3
    """Smoothing regularisation for the inversion."""
    update_state: bool = True
    """Whether to write the fitted filter back to the QuAM state."""


class ShortTimeCryoscopeParameters(
    NodeParameters,
    CommonNodeParameters,
    ShortTimeCryoscopeNodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
