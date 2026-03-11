from typing import List, Literal, Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Parameters for 16a_long_cryoscope"""

    num_shots: int = 30
    """Number of shots to acquire."""
    operation: str = "x180"
    """Operation to excite the qubit"""
    operation_amplitude_factor: float = 1.0
    """Amplitude factor for the operation."""
    duration_in_ns: int = 100000
    """Maximum duration of the sequence."""
    time_axis: Literal["linear", "log"] = "log"
    """Time axis for the operation."""
    time_step_in_ns: int = 4
    """Time step in nanoseconds. For linear time axis."""
    time_step_num: int = 300
    """Number of time steps. Used for log time axis."""
    min_wait_time_in_ns: int = 16
    """Minimum wait time in nanoseconds."""
    frequency_span_in_mhz: float = 100
    """Frequency span in MHz for the qubit spectroscopy tone"""
    frequency_step_in_mhz: float = 1
    """Frequency step in MHz for the qubit spectroscopy tone"""
    detuning_in_mhz: int = 500
    """Objective qubit detuning in MHz"""
    fitting_base_fractions: List[float] = [0.4, 0.15, 0.05]
    """Base fractions for the fitting of the exponential sum."""
    update_state: bool = False
    """Whether to update the state. CAUTION: assumes fitting will be acceptable"""
    update_state_from_GUI: bool = False
    """Whether to update the state from the GUI, select when fitting is successful"""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
