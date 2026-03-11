from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.analysis import fit_oscillation, unwrap_phase
from qualibration_libs.data import convert_IQ_to_V
from scipy import linalg
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize
from scipy.signal import deconvolve, lfilter, savgol_filter


def savgol(da, dim, range=3, order=2):
    def diff_func(x):
        return savgol_filter(x, range, order, deriv=0, delta=1)

    return xr.apply_ufunc(diff_func, da, input_core_dims=[[dim]], output_core_dims=[[dim]])


def diff_savgol(da, dim, range=3, order=2):
    def diff_func(x):
        return savgol_filter(x / (2 * np.pi), range, order, deriv=1, delta=1)

    return xr.apply_ufunc(diff_func, da, input_core_dims=[[dim]], output_core_dims=[[dim]])


def cryoscope_frequency(ds, stable_time_indices, quad_term=-1, sg_range=3, sg_order=2):
    ds = ds.copy()

    freq_cryoscope = diff_savgol(ds, "time", range=sg_range, order=sg_order)

    ds["freq"] = freq_cryoscope

    flux_cryoscope = np.sqrt(np.abs(1e9 * freq_cryoscope / quad_term)).fillna(0)

    if quad_term == -1:
        flux_cryoscope = flux_cryoscope / flux_cryoscope.sel(
            time=slice(stable_time_indices[0], stable_time_indices[1])
        ).mean(dim="time")

    ds["flux"] = flux_cryoscope

    return ds


def expdecay(x, s, a, t):
    """Exponential decay defined as 1 + a * np.exp(-x / t).
    :param x: numpy array for the time vector in ns
    :param a: float for the exponential amplitude
    :param t0: time shift
    :param t: float for the exponential decay time in ns
    :return: numpy array for the exponential decay
    """
    return s * (1 + a * np.exp(-(x) / t))


def two_expdecay(x, s, a, t, a2, t2):
    """Double exponential decay defined as s * (1 + a * np.exp(-x / t) + a2 * np.exp(-x / t2)).
    :param x: numpy array for the time vector in ns
    :param s: float for the scaling factor
    :param a: float for the first exponential amplitude
    :param t: float for the first exponential decay time in ns
    :param a2: float for the second exponential amplitude
    :param t2: float for the second exponential decay time in ns
    :return: numpy array for the double exponential decay
    """
    return s * (1 + a * np.exp(-(x) / t) + a2 * np.exp(-(x) / t2))


def single_exp(da, plot=True):
    first_vals = da.sel(time=slice(0, 1)).mean().values
    final_vals = da.sel(time=slice(20, None)).mean().values
    print(first_vals, final_vals)

    fit = da.curvefit(
        "time",
        expdecay,
        p0={"a": 1 - first_vals / final_vals, "t": 50, "s": final_vals},
    ).curvefit_coefficients

    fit_vals = {k: v for k, v in zip(fit.to_dict()["coords"]["param"]["data"], fit.to_dict()["data"])}

    t_s = 1
    alpha = np.exp(-t_s / fit_vals["t"])
    A = fit_vals["a"]
    fir = [1 / (1 + A), -alpha / (1 + A)]
    iir = [(A + alpha) / (1 + A)]

    if plot:
        fig, ax = plt.subplots()
        ax.plot(da.time, da, label="data")
        ax.plot(da.time, expdecay(da.time, **fit_vals), label="fit")
        ax.grid("all")
        ax.legend()
        print(f"Qubit - FIR: {fir}\nIIR: {iir}")
    else:
        fig = None
        ax = None
    return fir, iir, fig, ax, (da.time, expdecay(da.time, **fit_vals))


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode):
    """
    Fit raw cryoscope data with exponential models.

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset containing I/Q or state data
    node : QualibrationNode
        Node containing parameters and configuration

    Returns
    -------
    tuple
        (fitted_dataset, fit_results_dict)
    """

    if hasattr(ds, "I"):
        data = "I"
    elif hasattr(ds, "state"):
        data = "state"
    else:
        raise ValueError("Dataset must contain either 'I' or 'state' data")

    dafit = fit_oscillation(ds[data], "frame")

    daphi = unwrap_phase(dafit.sel(fit_vals="phi"), "time")
    sg_order = 2
    sg_range = 3

    qubit_name = node.parameters.qubits[0]
    qubit = node.machine.qubits[qubit_name]

    ds_fit = cryoscope_frequency(
        daphi,
        quad_term=qubit.freq_vs_flux_01_quad_term,
        stable_time_indices=(node.parameters.cryoscope_len - 20, node.parameters.cryoscope_len),
        sg_order=sg_order,
        sg_range=sg_range,
    )

    qubit = node.namespace["qubits"][0].name

    # Find the index where ds_fit.flux is closest to 1/e
    qubit_flux = ds_fit.flux.sel(qubit=qubit)
    flux_vals = qubit_flux.values
    time_vals = ds_fit.time.values

    fitting_start_fractions = node.parameters.exponential_fit_time_fractions
    success, best_fractions, components, a_dc, best_rms = optimize_start_fractions(
        time_vals, flux_vals, fitting_start_fractions
    )

    ds_fit.attrs["fit_success"] = int(success)
    if components is not None:
        try:
            amps = [float(a) for a, _ in components]
            taus = [float(t) for _, t in components]
        except Exception:
            amps, taus = [], []
        ds_fit.attrs["fit_component_amps"] = np.array(amps)
        ds_fit.attrs["fit_component_taus_ns"] = np.array(taus)
    ds_fit.attrs["fit_a_dc"] = float(a_dc) if a_dc is not None else np.nan

    ds["fit_results"] = ds_fit

    fit, fit_results = _extract_relevant_fit_parameters(ds, node)

    return fit, fit_results


def _extract_relevant_fit_parameters(ds: xr.Dataset, node: QualibrationNode):
    """Extract relevant fit parameters from the dataset and add metadata."""
    # Assess whether the fit was successful or not

    # Check if ds has fit_results (normal case) or use ds directly (error case)
    if "fit_results" in ds:
        fit = ds["fit_results"]
    else:
        fit = ds

    fit_results = {}

    # Get qubit names from the node if qubit dimension doesn't exist
    if hasattr(fit, "qubit") and hasattr(fit.qubit, "values"):
        qubit_names = fit.qubit.values
    else:
        qubit_names = [q.name for q in node.namespace["qubits"]]

    for q in qubit_names:
        success = fit.attrs.get("fit_success", False)
        # Reconstruct components from stored 1D arrays if available
        if "fit_component_amps" in fit.attrs and "fit_component_taus_ns" in fit.attrs:
            amps = fit.attrs.get("fit_component_amps", [])
            taus = fit.attrs.get("fit_component_taus_ns", [])
            try:
                components = list(zip(list(amps), list(taus)))
            except Exception:
                components = []
        else:
            # Backward compatibility (older in-memory attribute, not NetCDF-safe but maybe present at runtime)
            components = fit.attrs.get("fit_components", [])
        a_dc = fit.attrs.get("fit_a_dc", None)

        fit_results[q] = FitParameters(success=success, components=components, a_dc=a_dc)
    return ds, fit_results


def log_fitted_results(fit_results: dict, log_callable=print):
    """Log the fitted results for each qubit.

    Parameters
    ----------
    fit_results : dict
        Dictionary containing fit results for each qubit.
    log_callable : callable, optional
        Function to use for logging (default is print).
    """
    for qubit_name, fit_result in fit_results.items():
        log_callable(f"=== {qubit_name} ===")
        if getattr(fit_result, "success", False):
            log_callable("Overall fit: SUCCESSFUL")
        else:
            log_callable("Overall fit: FAILED")

        # New logging for FitParametersNEW structure
        if hasattr(fit_result, "components") and fit_result.components is not None:
            components = fit_result.components
            a_dc = getattr(fit_result, "a_dc", None)
            if a_dc is not None:
                log_callable(f"  DC term (a_dc): {a_dc:.6g}")
            if isinstance(components, (list, tuple)) and len(components) > 0:
                log_callable("  Exponential components (amplitude, tau [ns]):")
                for idx, comp in enumerate(components, start=1):
                    try:
                        amp, tau = comp
                        if a_dc not in (None, 0):
                            log_callable(f"    #{idx}: amp = {amp:.6g} (rel {amp / a_dc:.3f}), tau = {tau:.3f} ns")
                        else:
                            log_callable(f"    #{idx}: amp = {amp:.6g}, tau = {tau:.3f} ns")
                    except Exception:
                        log_callable(f"    #{idx}: {comp}")
            else:
                log_callable("  No exponential components fitted.")
        else:
            # Backwards compatibility: old FitParameters style
            if hasattr(fit_result, "fit1_success") or hasattr(fit_result, "fit2_success"):
                if getattr(fit_result, "fit1_success", False):
                    A = getattr(fit_result, "fit1_A", None)
                    tau = getattr(fit_result, "fit1_tau", None)
                    if A is not None and tau is not None:
                        log_callable(f"  Single exp: A = {A:.6g}, tau = {tau:.3f} ns")
                if getattr(fit_result, "fit2_success", False):
                    A1 = getattr(fit_result, "fit2_A1", None)
                    tau1 = getattr(fit_result, "fit2_tau1", None)
                    A2 = getattr(fit_result, "fit2_A2", None)
                    tau2 = getattr(fit_result, "fit2_tau2", None)
                    if None not in (A1, tau1, A2, tau2):
                        log_callable(
                            f"  Double exp: A1 = {A1:.6g}, tau1 = {tau1:.3f} ns | A2 = {A2:.6g}, tau2 = {tau2:.3f} ns"
                        )
        log_callable("")


@dataclass
class FitParameters:
    # List of (amplitude, tau) tuples for each exponential component
    components: list
    # Constant (DC) term
    a_dc: float
    # Overall success flag
    success: bool = False


def single_exp_decay(t: np.ndarray, amp: float, tau: float) -> np.ndarray:
    """Single exponential decay without offset

    Args:
        t (array): Time points
        amp (float): Amplitude of the decay
        tau (float): Time constant of the decay

    Returns:
        array: Exponential decay values
    """
    return amp * np.exp(-t / tau)


def sequential_exp_fit(
    t: np.ndarray,
    y: np.ndarray,
    start_fractions: List[float],
    fixed_taus: List[float] = None,
    a_dc: float = None,
    verbose: bool = 1,
) -> Tuple[List[Tuple[float, float]], float, np.ndarray]:
    """
    Fit multiple exponentials sequentially by:
    1. First fit a constant term from the tail of the data
    2. Fit the longest time constant using the latter part of the data
    3. Subtract the fit
    4. Repeat for faster components

    Args:
        t (array): Time points in nanoseconds, representing the time resolution of the pulse.
        y (array): Amplitude values of the pulse in volts.
    start_fractions (list): Fractions (0-1) where each component fit starts (user defined ordering).
        fixed_taus (list, optional): Fixed tau values (in nanoseconds) for each exponential component.
                                   If provided, only amplitudes are fitted, taus are constrained.
                                   Must have same length as start_fractions.
        a_dc (float, optional): Fixed constant term. If provided, the constant term is not fitted.
    verbose (int): Verbosity (0: silent, 1: summary, 2: detailed step-by-step info).

    Returns:
        tuple: (components, a_dc, residual) where:
            - components: List of (amplitude, tau) pairs for each fitted component
            - a_dc: Fitted constant term or the fixed constant term
            - residual: Data minus fitted curve after subtracting all components.
    """

    components = []  # List to store (amplitude, tau) pairs
    t_offset = t - t[0]  # Make time start at 0

    # Find the flat region in the tail by looking at local variance
    window = max(5, len(y) // 20)  # Window size by dividing signal into 20 equal pieces or at least 5 points
    rolling_var = np.array([np.var(y[i : i + window]) for i in range(len(y) - window)])
    # Find where variance drops below threshold, indicating flat region
    var_threshold = np.mean(rolling_var) * 0.1  # 10% of mean variance
    if a_dc is None:
        try:
            flat_start = np.where(rolling_var < var_threshold)[0][-1]
            # Use the flat region to estimate constant term
            a_dc = np.mean(y[flat_start:])
        except IndexError:
            print("No flat region found, using last point of the signal as constant term")
            a_dc = y[-1]

    if verbose:
        print(f"\nFitted constant term: {a_dc:.3e}")

    y_residual = y.copy() - a_dc

    for i, start_frac in enumerate(start_fractions):
        # Calculate start index for this component
        start_idx = int(len(t) * start_frac)
        if verbose:
            print(f"\nFitting component {i + 1} using data from t = {t[start_idx]:.1f} ns (fraction: {start_frac:.3f})")

        # Fit current component
        try:
            # Prepare fitting parameters based on whether tau is fixed
            if fixed_taus is not None:
                # Use fixed tau - only fit amplitude using lambda
                tau_fixed = fixed_taus[i]
                p0 = [y_residual[start_idx]]  # Only amplitude initial guess
                if verbose:
                    print(f"Using fixed tau = {tau_fixed:.3f} ns")

                # Perform the fit on the current interval
                t_fit = t_offset[start_idx:]
                y_fit = y_residual[start_idx:]
                popt, _ = curve_fit(lambda t, amp: single_exp_decay(t, amp, tau_fixed), t_fit, y_fit, p0=p0)

                # Store the components
                amp = popt[0]
                tau = tau_fixed
            else:
                # Fit both amplitude and tau (original behavior)
                p0 = [y_residual[start_idx], t_offset[start_idx] / 3]  # amplitude  # tau

                # Set bounds for the fit
                bounds = (
                    [-np.inf, 0.1],
                    # lower bounds: amplitude can be negative, tau must be positive (0.1 ns is arbitrary)
                    [np.inf, np.inf],  # upper bounds
                )

                # Perform the fit on the current interval
                t_fit = t_offset[start_idx:]
                y_fit = y_residual[start_idx:]
                popt, _ = curve_fit(single_exp_decay, t_fit, y_fit, p0=p0, bounds=bounds)

                # Store the components
                amp, tau = popt

            components.append((amp, tau))
            if verbose:
                tau_status = "(fixed)" if fixed_taus is not None else ""
                print(f"Found component: amplitude = {amp:.3e}, tau = {tau:.3f} ns {tau_status}")

            # Subtract this component from the entire signal
            y_residual -= amp * np.exp(-t_offset / tau)

        except (RuntimeError, ValueError) as e:
            if verbose:
                print(f"Warning: Fitting failed for component {i + 1}: {e}")
            break

    return components, a_dc, y_residual


def optimize_start_fractions(t, y, start_fractions, bounds_scale=0.5, fixed_taus=None, a_dc=None, verbose=1):
    """
        Optimize the start fractions for a sum of exponentials fit to data by minimizing the RMS error
    between the data and the fitted sum using `scipy.optimize.minimize`.
    This function attempts to find the optimal set of start fractions (time points at which each exponential
    component begins) that best fit the provided data with a sum of exponentials. Optionally, the time constants
    (taus) for each component can be fixed. The optimization is performed by minimizing the root mean square (RMS)
    of the residuals between the data and the fitted model.
    Parameters
    ----------
    t : np.ndarray
        1D array of time points in nanoseconds, representing the time resolution of the pulse.
    y : np.ndarray
        1D array of amplitude values of the pulse in volts, corresponding to each time point in `t`.
    start_fractions : list of float
        Initial guess for the start fractions (time points) for each exponential component. Must be in descending order.
    bounds_scale : float, optional
        Scale factor for bounds around start fractions during optimization (default is 0.5, meaning ±50%).
    fixed_taus : list of float or None, optional
        If provided, a list of fixed tau values (in nanoseconds) for each exponential component. If set, only amplitudes
        are fitted and taus are constrained. Must have the same length as `start_fractions`.
    a_dc : float or None, optional
        Constant (DC) term. If not provided, the constant term is estimated from the tail of the data.
    verbose : int, optional
    Verbosity (0: silent, 1: summary, 2: detailed) (default 1).
    Returns
    -------
    success : bool
        True if the optimization converged successfully, False otherwise.
    best_fractions : list of float
        The optimized start fractions (time points) for each exponential component.
    best_components : list of tuple (float, float)
        List of (amplitude, tau) tuples for each fitted exponential component.
    best_dc : float
        The fitted or provided constant (DC) term.
    best_rms : float
        The root mean square (RMS) of the residuals for the best fit.
    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 100, 101)
    >>> y = 2.0 * np.exp(-t / 10) + 1.0 * np.exp(-t / 30) + 0.1 + 0.05 * np.random.randn(len(t))
    >>> start_fractions = [80, 40]
    >>> success, best_fractions, best_components, best_dc, best_rms = optimize_start_fractions(
    ...     t, y, start_fractions, bounds_scale=0.5, fixed_taus=None, a_dc=None, verbose=False
    ... )
    >>> print("Success:", success)
    >>> print("Best fractions:", best_fractions)
    >>> print("Best components (amp, tau):", best_components)
    >>> print("Best DC:", best_dc)
    >>> print("Best RMS:", best_rms)
    Notes
    -----
    - The function requires that `start_fractions` are in descending order.
    - If `fixed_taus` is provided, it must have the same length as `start_fractions` and all values must be positive.
    - The function uses `sequential_exp_fit` internally to perform the fitting for each set of start fractions.
    """
    # Validate fixed_taus parameter
    if fixed_taus is not None:
        if len(fixed_taus) != len(start_fractions):
            raise ValueError("fixed_taus must have the same length as start_fractions")
        if any(tau <= 0 for tau in fixed_taus):
            raise ValueError("All fixed_taus values must be positive")

    def objective(x):
        """
        Objective function to minimize: RMS between the data and the fitted sum of
        exponentials.
        """
        # Ensure fractions are ordered in descending order
        if not np.all(np.diff(x) < 0):
            return 1e6  # Return large value if constraint is violated

        components, _, residual = sequential_exp_fit(t, y, x, fixed_taus=fixed_taus, a_dc=a_dc, verbose=verbose)
        if len(components) == len(start_fractions):
            current_rms = np.sqrt(np.mean(residual**2))
        else:
            current_rms = 1e6  # Return large value if fitting fails

        return current_rms

    # Define bounds for optimization
    bounds = []
    for start in start_fractions:
        min_val = start * (1 - bounds_scale)
        max_val = start * (1 + bounds_scale)
        bounds.append((min_val, max_val))
    if verbose > 0:
        print("\nOptimizing start_fractions using scipy.optimize.minimize...")
        print(f"Initial values: {[f'{f:.5f}' for f in start_fractions]}")
        print(f"Bounds: ±{bounds_scale * 100}% around initial values")

    # Run optimization
    result = minimize(
        objective,
        x0=start_fractions,
        bounds=bounds,
        method="Nelder-Mead",  # This method works well for non-smooth functions
        options={"disp": True, "maxiter": 1000},
    )

    # Get final results
    if result.success:
        best_fractions = result.x
        components, a_dc, best_residual = sequential_exp_fit(
            t, y, best_fractions, fixed_taus=fixed_taus, a_dc=a_dc, verbose=False
        )
        best_rms = np.sqrt(np.mean(best_residual**2))
        if verbose > 0:
            print("\nOptimization successful!")
            print(f"Initial fractions: {[f'{f:.5f}' for f in start_fractions]}")
            print(f"Optimized fractions: {[f'{f:.5f}' for f in best_fractions]}")
            if fixed_taus is not None:
                print(f"Fixed taus: {[f'{tau:.3f} ns' for tau in fixed_taus]}")
            print(f"Final RMS: {best_rms:.3e}")
            print(f"Number of iterations: {result.nit}")
    else:
        if verbose > 0:
            print("\nOptimization failed. Using initial values.")
        best_fractions = start_fractions
        components, a_dc, best_residual = sequential_exp_fit(
            t, y, best_fractions, fixed_taus=fixed_taus, a_dc=a_dc, verbose=False
        )
        best_rms = np.sqrt(np.mean(best_residual**2))

    components = [(amp * np.exp(t[0] / tau), tau) for amp, tau in components]
    if verbose > 0:
        print("Optimized components [(a1, tau1), (a2, tau2)...]:")
        print(components)
    return result.success, best_fractions, components, a_dc, best_rms


# ===========================================================================
# Short-time cryoscope: FIR analysis utilities
# (ported from iqcc_calibration_tools/analysis/cryoscope_tools.py)
# ===========================================================================


def conv_causal(v: np.ndarray, h: np.ndarray, N: Optional[int] = None) -> np.ndarray:
    """Perform a causal (one-sided) convolution of signal v with filter h.

    Parameters
    ----------
    v : array-like
        Input sequence.
    h : array-like
        Filter coefficients.
    N : int or None
        Number of output points. If None, returns up to len(v) samples.

    Returns
    -------
    y : ndarray
        Result of causal convolution truncated to N (or len(v)) samples.
    """
    v = np.asarray(v, dtype=float)
    h = np.asarray(h, dtype=float)
    y = np.convolve(v, h, mode="full")
    return y[: (len(v) if N is None else N)]


def build_toeplitz_matrix(v: np.ndarray, L: int) -> np.ndarray:
    """Build a Toeplitz convolution matrix from input sequence v.

    Parameters
    ----------
    v : array-like
        Input sequence.
    L : int
        Number of filter taps.

    Returns
    -------
    V : ndarray, shape (len(v), L)
        Toeplitz matrix such that V @ h ≈ conv(v, h)[:len(v)].
    """
    v = np.asarray(v, float)
    return linalg.toeplitz(c=v, r=np.concatenate([[v[0]], np.zeros(L - 1)]))


def resample_to_target_rate(
    data: np.ndarray,
    original_Ts: float,
    target_Ts: float,
    kind: str = "cubic",
) -> np.ndarray:
    """Resample time-domain data to a new sampling interval.

    Parameters
    ----------
    data : array-like
        Original time-domain samples.
    original_Ts : float
        Original sampling interval (ns).
    target_Ts : float
        Target sampling interval (ns).
    kind : str
        Interpolation kind ('linear', 'cubic', etc.).

    Returns
    -------
    resampled : ndarray
    """
    data = np.asarray(data)
    N = len(data)
    t_original = np.arange(N) * original_Ts
    max_time = t_original[-1]
    num_samples = int(np.floor(max_time / target_Ts)) + 1
    t_target = np.arange(num_samples) * target_Ts
    t_target = t_target[t_target <= max_time]
    interp_fun = interp1d(t_original, data, kind=kind, fill_value="extrapolate", bounds_error=False)
    return interp_fun(t_target)


def fit_fir(
    phi: np.ndarray,
    v: np.ndarray,
    L: int,
    Ts: float = 0.5,
    lam1: float = 1e-2,
    lam2: float = 1e-2,
    tail_ns: Optional[float] = None,
) -> np.ndarray:
    """Fit FIR filter coefficients h such that phi ≈ Toeplitz(v) @ h.

    Tikhonov regularisation with an identity term (lam1) and an
    exponential-tail term (lam2) is applied for stability.

    Parameters
    ----------
    phi : array-like
        Measured response signal.
    v : array-like
        Input (ideal step) signal.
    L : int
        FIR filter length.
    Ts : float
        Sampling interval (ns).
    lam1, lam2 : float
        Regularisation parameters.
    tail_ns : float or None
        Exponential tail time constant. Defaults to (L*Ts)/3.

    Returns
    -------
    h : ndarray, shape (L,)
        Fitted FIR coefficients.
    """
    phi = np.asarray(phi, float)
    v = np.asarray(v, float)
    V = build_toeplitz_matrix(v, L)
    if tail_ns is None:
        tail_ns = (L * Ts) / 3.0
    idx = np.arange(L)
    x = np.exp(idx * Ts / tail_ns)
    A = V.T @ V + lam1 * np.eye(L) + lam2 * np.diag(x)
    b = V.T @ phi
    h = linalg.solve(A, b, assume_a="pos")
    return h


def optimize_fir_parameters(
    response: np.ndarray,
    Ts: float = 0.5,
    L_values: List[int] = None,
    lam1_values: List[float] = None,
    lam2_values: List[float] = None,
) -> Tuple[list, float, Optional[dict], Optional[np.ndarray], Optional[np.ndarray]]:
    """Grid-search over (L, lam1, lam2) to minimise FIR reconstruction error.

    Parameters
    ----------
    response : array-like
        Measured (distorted) response to model (normalised amplitude).
    Ts : float
        Sampling interval (ns).
    L_values, lam1_values, lam2_values : lists
        Search grids for filter length and regularisation parameters.

    Returns
    -------
    results : list of dicts
    best_error : float
    best_params : dict or None
    best_h : ndarray or None
    best_reconstructed : ndarray or None
    """
    if L_values is None:
        L_values = [16, 20, 24, 28, 32, 40, 48]
    if lam1_values is None:
        lam1_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    if lam2_values is None:
        lam2_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    ideal_response = np.ones(len(response))
    results = []
    best_error = float("inf")
    best_params = None
    best_h = None
    best_reconstructed = None

    for L in L_values:
        for lam1 in lam1_values:
            for lam2 in lam2_values:
                try:
                    h = fit_fir(response, ideal_response, L=L, Ts=Ts, lam1=lam1, lam2=lam2)
                    h /= np.sum(h)
                    V = build_toeplitz_matrix(ideal_response, L)
                    reconstructed = V @ h
                    reconstruction_error = np.linalg.norm(response - reconstructed) / np.linalg.norm(response)
                    result = {
                        "L": L,
                        "lam1": lam1,
                        "lam2": lam2,
                        "error": reconstruction_error,
                        "h": h.copy(),
                        "reconstructed": reconstructed.copy(),
                    }
                    results.append(result)
                    if reconstruction_error < best_error:
                        best_error = reconstruction_error
                        best_params = result
                        best_h = h.copy()
                        best_reconstructed = reconstructed.copy()
                except Exception as e:
                    print(f"Warning: FIR fit failed for L={L}, lam1={lam1:.0e}, lam2={lam2:.0e}: {e}")

    return results, best_error, best_params, best_h, best_reconstructed


def invert_fir(
    h: np.ndarray,
    Ts: float = 0.5,
    M: Optional[int] = None,
    method: Literal["optimization", "analytical"] = "optimization",
    sigma_ns: float = 0.75,
    lam_smooth: float = 5e-2,
    normalize_dc_gain: bool = False,
) -> np.ndarray:
    """Compute an approximate causal FIR inverse of h.

    Solves:  min_{h_inv} || d - conv(h, h_inv) ||^2 + lam_smooth * ||Δ h_inv||^2

    Parameters
    ----------
    h : array-like
        Forward FIR coefficients.
    Ts : float
        Sampling interval (ns).
    M : int or None
        Length of the inverse filter. Defaults to len(h).
    method : 'optimization' or 'analytical'
    sigma_ns : float
        Standard deviation of the target Gaussian approximating a causal delta.
    lam_smooth : float
        First-difference smoothing regularisation weight.
    normalize_dc_gain : bool
        Whether to normalise the composite DC gain of h * h_inv to 1.

    Returns
    -------
    h_inv : ndarray, shape (M,)
    """
    h = np.asarray(h, float)
    L = len(h)
    if M is None:
        M = L

    if method == "optimization":
        t = np.arange(M) * Ts
        d = np.exp(-0.5 * (t / sigma_ns) ** 2)
        d /= d.sum()
        h_padded = np.pad(h, (0, max(0, M - L)), mode="constant")
        H = build_toeplitz_matrix(h_padded, M)[:M, :]
        D = np.eye(M, k=0) - np.eye(M, k=1)
        D = D[:-1, :]
        A = H.T @ H + lam_smooth * (D.T @ D)
        b = H.T @ d
        h_inv = linalg.solve(A, b, assume_a="pos")
        if normalize_dc_gain:
            gain = h.sum() * h_inv.sum()
            if gain != 0:
                h_inv /= gain
    else:  # analytical
        h_inv = np.zeros(M)
        h_inv[0] = 1.0 / h[0]
        for m in range(1, min(L, M)):
            s = sum(h_inv[m - i] * h[i] for i in range(1, min(m + 1, L)))
            h_inv[m] = -s / h[0]

    return h_inv


def analyze_and_plot_inverse_fir(
    response: np.ndarray,
    time: np.ndarray,
    Ts: float = 0.5,
    L_values: Optional[List[int]] = None,
    lam1_values: Optional[List[float]] = None,
    lam2_values: Optional[List[float]] = None,
    M: Optional[int] = None,
    sigma_ns: float = 0.75,
    lam_smooth: float = 5e-2,
    method: Literal["optimization", "analytical"] = "optimization",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, "plt.Figure", "plt.Figure"]:
    """Full short-time cryoscope FIR pipeline: fit forward FIR then compute its inverse.

    Runs a grid search for the best forward FIR (h) that reconstructs *response*
    from the ideal step, then computes a causal inverse FIR (h_inv) such that
    pre-distorting with h_inv and then experiencing h returns a flat response.

    Parameters
    ----------
    response : array-like
        Normalised flux step response (length N).
    time : array-like
        Corresponding time axis (ns).
    Ts : float
        Sampling interval (ns, typically 0.5 for 2 GS/s data).
    L_values, lam1_values, lam2_values : list or None
        Grid-search ranges passed to :func:`optimize_fir_parameters`.
    M : int or None
        Length of the inverse FIR. If None, matches length of best h.
    sigma_ns, lam_smooth : float
        Parameters for :func:`invert_fir`.
    method : 'optimization' or 'analytical'
    verbose : bool

    Returns
    -------
    best_h : ndarray
        Best forward FIR coefficients.
    h_inv : ndarray
        Inverse FIR coefficients (to load onto the OPX).
    best_reconstructed : ndarray
        Reconstructed response using best_h.
    fig_fir_fit : Figure
        FIR fitting summary figure.
    fig_inv : Figure
        Inverse FIR and correction summary figure.
    """
    if L_values is None:
        L_values = [16, 20, 24, 28, 32, 40, 48]
    if lam1_values is None:
        lam1_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    if lam2_values is None:
        lam2_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    response = np.asarray(response, float)
    time = np.asarray(time, float)

    # --- Step 1: optimise forward FIR ---
    results, best_error, best_params, best_h, best_reconstructed = optimize_fir_parameters(
        response, Ts=Ts, L_values=L_values, lam1_values=lam1_values, lam2_values=lam2_values
    )

    if verbose and best_params is not None:
        print(f"Best FIR: L={best_params['L']}, lam1={best_params['lam1']:.0e}, "
              f"lam2={best_params['lam2']:.0e}, error={best_error:.4e}")

    # --- FIR fit figure ---
    fig_fir_fit, axes = plt.subplots(2, 2, figsize=(14, 8))
    ax = axes[0, 0]
    ax.plot(time, response, "r-", label="Measured", linewidth=2, alpha=0.7)
    if best_reconstructed is not None:
        ax.plot(time, best_reconstructed, "b--", label=f"Reconstructed (err={best_error:.3e})", linewidth=2)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Best FIR Reconstruction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if best_reconstructed is not None:
        residual = response - best_reconstructed
        ax.plot(time, residual, "m-", linewidth=1.5)
        ax.fill_between(time, -np.std(residual), np.std(residual), alpha=0.2, color="gray")
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Time (ns)")
    ax.set_title("Residual")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if best_h is not None:
        ax.plot(best_h, "b-o", markersize=4, linewidth=2)
    ax.set_xlabel("Tap index")
    ax.set_title(f"Forward FIR (L={best_params['L'] if best_params else '?'})")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    errors_by_L: dict = {}
    for r in results:
        errors_by_L.setdefault(r["L"], []).append(r["error"])
    L_sorted = sorted(errors_by_L.keys())
    ax.errorbar(
        L_sorted,
        [np.mean(errors_by_L[L]) for L in L_sorted],
        yerr=[np.std(errors_by_L[L]) for L in L_sorted],
        fmt="o-", capsize=5, linewidth=2,
    )
    if best_error < float("inf"):
        ax.axhline(best_error, color="r", linestyle="--", alpha=0.5, label=f"Best: {best_error:.3e}")
    ax.set_xlabel("Filter length L")
    ax.set_ylabel("Reconstruction error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig_fir_fit.tight_layout()

    # --- Step 2: compute inverse FIR ---
    h_inv = invert_fir(best_h, Ts=Ts, M=M, method=method, sigma_ns=sigma_ns, lam_smooth=lam_smooth)

    # Simulate correction: predistort ideal signal, then apply forward distortion
    ideal_response = np.ones(len(response))
    L_guard = len(h_inv)
    guard = np.zeros(L_guard)
    ideal_padded = np.concatenate([guard, ideal_response, guard])
    predistorted_padded = conv_causal(ideal_padded, h_inv)
    predistorted_response = predistorted_padded[L_guard : L_guard + len(ideal_response)]
    corrected_response = conv_causal(predistorted_response, best_h, N=len(ideal_response))

    correction_error = np.linalg.norm(corrected_response - ideal_response) / np.linalg.norm(ideal_response)
    if verbose:
        print(f"Correction error (NRMS): {correction_error:.3e}")

    delta = conv_causal(best_h, h_inv, N=len(best_h))

    # --- Inverse FIR figure ---
    fig_inv, axes2 = plt.subplots(2, 2, figsize=(14, 8))
    ax = axes2[0, 0]
    ax.plot(time, ideal_response, "g--", label="Ideal", linewidth=2, alpha=0.7)
    ax.plot(time, response, "r-", label="Distorted", linewidth=2)
    if best_reconstructed is not None:
        ax.plot(time, best_reconstructed, "b:", label="Predicted (FIR)", linewidth=2, alpha=0.7)
    ax.axhline(1.001, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(0.999, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylim([0.95, 1.05])
    ax.set_xlabel("Time (ns)")
    ax.set_title("Signals and FIR prediction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes2[0, 1]
    ax.plot(best_h, "b-o", label="Forward FIR (h)", markersize=4, linewidth=2)
    ax.plot(h_inv, "r-s", label="Inverse FIR (h_inv)", markersize=4, linewidth=2)
    ax.set_xlabel("Tap index")
    ax.set_title("FIR filters")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes2[1, 0]
    ax.plot(time, ideal_response, "g--", label="Ideal", linewidth=2, alpha=0.7)
    ax.plot(time, corrected_response, "m-", label="Corrected (sim)", linewidth=2)
    ax.axhline(1.001, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(0.999, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylim([0.95, 1.05])
    ax.set_xlabel("Time (ns)")
    ax.set_title("Predicted correction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes2[1, 1]
    t_delta = np.arange(len(delta)) * Ts
    ax.plot(t_delta, delta, "g-o", markersize=4, linewidth=2)
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Time (ns)")
    ax.set_title(f"h * h_inv (≈ δ, peak={np.max(np.abs(delta)):.3e})")
    ax.grid(True, alpha=0.3)

    fig_inv.tight_layout()

    return best_h, h_inv, best_reconstructed, fig_fir_fit, fig_inv


# ===========================================================================
# Short-time cryoscope: raw-data → phase → frequency → flux pipeline
# ===========================================================================


def extract_short_time_phase(ds: xr.Dataset) -> xr.Dataset:
    """Fit a sine wave to state vs. frame data at each time step to extract phase.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing a ``state`` variable with dimensions
        ``(qubit, time, frame)``.

    Returns
    -------
    xr.Dataset
        Input dataset with a new ``phase`` coordinate of shape (qubit, time).
    """

    def sine_fit(x, phase, A, offset):
        return A * np.sin(2 * np.pi * x + phase) + offset

    phases = []
    for q_idx in range(len(ds.qubit)):
        phase_q = []
        for t_idx in range(len(ds.time)):
            y_data = ds.state.isel(qubit=q_idx, time=t_idx).values
            x_data = ds.frame.values
            try:
                popt, _ = curve_fit(
                    sine_fit,
                    x_data,
                    y_data,
                    p0=[0, 0.5, 0.5],
                    bounds=([-np.pi, 0, -np.inf], [np.pi, np.inf, np.inf]),
                )
                phase_q.append(popt[0])
            except RuntimeError:
                phase_q.append(0.0)
        phases.append(np.unwrap(phase_q))

    return ds.assign_coords(phase=(["qubit", "time"], phases))


def extract_short_time_freqs(ds: xr.Dataset, detuning_MHz: float) -> xr.Dataset:
    """Compute qubit frequency shift at each time step via the gradient of the phase.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with a ``phase`` coordinate (rad) of shape (qubit, time).
    detuning_MHz : float
        Cryoscope detuning used for the experiment (MHz). This is added back
        to the numerical derivative to recover the instantaneous frequency.

    Returns
    -------
    xr.Dataset
        Input dataset with a new ``frequencies`` coordinate (MHz) of shape
        (qubit, time).
    """

    def calc_freq(phase_values, time_values):
        dphase_dt = np.gradient(phase_values, time_values, axis=-1)
        return dphase_dt / (2 * np.pi) + detuning_MHz / 1e3  # GHz

    freqs = xr.apply_ufunc(
        calc_freq,
        ds.phase,
        ds.time,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
    )
    return ds.assign_coords(frequencies=1e3 * freqs)  # store in MHz


def extract_short_time_flux(ds: xr.Dataset, qubits) -> xr.Dataset:
    """Convert frequency shift to flux amplitude using the qubit's quadratic term.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with a ``frequencies`` coordinate (MHz) of shape (qubit, time).
    qubits : list of QUAM qubit objects
        Qubits whose ``freq_vs_flux_01_quad_term`` is used for the conversion.

    Returns
    -------
    xr.Dataset
        Input dataset with a new ``flux`` coordinate (V) of shape (qubit, time).
    """
    fluxes = []
    for qubit in qubits:
        freq_MHz = ds.sel(qubit=qubit.name).frequencies.values
        fluxes.append(np.sqrt(np.abs(-1e6 * freq_MHz / qubit.freq_vs_flux_01_quad_term)))
    return ds.assign_coords(flux=(["qubit", "time"], fluxes))


def process_raw_dataset_short_time(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Extract phase, frequency, and flux from a short-time cryoscope raw dataset.

    Wraps :func:`extract_short_time_phase`, :func:`extract_short_time_freqs`,
    and :func:`extract_short_time_flux` into a single call.

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset with ``state`` variable of shape (qubit, time, frame).
    node : QualibrationNode
        Node providing ``parameters.detuning_target_in_MHz`` and the qubit list
        via ``node.namespace["qubits"]``.

    Returns
    -------
    xr.Dataset
        Enriched dataset with ``phase``, ``frequencies``, and ``flux`` coords.
    """
    ds = extract_short_time_phase(ds)
    ds = extract_short_time_freqs(ds, node.parameters.detuning_target_in_MHz)
    ds = extract_short_time_flux(ds, node.namespace["qubits"])
    return ds
