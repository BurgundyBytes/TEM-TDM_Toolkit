import numpy as np
import math
import pandas as pd
from scipy.interpolate import CubicSpline
import os
import sys
from typing import Optional, Dict, Any, List, Tuple
import logging
import time

logger = logging.getLogger(__name__)

# custom modules
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, src_path)
from src.utilities import utils, plotting, metrics 


# Consistent type definitions
SignalDict = Dict[str, Any]
ConfigDict = Dict[str, Any]
BoolsDict = Dict[str, bool]
MetricsDict = Dict[str, Any] 
OptimalParamsDict = Dict[str, Any] 
OptimalSimResultDict = Dict[str, Any] 

# --- Constants ---
DEFAULT_NYQUIST_MIN_POINTS = 5 # Default minimum N if not in config
MIN_POINTS_FOR_CUBIC = 3     # Minimum points for cubic spline

# --- Helper Functions ---
def _determine_n_range(signal_data: SignalDict, n_optima: int) -> Optional[range]:
    """
    Determines the range of N points to test based on config and max value.
    Ensures n_min is at least MIN_POINTS_FOR_CUBIC.
    """
    freq_max = signal_data.get('freq_max')
    if freq_max is None or freq_max <= 0:
        logger.error("Invalid 'freq_max'.")
        return None
    if n_optima < MIN_POINTS_FOR_CUBIC:
        logger.error(f"n_optima ({n_optima}) < minimum required ({MIN_POINTS_FOR_CUBIC}).")
        return None

    # Calculate potential start point based on Nyquist, ensuring it's an integer
    n_nyquist_float = 2.0 * freq_max
    n_nyquist_int = int(math.ceil(n_nyquist_float)) # Use ceiling for safety

    # Determine actual start N: at least MIN_POINTS_FOR_CUBIC and at least n_nyquist_int
    start_n = max(MIN_POINTS_FOR_CUBIC, n_nyquist_int)

    # Determine end N (exclusive for range)
    end_n_exclusive = n_optima + 1

    # Check if the calculated start is already beyond the end
    if start_n >= end_n_exclusive:
        logger.warning(f"Calculated start N ({start_n}) >= end N ({n_optima}). "
                       f"Adjusting range to only test N={n_optima}.")
        # Create a range that only includes n_optima
        # Ensure n_optima itself meets the MIN_POINTS_FOR_CUBIC check (done at start)
        n_points_range = range(n_optima, n_optima + 1)
    else:
        # Normal case: create range from start_n up to n_optima
        n_points_range = range(start_n, end_n_exclusive)

    if len(n_points_range) == 0:
        # This case should theoretically be covered by the start_n >= end_n_exclusive check now
        logger.error("Resulting N range is unexpectedly empty.")
        return None

    logger.info(f"_determine_n_range: Nyquist N range set to: {n_points_range.start} to {n_points_range.stop - 1}")
    return n_points_range

def _sample_signal_by_index(t_orig: np.ndarray, u_orig: np.ndarray, n_points: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Samples the signal at n_points uniformly spaced *indices*.
    Returns sampled time and signal arrays, or None if invalid.
    """
    if n_points > len(t_orig):
        logger.warning(f"Requested n_points ({n_points}) > signal length ({len(t_orig)}). Using all points.")
        sampled_indices = np.arange(len(t_orig))
    elif n_points < MIN_POINTS_FOR_CUBIC:
        logger.error(f"Cannot sample with n_points ({n_points}) < {MIN_POINTS_FOR_CUBIC}.")
        return None
    else:
        # Ensure integer indices covering the full range 0 to len-1
        sampled_indices = np.linspace(0, len(t_orig) - 1, n_points, dtype=int)  # Evenly spaced indices
        sampled_time = t_orig[sampled_indices]
        sampled_signal = u_orig[sampled_indices]

    return sampled_time, sampled_signal


def _reconstruct_with_spline(t_sampled: np.ndarray, u_sampled: np.ndarray, t_target: np.ndarray, n_id_log: str) -> Optional[np.ndarray]:
    """
    Reconstructs signal on t_target using Cubic Spline interpolation.
    Handles potential errors (e.g., non-unique sample times).
    """
    try:
        # CubicSpline requires unique time points
        unique_times_indices = np.unique(t_sampled, return_index=True)[1]

        if len(unique_times_indices) < MIN_POINTS_FOR_CUBIC:
            logger.warning(f"\t{n_id_log}: Not enough unique time points ({len(unique_times_indices)}) for Cubic Spline. Skipping reconstruction.")
            return None

        t_unique = t_sampled[unique_times_indices]
        u_unique = u_sampled[unique_times_indices]

        cs = CubicSpline(t_unique, u_unique)
        u_reconstructed = cs(t_target) # Evaluate spline at original time points
        return u_reconstructed

    except ValueError as cve:
        logger.warning(f"\t{n_id_log}: CubicSpline failed: {cve}. Skipping reconstruction.")
        return None
    except Exception as e:
        logger.error(f"\t{n_id_log}: Unexpected error during spline reconstruction: {e}", exc_info=True)
        return None


# --- Main Nyquist Study Function ---
def perform_nyquist_study(signal_data: SignalDict, optimal_params: OptimalParamsDict, optimal_sim_results: Optional[OptimalSimResultDict], nyquist_path: str, bools: BoolsDict, config: Optional[ConfigDict] = None) -> Optional[pd.DataFrame]:
    '''
    Performs Nyquist analysis using index sampling and cubic spline reconstruction.

    Inputs
    ------
    - signal_data: SignalDict
        Dictionary containing signal data including time and signal values.
    - optimal_params: OptimalParamsDict
        Dictionary containing optimal parameters for the signal.
    - optimal_sim_results: Optional[OptimalSimResultDict]
        Dictionary containing simulation results from the optimal simulation.
    - nyquist_path: str
        Path to save Nyquist analysis results.
    - bools: BoolsDict
        Dictionary of boolean flags for workflow control.
    - config: Optional[ConfigDict]
        Configuration dictionary. Default is None.

    Outputs
    -------
    - nyquist_sim_result: Optional[pd.DataFrame]
        DataFrame containing Nyquist analysis results or None if errors occurred.
    '''
    signal_name = signal_data.get('name', 'unnamed_signal')
    run_id = f"Nyquist_{signal_name}"
    logger.info(f"--- Running {run_id} ---")

    # --- Validate Inputs ---
    t_orig = signal_data.get('t')
    u_orig = signal_data.get('u')
    N_spikes = optimal_sim_results.get(utils.COL_N_SPIKES, 0)

    if t_orig is None or u_orig is None or not isinstance(t_orig, np.ndarray) or not isinstance(u_orig, np.ndarray) or len(t_orig) < MIN_POINTS_FOR_CUBIC or len(t_orig) != len(u_orig):
        logger.error(f"{run_id}: Invalid original signal data ('t', 'u'). Needs >= {MIN_POINTS_FOR_CUBIC} points. Aborting.")
        return None
    if N_spikes < MIN_POINTS_FOR_CUBIC:
        logger.error(f"{run_id}: N_spikes ({N_spikes}) must be at least {MIN_POINTS_FOR_CUBIC}. Aborting.")
        return None
    if not nyquist_path or not isinstance(nyquist_path, str):
        logger.error(f"{run_id}: Invalid output_dir provided. Aborting.")
        return None
    
    # --- Determine Range and Stable Region ---
    n_points_range = _determine_n_range(signal_data, N_spikes)
    if n_points_range is None:
        logger.error(f"{run_id}: Could not determine a valid range for N points. Aborting.")
        return None
    logger.info(f"\tTesting N from {n_points_range.start} to {n_points_range.stop - 1} ({len(n_points_range)} values).")


    # --- Main Loop ---
    results_list: List[MetricsDict] = []
    plot_flag = bools.get('plots', False)

    for n_points in n_points_range:
        start_time_iter = time.time()

        # 1. Sample Signal
        sample_result = _sample_signal_by_index(t_orig, u_orig, n_points)
        if sample_result is None:
            continue # Error logged in helper
        t_sampled, u_sampled = sample_result

        # 2. Reconstruct Signal
        u_reconstructed = _reconstruct_with_spline(t_sampled, u_sampled, t_orig, n_points)
        if u_reconstructed is None:
            continue # Error logged in helper
        
        # Get optimal params
        if n_points == N_spikes:
            u_rec_trad = u_reconstructed 
            t_trad = t_orig 

        # 3. Calculate Metrics
        metrics_dict = metrics.calculate_nyquist_metrics(u_orig, u_reconstructed, n_points, start_time_iter)
        if metrics_dict is None:
            logger.warning(f"\tN={n_points}: Metric calculation failed. Skipping result for this N.")
            continue
        
        results_list.append(metrics_dict)

        # 4. Plotting for individual N (Optional)
        if plot_flag:
            # Plot 1: Reconstruction
            recon_plot_filename = os.path.join(nyquist_path, f"nyquist_recon_{signal_name}_N{n_points}.png")
            recon_title = f"Nyquist Sampling (N={n_points}) vs Original ({signal_name})"
            try:
                plotting.plot_nyquist_reconstruction(t_orig, u_orig, t_sampled, u_sampled, u_reconstructed, recon_title, recon_plot_filename)
            except AttributeError: logger.warning("plot_nyquist_reconstruction_simple not found in plotting module.")
            except Exception as e: logger.error(f"Error plotting reconstruction for N={n_points}: {e}", exc_info=False)

            # Plot 2: Error Profile
            error_plot_filename = os.path.join(nyquist_path, f"nyquist_error_{signal_name}_N{n_points}.png")
            error_title = f"Reconstruction Error (N={n_points}) ({signal_name})"
            try:
                plotting.plot_nyquist_error_profile(t_orig, u_orig-u_reconstructed, error_title, error_plot_filename)
            except AttributeError: logger.warning("plot_nyquist_error_simple not found in plotting module.")
            except Exception as e: logger.error(f"Error plotting error profile for N={n_points}: {e}", exc_info=False)


    # --- Finalize and Save ---
    if not results_list:
        logger.warning(f"{run_id}: Analysis completed, but no results generated.")
        return None

    df_nyquist_summary = pd.DataFrame(results_list).sort_values(by=utils.COL_N_POINTS).reset_index(drop=True)
    logger.info(f"{run_id}: Processing complete. Generated {len(df_nyquist_summary)} results.")

    base_filename = f"nyquist_summary_{signal_name}"
    try:
        utils.store_df_to_excel(df_nyquist_summary, nyquist_path, base_filename)
        if bools.get('pickle'):
            utils.save_pickle(df_nyquist_summary, nyquist_path, base_filename)
        logger.info(f"\tSaved Nyquist summary files ({base_filename}.xlsx/pkl).")

        # Plot overall summary
        if plot_flag and not df_nyquist_summary.empty:
            summary_plot_path = os.path.join(nyquist_path, f"nyquist_error_vs_N_{signal_name}.png")
            try:
                nyquist_summary_title = f"Nyquist Error vs. N Samples ({signal_name})"
                plotting.plot_nyquist_summary(df_nyquist_summary, [utils.COL_MAX_ERR, utils.COL_MED_ERR, utils.COL_RMSE, utils.COL_NRMSE_STD], nyquist_summary_title, summary_plot_path, log_y=True)
            except AttributeError: logger.warning("plot_nyquist_summary_simple not found in plotting module.")
            except Exception as e: logger.error(f"\tError plotting Nyquist summary: {e}", exc_info=True)

    except Exception as save_err:
        logger.error(f"\tError saving Nyquist summary data: {save_err}", exc_info=True)

    nyquist_sim_result = {
        'summary_df': df_nyquist_summary,
        'n_points_range': n_points_range,
        'n_nyquist_optimal': N_spikes,
        't_trad': t_trad,
        'u_rec_trad': u_rec_trad
    }

    return nyquist_sim_result 