import os
import sys
import numpy as np
import time
from scipy import stats
from sklearn.metrics import r2_score
from typing import Dict, Any, Optional
import logging

# custom modules
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, src_path)
from src.utilities import utils

logger = logging.getLogger(__name__)

# --- Private Helper for Core Error Stats ---
def _calculate_error_stats(errors: np.ndarray) -> Dict[str, float]:
    """
    Calculates common statistics from an array of errors.

    Inputs
    ------
    - errors: ndarray of floats
        Array of error values (e.g., differences between original and reconstructed signals).
    
    Outputs
    -------
    - stats_dict: dict
        Dictionary containing calculated statistics:
        - max_err: Maximum absolute error.
        - med_err: Median error.    
        - mean_err: Mean error.
        - mse: Mean squared error.
        - rmse: Root mean squared error.
        - mode_err: Mode of the errors.
        - error_midpoint: Midpoint of the errors array.
        - nrmse_std: Normalized root mean squared error (normalized by signal std dev).

    Raises
    ------
    - ValueError: If the input array is empty.
    - Exception: For any other unexpected errors during calculation.
    """
    # Initialize with utils constants
    stats_dict = {
        utils.COL_MAX_ERR: np.nan,
        utils.COL_MED_ERR: np.nan,
        utils.COL_MEAN_ERR: np.nan,
        utils.COL_MSE: np.nan,
        utils.COL_RMSE: np.nan,
        utils.COL_MODE_ERR: np.nan,
        utils.COL_ERROR_MIDPOINT: np.nan,
    }
    if errors.size == 0:
        logger.debug("_calculate_error_stats: Errors array is empty.")
        return stats_dict

    try:
        # Use utils constants as keys
        stats_dict[utils.COL_MAX_ERR] = np.max(np.abs(errors))
        stats_dict[utils.COL_MED_ERR] = np.median(errors)
        stats_dict[utils.COL_MEAN_ERR] = np.mean(errors)
        stats_dict[utils.COL_MSE] = np.mean(errors**2)
        stats_dict[utils.COL_RMSE] = np.sqrt(stats_dict[utils.COL_MSE])

        midpoint_index = len(errors) // 2
        stats_dict[utils.COL_ERROR_MIDPOINT] = errors[midpoint_index]

        # Calculate mode safely
        mode_result = stats.mode(errors, keepdims=False)
        # Check count before accessing mode (mode_result.mode can be array if multi-modal)
        if mode_result.count > 0 and np.isscalar(mode_result.mode):
            stats_dict[utils.COL_MODE_ERR] = mode_result.mode
        elif mode_result.count > 0: # Handle potential multi-modal case (take first mode)
            stats_dict[utils.COL_MODE_ERR] = mode_result.mode.flat[0]


    except Exception as e:
        logger.warning(f"Could not calculate some error stats: {e}", exc_info=False)

    return stats_dict


# --- Public Metrics Functions ---
def calculate_asdm_metrics(u_stable: np.ndarray, s: np.ndarray, u_rec_stable: np.ndarray, start_time: float) -> Optional[Dict[str, Any]]:
    '''
    Calculate metrics comparing stable regions of original and ASDM reconstructed signals.

    Inputs
    ------
    - u_stable: ndarray of floats
        Stable region of the original signal.
    - s: ndarray of floats
        Encoded signal (spike intervals).
    - u_rec_stable: ndarray of floats
        Stable region of the reconstructed signal.
    - start_time: float
        Start time for the metric calculation (used for elapsed time calculation).
    
    Outputs
    -------
    - metrics_dict: dict
        Dictionary containing calculated metrics:
        - time: Elapsed time for calculation.
        - in_len: Length of the input stable region.
        - n_spikes: Number of spikes in the encoded signal.
        - max_err: Maximum absolute error.
        - med_err: Median error.
        - mean_err: Mean error.
        - mse: Mean squared error.
        - rmse: Root mean squared error.
        - mode_err: Mode of the errors.
        - error_midpoint: Midpoint of the errors array.
        - nrmse_std: Normalized root mean squared error (normalized by signal std dev).
        - r2: R-squared value of the reconstruction.

    Raises
    ------
    - ValueError: If the input arrays are empty or have mismatched shapes.
    - Exception: For any other unexpected errors during calculation.
    '''
    elapsed_time = time.time() - start_time
    logger.debug("Calculating ASDM metrics (stable region)...")

    # Input validation
    if u_stable is None or u_rec_stable is None or s is None:
        logger.warning("calculate_asdm_metrics: Missing input arrays.")
        return None
    if u_stable.size == 0 or u_rec_stable.size == 0:
        logger.warning("calculate_asdm_metrics: Empty stable signal region(s) provided.")
        return None
    if u_stable.shape != u_rec_stable.shape:
        logger.warning(f"calculate_asdm_metrics: Shape mismatch u_stable {u_stable.shape} vs u_rec_stable {u_rec_stable.shape}. Skipping.")
        return None

    try:
        input_length_stable = len(u_stable)
        encoded_length = len(s) 

        errors = u_stable - u_rec_stable
        error_stats = _calculate_error_stats(errors) # Calculates basic error stats

        # --- Calculate NRMSE (normalized by signal std dev) and R2 ---
        nrmse_std = np.nan
        r2 = np.nan
        try:
            std_u = np.std(u_stable)
            rmse = error_stats.get(utils.COL_RMSE, np.nan)

            if std_u > 1e-12:
                if not np.isnan(rmse):
                    nrmse_std = rmse/std_u
            elif error_stats.get(utils.COL_MSE, np.nan) < 1e-12: 
                nrmse_std = 0.0 # Perfect reconstruction of constant
            else: # Non-constant error on zero-std signal? -> Infinite NRMSE
                nrmse_std = np.inf

            r2 = r2_score(u_stable, u_rec_stable)

        except Exception as e:
            logger.warning(f"Could not calculate NRMSE/R2: {e}", exc_info=False)


        # --- Combine Results ---
        metrics_dict = {
            utils.COL_TIME: elapsed_time,
            utils.COL_IN_LEN: input_length_stable, 
            utils.COL_N_SPIKES: encoded_length,
            **error_stats, # Unpack results 
            utils.COL_NRMSE_STD: nrmse_std, 
            utils.COL_R2: r2               
        }
        logger.debug(f"\tASDM Metrics: {utils.COL_N_SPIKES}={encoded_length}, {utils.COL_MED_ERR}={metrics_dict.get(utils.COL_MED_ERR, np.nan):.3e}, {utils.COL_MAX_ERR}={metrics_dict.get(utils.COL_MAX_ERR, np.nan):.3e}, Time={elapsed_time:.4f}s")
        return metrics_dict

    except Exception as e:
        logger.error(f"Unexpected error during ASDM metric calculation: {e}", exc_info=True)
        return None


def calculate_nyquist_metrics(u_orig: np.ndarray, u_reconstructed: np.ndarray, n_points: int, start_time: float) -> Optional[Dict[str, Any]]:
    '''
    Calculate metrics comparing the full original signal and a reconstructed signal.

    Inputs
    ------
    - u_orig: ndarray of floats
        Original signal.
    - u_reconstructed: ndarray of floats
        Reconstructed signal.
    - n_points: int
        Number of points in which the original signal was sampled.
    - start_time: float
        Start time for the metric calculation (used for elapsed time calculation).
    
    Outputs
    - metrics_dict: dict
        Dictionary containing calculated metrics:
        - time: Elapsed time for calculation.
        - in_len: Length of the input signal.
        - n_points: Number of points in which the original signal was sampled.
        - max_err: Maximum absolute error.
        - med_err: Median error.
        - mean_err: Mean error.
        - mse: Mean squared error.
        - rmse: Root mean squared error.
        - mode_err: Mode of the errors.
        - error_midpoint: Midpoint of the errors array.
        - nrmse_std: Normalized root mean squared error (normalized by signal std dev).
        - r2: R-squared value of the reconstruction.    

    Raises
    ------
    - ValueError: If the input arrays are empty or have mismatched shapes.
    - Exception: For any other unexpected errors during calculation.
    '''
    elapsed_time = time.time() - start_time

    # Input validation... (keep as is)
    if u_orig is None or u_reconstructed is None:
        logger.warning("calculate_nyquist_metrics: Missing input arrays.")
        return None
    if u_orig.size == 0 or u_reconstructed.size == 0:
        logger.warning(f"calculate_nyquist_metrics: Empty signal arrays provided (N={n_points}).")
        return None
    if u_orig.shape != u_reconstructed.shape:
        logger.warning(f"calculate_nyquist_metrics: Shape mismatch u_orig {u_orig.shape} vs u_reconstructed {u_reconstructed.shape} (N={n_points}). Skipping.")
        return None

    try:
        input_length_full = len(u_orig)

        errors = u_orig - u_reconstructed
        error_stats = _calculate_error_stats(errors) # Calculates basic error stats

        # --- Calculate NRMSE (normalized by signal std dev) and R2 ---
        nrmse_std = np.nan
        r2 = np.nan
        try:
            std_u = np.std(u_orig)
            rmse = error_stats.get(utils.COL_RMSE, np.nan) 

            if std_u > 1e-12:
                if not np.isnan(rmse):
                    nrmse_std = rmse / std_u
            elif error_stats.get(utils.COL_MSE, np.nan) < 1e-12: 
                nrmse_std = 0.0 # Perfect reconstruction of constant
            else: # Non-constant error on zero-std signal? -> Infinite NRMSE
                nrmse_std = np.inf

            r2 = r2_score(u_orig, u_reconstructed)

        except Exception as e:
            logger.warning(f"Could not calculate NRMSE/R2 (N={n_points}): {e}", exc_info=False)

        # --- Combine Results ---
        metrics_dict = {
            utils.COL_TIME: elapsed_time,
            utils.COL_IN_LEN: input_length_full,
            utils.COL_N_POINTS: n_points, 
            **error_stats, # Unpack results 
            utils.COL_NRMSE_STD: nrmse_std, 
            utils.COL_R2: r2                 
        }
        # Use constants in log message formatting string
        logger.debug(f"\tN={n_points}: {utils.COL_MAX_ERR}={metrics_dict.get(utils.COL_MAX_ERR, np.nan):.3e}, {utils.COL_MED_ERR}={metrics_dict.get(utils.COL_MED_ERR, np.nan):.3e}, {utils.COL_RMSE}={metrics_dict.get(utils.COL_RMSE, np.nan):.3e}")
        return metrics_dict

    except Exception as e:
        logger.error(f"Unexpected error during Nyquist metric calculation (N={n_points}): {e}", exc_info=True)
        return None