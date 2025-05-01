import numpy as np
import pandas as pd
import os
import sys
from typing import Optional, Dict, List, Any, Tuple
import logging
import time

logger = logging.getLogger(__name__)

# custom modules
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, src_path)
from src.utilities import utils, plotting, metrics, asdm


# --- Data Types ---
SignalDict = Dict[str, Any]
ConfigDict = Dict[str, Any]
BoolsDict = Dict[str, bool]
SignalResultsDict = Dict[str, Optional[pd.DataFrame]]
ParametricResultsDict = Dict[str, SignalResultsDict]
OptimalParamsDict = Dict[str, Any]
OptimalSimResultDict = Dict[str, Any]
EncoderParamsDict = Dict[str, Optional[float]] 

# --- Constants ---
DEFAULT_STABILITY_FACTOR = 1.5 # Default b = 1.5 * c
DEFAULT_R_MARGIN = 0.9 # Default target r <= 0.9
KEY_PRIORITY = 'Encoder Selection Priority' # Config key for priority
KEY_STABILITY_FACTOR = 'Encoder Stability Factor' # Config key for b = factor*c
KEY_NOISE_DELTA_FACTOR = 'Encoder Noise Delta Factor' # Config key delta = factor*noise
KEY_TARGET_K = 'Encoder Target K' # Config key for fixed k
KEY_TARGET_DELTA = 'Encoder Target Delta' # Config key for fixed delta
KEY_MAX_ITERATIONS = 'Encoder Max Iterations' # Config key for retry loop


# --- Helper Functions ---

def _calculate_physical_params(d_norm_opt: float, b_selected: float, priority: str, config: ConfigDict, noise_rms_estimate: float = 1e-4) -> Tuple[Optional[float], Optional[float]]:
    '''
    Calculates k and delta based on d_norm_opt, b_selected, and priority.
    Priority options: 'noise', 'low_spike', 'fixed_k', 'fixed_delta'

    Inputs
    ------
    - d_norm_opt:

    '''
    target_k_delta = (d_norm_opt * b_selected) / 2.0
    k_calc, delta_calc = None, None

    if priority in ['noise', 'low_spike']:
        # Prioritize delta based on noise or preference for larger delta
        if priority == 'noise':
            noise_factor = config.get(KEY_NOISE_DELTA_FACTOR, 5.0)
            delta_calc = noise_factor * noise_rms_estimate 
            logger.info(f"\tPriority='{priority}': Selecting delta={delta_calc:.4e} based on noise_rms={noise_rms_estimate:.1e} * factor={noise_factor}")
        else: # low_spike -> try to use a target delta if specified, else fallback might be needed
            delta_calc = config.get(KEY_TARGET_DELTA) # Use target delta if aiming for low spikes
            if delta_calc is None:
                logger.warning(f"\tPriority='low_spike' but '{KEY_TARGET_DELTA}' not set. Add it to config or choose 'noise'. Falling back to noise-based delta.")
                noise_factor = config.get(KEY_NOISE_DELTA_FACTOR, 5.0)
                delta_calc = noise_factor * noise_rms_estimate # Fallback
            else:
                logger.info(f"\tPriority='{priority}': Using target delta={delta_calc:.4e} from config.")

        if delta_calc is None or delta_calc <= 1e-12: # Check for invalid delta
            logger.error(f"\tCannot calculate k: Calculated/Selected delta ({delta_calc}) is invalid.")
            return None, None
        k_calc = target_k_delta / delta_calc

    elif priority == 'fixed_k':
        k_calc = config.get(KEY_TARGET_K)
        if k_calc is None or k_calc <= 0:
            logger.error(f"\tPriority='fixed_k' but '{KEY_TARGET_K}' not set or invalid ({k_calc}) in config.")
            return None, None
        logger.info(f"\tPriority='fixed_k': Using target k={k_calc:.4e} from config.")
        delta_calc = target_k_delta / k_calc

    elif priority == 'fixed_delta':
        delta_calc = config.get(KEY_TARGET_DELTA)
        if delta_calc is None or delta_calc <= 0:
             logger.error(f"\tPriority='fixed_delta' but '{KEY_TARGET_DELTA}' not set or invalid ({delta_calc}) in config.")
             return None, None
        logger.info(f"\tPriority='fixed_delta': Using target delta={delta_calc:.4e} from config.")
        k_calc = target_k_delta / delta_calc

    else:
        logger.error(f"Invalid priority '{priority}' specified in config key '{KEY_PRIORITY}'.")
        return None, None

    # Final check on calculated values
    if k_calc is None or delta_calc is None or k_calc <= 0 or delta_calc <= 0:
        logger.error(f"\tFailed to calculate valid k ({k_calc}) or delta ({delta_calc}).")
        return None, None

    logger.info(f"\tCalculated: k={k_calc:.4e}, delta={delta_calc:.4e} (target k*delta = {target_k_delta:.4e})")
    return k_calc, delta_calc


def _verify_recovery_condition(k: float, delta: float, b: float, c: float, omega: float, r_margin: float = DEFAULT_R_MARGIN) -> bool:
    '''
    Checks if the perfect recovery condition r < r_margin is met.
    
    Inputs
    ------
    
    '''
    if b <= c: # Stability check first
        logger.error(f"\tVerification FAILED: Stability requires b > c (b={b:.4e}, c={c:.4e})")
        return False
    if omega is None or omega <= 0:
        logger.error(f"\tVerification FAILED: Invalid Omega (bandwidth) provided: {omega}")
        return False

    r_value = (2.0 * k * delta / (b - c)) * (omega / np.pi)
    logger.info(f"\tVerifying recovery condition: r = (2*k*delta/(b-c))*(Omega/pi) = {r_value:.4f}")

    if r_value < r_margin:
        logger.info(f"\tVerification PASSED: r ({r_value:.4f}) < margin ({r_margin:.4f})")
        return True
    else:
        logger.error(f"\tVerification FAILED: r ({r_value:.4f}) >= margin ({r_margin:.4f})")
        return False

def _simulate_with_physical_params(signal_data: SignalDict, encoder_params: EncoderParamsDict, encoder_output_path: Optional[str], bools: BoolsDict, config: ConfigDict) -> Optional[OptimalSimResultDict]:
    """
    Runs the simulation using the calculated k, b, delta.
    Reuses logic similar to _simulate_and_save_optimal_step from optima.py.
    """
    signal_name = signal_data.get('name', 'unnamed_signal')
    logger.info(f"--- Simulating with Physical Encoder Params [{signal_name}] ---")

    # --- Validate Inputs & Setup ---
    if not encoder_output_path:
        logger.warning(f"\tSkipping physical param simulation for {signal_name}: Output path not provided.")
        return None

    try:
        # Extract required data
        t = signal_data.get('t')
        u = signal_data.get('u')
        dur = signal_data.get('dur')

        # Use the *calculated* k, b, delta
        k = encoder_params.get('k')
        b = encoder_params.get('b')
        delta = encoder_params.get('delta')

        # Optimal d_norm for reference, not directly used in simulation physics here
        d_norm_opt = encoder_params.get('d_norm_optimal')
        freq_max = signal_data.get('freq_max')

        # Basic Validation
        required = {'t': t, 'u': u, 'dur': dur, 'k': k, 'b': b, 'delta': delta, 'freq_max': freq_max}
        missing = [key for key, val in required.items() if val is None]
        if missing:
            logger.error(f"\tSkipping simulation: Missing required data: {missing}")
            return None

        dte = signal_data.get('dte')
        fs =  signal_data.get('fs')

        logger.info(f"\tRunning simulation with k={k:.4e}, b={b:.4e}, delta={delta:.4e}...")
        start_time = time.time()

        # --- Run Simulation ---
        s = asdm.asdm_encode_physical(u, 1/fs, k, b, delta, dte) 
        bw = utils.estimate_decoder_bandwidth(freq_max)
        u_rec = asdm.asdm_decode_physical(s, dur, 1/fs, b, bw) 

        if s is None or len(s) < 2:
            logger.error(f"\tEncoding failed for {signal_name}.")
            return None
        N_spikes = len(s)

        if u_rec is None or len(u_rec) == 0:
            logger.warning(f"\tDecoding failed for {signal_name}. Skipping metrics.")
            return None

        # Align length (simple truncation/padding) 
        if len(u_rec) != len(t):
            logger.warning(f"\t(Encoder) Length mismatch: Original ({len(t)}) vs Reconstructed ({len(u_rec)}). Aligning...")
            u_rec = np.zeros_like(t) # Create array of target shape
            common_length = min(len(u_rec), len(t))
            u_rec[:common_length] = u_rec[:common_length] # Copy common part
        else:
            u_rec = u_rec # No alignment needed

        # --- Stable Region & Metrics ---
        metrics_dict = None
        simulation_results: Optional[OptimalSimResultDict] = None
        t_stable, u_stable, u_rec_stable = None, None, None
        start_idx, end_idx = None, None        
        try:
            stable_percentage = config.get('Stable Region Percentage', 0.9)
            start_idx, end_idx = utils.get_stable_region(t, stable_percentage)
            # Check if get_stable_region returned fallback (0, len(t)) due to error/short signal
            if end_idx == len(t) and start_idx == 0 and len(t)>1: # Avoid trigger on len=1 case
                logger.info(f"  Stable region covers full signal [{start_idx}:{end_idx}].")
            else:
                logger.info(f"  Using stable region indices [{start_idx}:{end_idx}] for metrics.")
            t_stable = t[start_idx:end_idx]
            u_stable = u[start_idx:end_idx]
            u_rec_stable = u_rec[start_idx:end_idx]

            # Pass original u_stable for comparison, spikes s, reconstructed u_rec_stable
            metrics_dict = metrics.calculate_asdm_metrics(u_stable, s, u_rec_stable, start_time)

            if metrics_dict:
                metrics_dict[utils.COL_STABLE_START_IDX] = start_idx
                metrics_dict[utils.COL_STABLE_END_IDX] = end_idx
                metrics_dict[utils.COL_N_SPIKES] = N_spikes 
                logger.info(f"\tPhysical Param Run Metrics (Stable): MedErr={metrics_dict.get(utils.COL_MED_ERR, 'N/A'):.4e}, NSpikes={N_spikes}")
            else:
                logger.warning(f"\tMetric calculation failed for {signal_name} (physical params).")
                metrics_dict = {'error_notes': "Metric calculation failed."}
        except Exception as metric_err:
            logger.error(f"\tError during stable region/metric calculation for {signal_name}: {metric_err}", exc_info=False)
            metrics_dict = {'error_notes': f"Metric calculation error: {metric_err}"}


        # --- Save Summary & Prepare Output ---
        summary_data = {
            'k': k, 'b': b, 'delta': delta, 'd_norm_target': d_norm_opt,
            'r_calculated': (2.0 * k * delta / (b - signal_data.get('c', b))) * (signal_data.get('freq_max', 0)*2*np.pi / np.pi) if signal_data.get('c') is not None else None, # Recalculate r for logging
            **(metrics_dict if metrics_dict else {})
        }
        df_summary = pd.DataFrame([summary_data])
        base_filename = f"encoder_params_summary_{signal_name}"
        utils.store_df_to_excel(df_summary, encoder_output_path, base_filename)
        if bools.get('pickle'):
            utils.save_pickle(df_summary, encoder_output_path, base_filename)
        logger.info(f"\tSaved physical encoder param run summary for {signal_name}.")

        simulation_results = {
            't': t, 'u': u, 'u_rec': u_rec, 's': s,
            'encoder_params': encoder_params,
            'metrics': metrics_dict,
            't_stable': t_stable, 'u_stable': u_stable, 'u_rec_stable': u_rec_stable,
            'stable_start_idx': start_idx, 'stable_end_idx': end_idx,
        }

        if bools.get('pickle'):
            raw_pickle_filename = f"encoder_params_raw_{signal_name}"
            utils.save_pickle(simulation_results, encoder_output_path, raw_pickle_filename)
            logger.info(f"\tSaved physical encoder raw data pickle for {signal_name}.")

        # --- Plotting (Optional) ---
        if bools.get('plots'):
            logger.info(f"\tPlotting physical encoder results for {signal_name}...")
            plot_title = f"Physical Enc {signal_name} (k={k:.1e},b={b:.2f},d={delta:.1e})"
            plot_base = f"encoder_params_{signal_name}"
            plot_t, plot_u, plot_u_rec, plot_suf = (t_stable, u_stable, u_rec_stable, " (Stable)") if t_stable is not None else (t, u, u_rec, " (Full)")

            if plot_t is not None:
                plotting.plot_process(plot_t, plot_u, plot_u_rec, plot_title + plot_suf, os.path.join(encoder_output_path, f"{plot_base}_process.png"))
                plotting.plot_with_spikes(plot_t, plot_u, plot_u_rec, s, plot_title + plot_suf, os.path.join(encoder_output_path, f"{plot_base}_spikes.png"))
            else:
                 logger.warning(f"\tSkipping plots for {signal_name} due to missing plottable data.")

        logger.info(f"\tPhysical param simulation completed successfully for {signal_name}.")
        return simulation_results

    except Exception as e:
        logger.error(f"\tCritical error during physical param simulation/processing for {signal_name}: {e}", exc_info=True)
        return None
    


# --- Main Encoder Study Function ---
def perform_encoder_study(signal_data: SignalDict, optimal_params: OptimalParamsDict, encoder_path: str, bools: BoolsDict, config: Optional[ConfigDict] = None) -> EncoderParamsDict:
    '''
    Determines physical encoder parameters (k, b, delta) from optimal d_norm
    and validates them through simulation.

    Inputs
    ------
    - signal_data: SignalDict
        Dictionary containing signal data including time and signal values.
    - optimal_params: OptimalParamsDict
        Dictionary containing optimal parameters for the signal.
    - encoder_path: str
        Path to save Encoder analysis results.
    - bools: BoolsDict
        Dictionary of boolean flags for workflow control.
    - config: Optional[ConfigDict]
        Configuration dictionary. Default is None.

    Outputs
    -------
    - EncoderParamsDict: EncoderParamsDict
        Dictionary containing the selected k, b, delta, or None.
        Also might return simulation results if needed downstream.
    '''
    signal_name = signal_data.get('name', 'unnamed_signal')
    run_id = f"Encoder_{signal_name}"
    logger.info(f"--- Running {run_id} ---")

    # --- Extract Base Data ---
    t_orig = signal_data.get('t')
    u_orig = signal_data.get('u')
    duration = signal_data.get('dur')
    freq_max = signal_data.get('freq_max')
    d_norm_opt = optimal_params.get(utils.COL_D_NORM)

    if d_norm_opt is None or u_orig is None or freq_max is None:
        logger.error(f"{run_id}: Missing required inputs: 'd_norm' from optimal_params, or 'u'/'freq_max' from signal_data. Aborting.")
        return None
    if d_norm_opt <= 0:
        logger.error(f"{run_id}: Invalid optimal d_norm ({d_norm_opt}). Aborting.")
        return None

    # 1. Estimate c (max amplitude) ---
    try:
        c = np.max(np.abs(u_orig))
        signal_data['c'] = c 
        logger.info(f"\tEstimated max signal amplitude c = {c:.4e}")
    except Exception as e:
        logger.error(f"\tFailed to estimate max amplitude 'c': {e}. Aborting.", exc_info=True)
        return None

    # 2. Estimate Omega (bandwidth in rad/s) ---
    bw = utils.estimate_decoder_bandwidth(freq_max)
    signal_data['bw'] = bw 
    logger.info(f"\tEstimated signal bandwidth Omega = {bw:.4e} rad/s (from freq_max={freq_max} Hz)")


    # --- Structured Selection Process ---
    selected_params: EncoderParamsDict = {'k': None, 'b': None, 'delta': None, 'd_norm_optimal': d_norm_opt}
    params_found = False
    max_iterations = config.get(KEY_MAX_ITERATIONS, 50) # Limit retries
    current_b_factor = config.get(KEY_STABILITY_FACTOR, DEFAULT_STABILITY_FACTOR)
    priority = config.get(KEY_PRIORITY, 'noise') # Default priority

    for iteration in range(max_iterations):
        logger.info(f"\n\t--- Iteration {iteration+1}/{max_iterations} ---")
        # 3. Select b 
        b_selected = current_b_factor * c
        if b_selected <= c : 
            logger.warning(f"\tIteration {iteration+1}: Calculated b ({b_selected:.4e}) not > c ({c:.4e}) with factor {current_b_factor}. Incrementing factor slightly.")
            current_b_factor *= 1.05 
            b_selected = current_b_factor * c
            if b_selected <= c: # Still failing? Abort.
                logger.error("\tCannot select b > c even after adjusting factor. Aborting selection.")
                break # Exit loop

        logger.info(f"\tIteration {iteration+1}: Selected b = {b_selected:.4e} (factor={current_b_factor:.2f}, c={c:.4e})")

        # 4. Calculate Target k*delta 
        target_k_delta = (d_norm_opt * b_selected) / 2.0
        logger.info(f"\tIteration {iteration+1}: Target k*delta = {target_k_delta:.4e}")

        # 5. Select/Calculate k and delta based on Priority ---
        k_calc, delta_calc = _calculate_physical_params(d_norm_opt, b_selected, priority, config)

        if k_calc is None or delta_calc is None:
            logger.warning(f"\tIteration {iteration+1}: Failed to calculate k/delta for priority '{priority}'. Check config/priority. Trying next iteration if possible.")
            # Consider if we should change priority or adjust factors here, maybe increase b factor?
            current_b_factor *= 1.1 
            continue # next iteration

        # 6. Verify r
        r_margin = config.get('R Margin', DEFAULT_R_MARGIN)
        if _verify_recovery_condition(k_calc, delta_calc, b_selected, c, bw, r_margin):
            selected_params['k'] = k_calc
            selected_params['b'] = b_selected
            selected_params['delta'] = delta_calc
            params_found = True
            logger.info(f"\tIteration {iteration+1}: Found valid parameters satisfying r < {r_margin}.")
            break # Exit loop successfully
        else:
            logger.warning(f"\tIteration {iteration+1}: Parameters failed r condition (r >= {r_margin}).")
            # Strategy: Increase b margin for next iteration
            logger.info("\tIncreasing stability factor 'b' for next attempt.")
            current_b_factor *= 1.2 # Increase factor more significantly if r fails
            # Alternative: Could try adjusting k/delta if priority allows

    # --- Output Selection Result ---
    if not params_found:
        logger.error(f"{run_id}: Failed to find suitable k, b, delta after {max_iterations} iterations.")
        return None

    logger.info(f"{run_id}: Successfully selected physical parameters: k={selected_params['k']:.4e}, b={selected_params['b']:.4e}, delta={selected_params['delta']:.4e}")


    # --- Simulate with Selected Parameters ---
    simulation_results = _simulate_with_physical_params(signal_data, selected_params, encoder_path, bools, config)

    if simulation_results:
        logger.info(f"{run_id}: Validation simulation completed.")
        # Add simulation metrics to the output dict if desired
        selected_params['simulation_metrics'] = simulation_results.get('metrics')
    else:
        logger.warning(f"{run_id}: Validation simulation failed or produced no results.")
        selected_params['simulation_metrics'] = {'error': 'Simulation failed'}

    return selected_params
