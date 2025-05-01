import os
import sys
import numpy as np
import time
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple 
import logging

logger = logging.getLogger(__name__)

# custom modules
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, src_path)
from src.utilities import utils
from src.utilities import asdm
from src.utilities import logging as project_logging
from src.utilities import plotting
from utilities import metrics

# Type alias for metrics dictionary
MetricsDict = Dict[str, Any]

def _run_single_simulation(
    u: np.ndarray, t: np.ndarray, dur: float, fs: float, d_norm: float, b: float, dte: float, freq_max: float,
    start_idx: int, end_idx: int, 
    signal_name: str, output_dir: Optional[str], study_type_tag: str, log_study: bool = False, plot_study: bool = False, save_raw: bool = False,
    base_log_filename: Optional[str] = None) -> Optional[MetricsDict]:
    '''
    Runs a single encode/decode simulation, calculates metrics on the stable region,
    and handles optional logging, plotting, and raw data saving.
    Uses provided stable region indices. Assumes additive bias model.
    '''
    start_time = time.time()
    run_id = f"fs={fs:.0f}_d={d_norm:.4f}" # Identifier for logging this specific run
    logger.info(f"\tSimulating ({run_id}): b={b:.3f}, dte={dte:.1e}...")

    # --- Core Encode/Decode ---
    s = None
    u_rec = None
    try:
        bw = utils.estimate_decoder_bandwidth(freq_max) 
        logger.debug(f"\t({run_id}) Using decoder bandwidth (rad/s): {bw:.2f}")

        # Encode
        s = asdm.asdm_encode(u/b, 1/fs, d_norm, dte)
        if s is None or len(s) < 2: 
            logger.warning(f"\t({run_id}) Encoding failed or produced < 2 spikes ({len(s) if s is not None else 0}). Skipping decode & metrics.")
            return None
        
        # Decode
        u_rec = asdm.asdm_decode(s, dur, 1/fs, bw)
        if u_rec is None or len(u_rec) == 0:
            logger.warning(f"\t({run_id}) Decoding failed or produced empty signal. Skipping metrics.")
            return None
        
        # Align length (simple truncation/padding) 
        if len(u_rec) != len(t):
            logger.warning(f"\tLength mismatch: Original ({len(t)}) vs Reconstructed ({len(u_rec)}). Aligning...")
            u_rec = np.zeros_like(t) # Create array of target shape
            common_length = min(len(u_rec), len(t))
            u_rec[:common_length] = u_rec[:common_length] # Copy common part
        else:
            u_rec = u_rec # No alignment needed


    except ValueError as ve: # Catch specific errors from asdm if possible
        logger.error(f"\t({run_id}) ValueError during ASDM: {ve}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"\t({run_id}) Error during ASDM encode/decode: {e}", exc_info=True)
        return None

    # --- Prepare Stable Region Data ---
    t_stable, u_stable, u_rec_stable = None, None, None
    try:        
        t_stable = t[start_idx:end_idx]
        u_stable = u[start_idx:end_idx]
        u_rec_stable = u_rec[start_idx:end_idx]

    except IndexError as ie: # Catch potential slicing errors
        logger.error(f"\t({run_id}) IndexError preparing stable region data: {ie}. Indices: ({start_idx}, {end_idx}), t_len={len(t)}, u_rec_len={len(u_rec)}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"\t({run_id}) Error preparing stable region data: {e}", exc_info=True)
        return None

    # --- Calculate Metrics ---
    metrics_dict = None
    try:
        metrics_dict = metrics.calculate_asdm_metrics(u_stable/b, s, u_rec_stable, start_time) # Think if you need to pass the original u_stable or the normalized one (u_stable/b)
        if metrics_dict is None:
            logger.error(f"\t({run_id}) Metric calculation failed or returned None.")
            return None
    except Exception as e:
        logger.error(f"\t({run_id}) Error calling calculate_asdm_metrics: {e}", exc_info=True)
        return None

    # Add simulation parameters back into metrics dict for aggregation/logging
    metrics_dict[utils.COL_FS] = fs           
    metrics_dict[utils.COL_D_NORM] = d_norm   
    metrics_dict[utils.COL_B] = b                    
    metrics_dict[utils.COL_DTE] = dte    
    metrics_dict[utils.COL_BW] = bw            

    # Log summary of the run completion
    logger.info(f"\t({run_id}) Done. MaxErr={metrics_dict.get(utils.COL_MAX_ERR, np.nan):.2e}, MedErr={metrics_dict.get(utils.COL_MED_ERR, np.nan):.2e}, NSpikes={metrics_dict.get(utils.COL_N_SPIKES, 0)}, Time={metrics_dict.get(utils.COL_TIME, np.nan):.2f}s")
    
    # --- Save Raw Data (Conditional) ---
    if save_raw and output_dir:
        try:
            raw_data_to_save = {
                't_stable': t_stable, 'u_stable': u_stable, 'u_rec_stable': u_rec_stable, 
                's': s, utils.COL_BW: bw, 
                **metrics_dict
            }
            fs_str = f"{fs:.0f}".replace('.', 'p')
            dnorm_str = f"{d_norm:.4f}".replace('.', 'p')
            pickle_filename = f"raw_{study_type_tag}_{signal_name}_f{fs_str}_d{dnorm_str}" # No extension needed for util function
            logger.debug(f"\t({run_id}) Saving raw data to {pickle_filename}.pkl")
            utils.save_pickle(raw_data_to_save, output_dir, pickle_filename)
        except Exception as e:
            # Log warning but continue, saving raw data is optional
            logger.warning(f"\t({run_id}) Failed to save raw pickle data: {e}", exc_info=True)

    # --- Logging Run Metrics (Conditional) ---
    if log_study and output_dir and base_log_filename:
        try:
            full_log_path = os.path.join(output_dir, base_log_filename)
            project_logging.log_parametric_run(full_log_path, metrics_dict)
            logger.debug(f"\t({run_id}) Appended metrics to log: {base_log_filename}")
        except Exception as e:
            logger.warning(f"\t({run_id}) Failed to log run data to {base_log_filename}: {e}", exc_info=True)

    # --- Plotting (Conditional) ---
    if plot_study and output_dir:
        # Generate filenames
        fs_str = f"{fs:.0f}".replace('.', 'p')
        dnorm_str = f"{d_norm:.4f}".replace('.', 'p')
        plot_title = f"{signal_name} ({run_id})" # Include run params in title
        base_filename = f"plot_{study_type_tag}_{signal_name}_f{fs_str}_d{dnorm_str}"

        # Plot 1: Process (Original vs Reconstructed vs Error)
        try:
            plot_filename_proc = f"{base_filename}_process.png" # Add suffix
            logger.debug(f"\t({run_id}) Plotting process to {plot_filename_proc}")
            plotting.plot_process(t_stable, u_stable/b, u_rec_stable, plot_title, os.path.join(output_dir, plot_filename_proc))
        except Exception as e:
            logger.warning(f"\t({run_id}) Failed to plot process data: {e}", exc_info=True)

        # Plot 2: Spikes (Original vs Spikes vs Reconstructed)
        try:
            plot_filename_spk = f"{base_filename}_spikes.png" # Add suffix
            logger.debug(f"\t({run_id}) Plotting spikes to {plot_filename_spk}")
            plotting.plot_with_spikes(t_stable, u_stable/b, u_rec_stable, s, plot_title, os.path.join(output_dir, plot_filename_spk))
        except Exception as e:
            logger.warning(f"\t({run_id}) Failed to plot spike data: {e}", exc_info=True)

    return metrics_dict


# --- Public Study Functions ---
def parametric_bias(u: np.ndarray, t: np.ndarray, dur: float, fs: float, d_norm: float, bias: np.ndarray, dte: float, freq_max: float, start_idx: int, end_idx: int, log_filename: str, output_dir: str, signal_name: str, log_study: bool = False, plot_study: bool = False, save_raw: bool = False) -> Optional[pd.DataFrame]:
    '''Parametric study on encoder bias'''
    metrics_results_list: List[MetricsDict] = []
    study_tag = "PB" # Tag for filenames

    # Basic checks using logger
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory for {study_tag} study: {output_dir}")
        except OSError as e:
            logger.error(f"Error ({study_tag} {signal_name}): Could not create output directory {output_dir}: {e}", exc_info=True)
            return None # Cannot proceed

    # Setup logging header (if logging enabled)
    full_log_path = None
    if log_study and output_dir and log_filename:
         try:
            full_log_path = os.path.join(output_dir, log_filename)
            fixed_params = {utils.COL_D_NORM: d_norm, utils.COL_FS: fs, utils.COL_DTE: dte}
            # Use the specific logging function for headers
            project_logging.log_study_header(full_log_path, f"Bias Sweep ({signal_name})", bias, fixed_params)
            logger.info(f"Initialized study log file: {log_filename}")
         except Exception as e:
            logger.error(f"Failed to initialize study log file {log_filename}: {e}", exc_info=True)
            # Decide if study should abort if logging fails? For now, continue.
            log_study = False # Disable further logging for this study if header failed

    # Loop through encoder bias
    for b in bias:
        try:
            # Call the simulation helper
            metrics_dict = _run_single_simulation(u, t, dur, fs, d_norm, b, dte, freq_max,
                start_idx, end_idx,
                signal_name, output_dir, study_tag,
                log_study, plot_study, save_raw, log_filename
            )
            if metrics_dict:
                metrics_results_list.append(metrics_dict)
        except Exception as e:
            # Catch unexpected errors from the simulation call itself
            logger.error(f"  Critical unhandled error during simulation call for b={b}: {e}. Skipping.", exc_info=True)
            continue # Continue to the next bias value

    # Process results
    if not metrics_results_list:
        logger.warning(f"Parametric study on bias ({signal_name}) yielded no valid results.")
        return None

    try:
        df_bias = pd.DataFrame(metrics_results_list)
        logger.info(f"Parametric study on bias ({signal_name}) complete. {len(df_bias)} points successfully processed.")
        return df_bias
    except Exception as e:
        logger.error(f"Failed to create DataFrame from bias study results for {signal_name}: {e}", exc_info=True)
        return None



def parametric_delta(u: np.ndarray, t: np.ndarray, dur: float, fs: float, deltas: np.ndarray, b: float, dte: float, freq_max: float, start_idx: int, end_idx: int, log_filename: str, output_dir: str, signal_name: str, log_study: bool = False, plot_study: bool = False, save_raw: bool = False) -> Optional[pd.DataFrame]:
    '''Parametric study on normalized threshold (delta).'''
    metrics_results_list: List[MetricsDict] = []
    study_tag = "PD"

    if abs(b) < 1e-12:
        logger.error(f"Error ({study_tag} {signal_name}): Encoder bias 'b' is zero or too close. Aborting.")
        return None
    
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory for {study_tag} study: {output_dir}")
        except OSError as e:
            logger.error(f"Error ({study_tag} {signal_name}): Could not create output directory {output_dir}: {e}", exc_info=True)
            return None # Cannot proceed

    # Setup logging header
    full_log_path = None
    if log_study and output_dir and log_filename:
         try:
            full_log_path = os.path.join(output_dir, log_filename)
            fixed_params = {utils.COL_B: b, utils.COL_FS: fs, utils.COL_DTE: dte}
            project_logging.log_study_header(full_log_path, f"Delta Sweep ({signal_name})", deltas, fixed_params)
            logger.info(f"Initialized study log file: {log_filename}")
         except Exception as e:
            logger.error(f"Failed to initialize study log file {log_filename}: {e}", exc_info=True)
            log_study = False

    # Loop through deltas
    for d_norm in deltas:
        try:
            metrics_dict = _run_single_simulation(u, t, dur, fs, d_norm, b, dte, freq_max,
                start_idx, end_idx,
                signal_name, output_dir, study_tag,
                log_study, plot_study, save_raw, log_filename
            )
            if metrics_dict:
                metrics_results_list.append(metrics_dict)
        except Exception as e:
            logger.error(f"  Critical unhandled error during simulation call for d_norm={d_norm}: {e}. Skipping.", exc_info=True)
            continue

    # Process results
    if not metrics_results_list:
        logger.warning(f"Parametric study on d_norm ({signal_name}) yielded no valid results.")
        return None
    
    try:
        df_delta = pd.DataFrame(metrics_results_list)
        logger.info(f"Parametric study on d_norm ({signal_name}) complete. {len(df_delta)} points successfully processed.")
        return df_delta
    except Exception as e:
        logger.error(f"Failed to create DataFrame from delta study results for {signal_name}: {e}", exc_info=True)
        return None


def parametric_bias_delta(u: np.ndarray, t: np.ndarray, dur: float, fs: float, deltas: np.ndarray, bias: np.ndarray, dte: float, freq_max: float, start_idx: int, end_idx: int, log_filename: str, output_dir: str, signal_name: str, log_study: bool = False, plot_study: bool = False, save_raw: bool = False) -> Optional[pd.DataFrame]:
    '''Bi-Parametric study on encoder bias and normalized threshold.'''
    metrics_results_list: List[MetricsDict] = []
    study_tag = "PBD"

    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory for {study_tag} study: {output_dir}")
        except OSError as e:
            logger.error(f"Error ({study_tag} {signal_name}): Could not create output directory {output_dir}: {e}", exc_info=True)
            return None 

    # Setup logging header
    full_log_path = None
    if log_study and output_dir and log_filename:
        try:
            full_log_path = os.path.join(output_dir, log_filename)
            fixed_params = {utils.COL_FS: fs, utils.COL_DTE: dte}
            param_ranges = {utils.COL_B: bias, utils.COL_D_NORM: deltas} # Pass dict for biparametric ranges
            project_logging.log_study_header(full_log_path, f"Bias/Delta Sweep ({signal_name})", param_ranges, fixed_params)
            logger.info(f"Initialized study log file: {log_filename}")
        except Exception as e:
            logger.error(f"Failed to initialize study log file {log_filename}: {e}", exc_info=True)
            log_study = False


    # Optional: Warning about large number of outputs
    num_points = len(bias) * len(deltas)
    if num_points > 1000 and (plot_study or save_raw):
        logger.warning(f"Plotting/Saving raw data enabled for large biparametric study ({num_points} points) for {signal_name}. This may generate many files/take time.")

    total_processed = 0
    # Loop through bias and deltas
    for b in bias:
        for d_norm in deltas:
            try:
                metrics_dict = _run_single_simulation(u, t, dur, fs, d_norm, b, dte, freq_max,
                    start_idx, end_idx,
                    signal_name, output_dir, study_tag,
                    log_study, plot_study, save_raw, log_filename
                )
                if metrics_dict:
                    metrics_results_list.append(metrics_dict)
                    total_processed += 1 # Count successful runs
            except Exception as e:
                logger.error(f"  Critical unhandled error during simulation call for b={b}, d_norm={d_norm}: {e}. Skipping.", exc_info=True)
                continue

    # Process results
    if not metrics_results_list:
        logger.warning(f"Bi-Parametric study ({signal_name}) yielded no valid results.")
        return None

    try:
        df_2d = pd.DataFrame(metrics_results_list)
        logger.info(f"Bi-Parametric study ({signal_name}) complete. {total_processed}/{num_points} points successfully processed.")
        return df_2d
    except Exception as e:
        logger.error(f"Failed to create DataFrame from biparametric study results for {signal_name}: {e}", exc_info=True)
        return None
    

