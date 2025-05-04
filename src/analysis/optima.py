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

# Add a key to indicate the source study
KEY_OPT_SOURCE_STUDY = "source_study_type"

def _filter_and_select(df: pd.DataFrame, id_cols: List[str], error_col: str = utils.COL_MED_ERR, time_col: str = utils.COL_TIME, error_threshold: Optional[float] = None, time_threshold: Optional[float] = None) -> Optional[pd.Series]:
    '''
    Helper to filter DataFrame and select the best row based on minimizing the median error within thresholds.
    Ensures relevant columns are numeric before filtering and sorting.

    The number of candidates is now hardcoded to 10 for simplicity. Could be made customized if needed.
    '''
    # Hardcoded number of candidates to keep for filtering
    n_candidates: int = 10
    # Define all columns needed for processing and output
    metric_cols = list(set([error_col, time_col, utils.COL_MAX_ERR, utils.COL_NRMSE_STD, utils.COL_R2, utils.COL_N_SPIKES])) 
    required_cols = list(set(id_cols + metric_cols)) # All columns that must be present

    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logger.warning(f"_filter_and_select: Input DataFrame missing required columns: {missing}. Cannot proceed.")
        return None
    if df.empty:
        logger.debug("_filter_and_select: Input DataFrame is empty.")
        return None

    try:
        # --- Data Preparation ---
        df_filtered = df[required_cols]
        df_filtered = df_filtered.dropna()
        
        # --- Apply Thresholds ---
        initial_thresh_count = len(df_filtered)
        if error_threshold is not None:
            # Ensure error_col exists and is numeric before filtering
            if error_col in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered[error_col]):
                df_filtered = df_filtered[df_filtered[error_col] <= error_threshold]
                logger.debug(f"\tApplied error threshold ({error_col} <= {error_threshold:.3e}). Kept {len(df_filtered)} / {initial_thresh_count} points.")
                initial_thresh_count = len(df_filtered)
            else:
                logger.warning(f"\tCannot apply error threshold: Column '{error_col}' not found or not numeric after cleaning.")
        else:
            logger.debug(f"\tNo error threshold applied.")

        if time_threshold is not None:
            # Ensure time_col exists and is numeric
            if time_col in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered[time_col]):
                df_filtered = df_filtered[df_filtered[time_col] <= time_threshold]
                logger.debug(f"\tApplied time threshold ({time_col} <= {time_threshold:.3f}s). Kept {len(df_filtered)} / {initial_thresh_count} points.")
            else:
                logger.warning(f"\tCannot apply time threshold: Column '{time_col}' not found or not numeric after cleaning.")
        else:
            logger.debug(f"\tNo time threshold applied.")
            

        # --- Selection Logic ---
        if df_filtered.empty:
            logger.warning("\tNo candidates satisfy the specified thresholds (or thresholds could not be applied).")
            return None
        else:
            # Ensure sorting columns exist and are numeric before sorting
            sort_cols = [error_col, time_col]
            valid_sort_cols = [col for col in sort_cols if col in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered[col])]

            if len(valid_sort_cols) != len(sort_cols):
                missing_sort = [col for col in sort_cols if col not in valid_sort_cols]
                logger.error(f"\tCannot sort results: Required sort columns are missing or not numeric after cleaning: {missing_sort}")
                return None

            # Sort by primary criterion (error) ascending, then tie-breaker (time) ascending
            df_sorted = df_filtered.sort_values(by=valid_sort_cols, ascending=[True, True])
            top_n_candidates = df_sorted.head(n_candidates) # Keep only top candidates
            top_n_candidates_sorted = top_n_candidates.sort_values(by=utils.COL_N_SPIKES, ascending=True)
            best_row = top_n_candidates.iloc[0]
            # Log relevant info - ensure columns exist before accessing
            log_err = best_row.get(error_col, 'N/A')
            log_time = best_row.get(time_col, 'N/A')
            log_fs = best_row.get(utils.COL_FS, 'N/A')
            log_dn = best_row.get(utils.COL_D_NORM, 'N/A')

            logger.info(f"\tSelected best candidate meeting thresholds: "
                        f"{utils.COL_FS}={log_fs}, {utils.COL_D_NORM}={log_dn}, "
                        f"{error_col}={log_err:.3e}, {time_col}={log_time:.3f}s")

            return best_row, top_n_candidates_sorted

    except KeyError as ke:
        # This might still occur if a required column was missing initially
        logger.error(f"_filter_and_select: Missing expected column during processing: {ke}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"_filter_and_select: Error during filtering/selection: {e}", exc_info=True)
        return None


def _find_optimal_conditions(df_bias: Optional[pd.DataFrame], df_delta: Optional[pd.DataFrame], df_2d: Optional[pd.DataFrame], error_threshold: Optional[float] = None, time_threshold: Optional[float] = None, config: Optional[ConfigDict] = None) -> Optional[OptimalParamsDict]:
    '''
    Finds optimal operating conditions (fs, d_norm) from parametric study summary results.

    Inputs
    ------
    - df_bias: pd.DataFrame
        DataFrame containing results from 1D bias sweep.
    - df_delta: pd.DataFrame
        DataFrame containing results from 1D delta sweep.
    - df_2d: pd.DataFrame
        DataFrame containing results from 2D parametric sweep.
    - error_threshold: float, optional
        Maximum allowable median error for selection.
    - time_threshold: float, optional
        Maximum allowable elapsed time for selection.
    - config: dict, optional
        Configuration dictionary containing thresholds and other parameters.

    Outputs
    -------
    - optimal_candidate_row: pd.Series or None
        Series containing the optimal parameters (fs, d_norm) or None if not found.

    Raises
    ------
    - KeyError: If expected columns are missing in the DataFrames.
    - ValueError: If the DataFrames are empty or if the thresholds cannot be applied.
    - Exception: For any other unexpected errors during processing.
    '''
    logger.info("Attempting to find optimal conditions...")
    optimal_candidate_row: Optional[pd.Series] = None
    source_df_type: Optional[str] = None

    # --- Priority 1: 2D Sweep ---
    if df_2d is not None and not df_2d.empty:
        logger.info("Analyzing 2D parametric sweep (Highest Priority)...")
        optimal_candidate_row, optimal_candidates = _filter_and_select(df_2d, id_cols=[utils.COL_B, utils.COL_D_NORM],
                                                    error_threshold=error_threshold, time_threshold=time_threshold)
        if optimal_candidate_row is not None:
            source_df_type = "bias_delta"
            logger.info("Found suitable candidate from 2D sweep.")
        else:
            logger.info("No suitable candidate found meeting criteria in 2D sweep.") 
    else:
        logger.info("2D sweep data not available or empty. Checking 1D sweeps...")

    # --- Priority 2: 1D Bias Sweep (if 2D failed) ---
    if optimal_candidate_row is None and df_bias is not None and not df_bias.empty:
        logger.info("Analyzing 1D Bias sweep (Priority 2)...")
        default_d_norm = config.get('Default Delta')
        if default_d_norm is None:
            logger.warning("Cannot fully evaluate 1D Bias sweep: 'Default Delta' not found in config. Sweep skipped.")
        else:
            try:
                optimal_candidate_row, optimal_candidates = _filter_and_select(df_bias, id_cols=[utils.COL_B, utils.COL_D_NORM],
                                                        error_threshold=error_threshold, time_threshold=time_threshold)

                if optimal_candidate_row is not None:
                    source_df_type = "bias"
                    logger.info(f"Found suitable candidate from 1D Bias sweep (using Default Delta={default_d_norm:.4f}).")
                else:
                    logger.info("No suitable candidate found meeting criteria in 1D Bias sweep.") 
            except ValueError:
                logger.error(f"Could not convert 'Default Delta' ({config.get('Default Delta')}) to float. Skipping 1D bias sweep.")
            except Exception as e:
                logger.error(f"Error processing 1D Bias sweep: {e}", exc_info=True)

    elif optimal_candidate_row is None: 
        logger.debug("1D Bias sweep data not available or empty.") 

    # --- Priority 3: 1D Delta Sweep (if 2D and Bias failed) ---
    if optimal_candidate_row is None and df_delta is not None and not df_delta.empty:
        logger.info("Analyzing 1D Delta sweep (Priority 3)...")
        default_b = config.get('Default Bias')
        if default_b is None:
            logger.warning("Cannot fully evaluate 1D Delta sweep: 'Default Bias' not found in config. Sweep skipped.")
        else:
            try:
                optimal_candidate_row, optimal_candidates = _filter_and_select(df_delta, id_cols=[utils.COL_B, utils.COL_D_NORM],
                                                        error_threshold=error_threshold, time_threshold=time_threshold)

                if optimal_candidate_row is not None:
                    source_df_type = "delta"
                    logger.info(f"Found suitable candidate from 1D Delta sweep (using Default Bias={default_b:.4f}).")
                else:
                    logger.info("No suitable candidate found meeting criteria in 1D Delta sweep.")
            except ValueError:
                logger.error(f"Could not convert 'Default Bias' ({config.get('Default Bias')}) to float. Skipping 1D Delta sweep.")
            except Exception as e:
                logger.error(f"Error processing 1D Delta sweep: {e}", exc_info=True)

    elif optimal_candidate_row is None: # Only log if we didn't find an optimum yet
        logger.debug("1D Delta sweep data not available or empty.") 


    # --- Format Output ---
    if optimal_candidate_row is not None and source_df_type is not None:
        logger.info(f"Final optimal candidate selected from '{source_df_type}' results.")
        try:
            result_dict: OptimalParamsDict = {
                utils.COL_B: optimal_candidate_row[utils.COL_B],
                utils.COL_D_NORM: optimal_candidate_row[utils.COL_D_NORM],
                utils.COL_MAX_ERR: optimal_candidate_row[utils.COL_MAX_ERR],
                utils.COL_MED_ERR: optimal_candidate_row[utils.COL_MED_ERR],
                utils.COL_NRMSE_STD: optimal_candidate_row[utils.COL_NRMSE_STD],
                utils.COL_R2: optimal_candidate_row[utils.COL_R2],
                utils.COL_TIME: optimal_candidate_row[utils.COL_TIME],
                utils.COL_N_SPIKES: optimal_candidate_row[utils.COL_N_SPIKES],
                KEY_OPT_SOURCE_STUDY: source_df_type
            }
            # Format log string more carefully for potential non-numeric types if error occurs
            log_items = []
            for k,v in result_dict.items():
                try:
                    log_items.append(f"{k}={v:.4g}" if isinstance(v, (float, int)) else f"{k}={v}")
                except TypeError:
                    log_items.append(f"{k}={v}") # Fallback if formatting fails
            log_str = "\tFinal Optimal Values: " + ", ".join(log_items)
            logger.info(log_str)
            return result_dict, optimal_candidates

        except Exception as format_err:
            logger.error(f"Error formatting optimal results dictionary from row: {optimal_candidate_row.to_dict()}. Error: {format_err}", exc_info=True)
            return None
    else:
        logger.warning("Could not determine optimal conditions meeting criteria from any available study data.")
        return None
    

def _find_signal_optima_step(study_results_for_signal: Optional[ParametricResultsDict], config: ConfigDict, signal_name: str) -> Optional[OptimalParamsDict]:
    '''
    Finds optimal parameters (bias, d_norm) for one signal based on available parametric summaries.
    Prioritizes bias_delta > bias > delta studies.
    Minimizes median error ('error_median' column assumed) within thresholds.

    Inputs
    - study_results_for_signal: dict
        Dictionary containing DataFrames for 'bias', 'delta', and 'bias_delta' studies.
        Each DataFrame should contain the relevant columns for analysis.
    - config: dict
        Configuration dictionary containing thresholds and other parameters.    
    - signal_name: str
        Name of the signal being processed (for logging and output purposes).
    
    Outputs
    -------
    - optimal_params: dict or None
        Dictionary containing optimal parameters (bias, d_norm) or None if not found.

    Raises
    ------
    - KeyError: If expected columns are missing in the DataFrames.
    '''
    logger.info(f"--- Finding Optimal Conditions [{signal_name}] ---")

    # Input validation
    if not study_results_for_signal or all(df is None or df.empty for df in study_results_for_signal.values()):
        logger.warning(f"\tNo valid parametric summary data available for {signal_name}. Cannot find optima.")
        return None

    # Extract DataFrames from the input dictionary
    df_bias = study_results_for_signal.get('bias')
    df_delta = study_results_for_signal.get('delta')
    df_2d = study_results_for_signal.get('bias_delta')

    try:
        error_thresh = config.get('Error threshold')
        time_thresh = config.get('Elapsed time threshold')

        optimal_params, optimal_candidates = _find_optimal_conditions(df_bias, df_delta, df_2d, error_thresh, time_thresh, config)

        if optimal_params:
            logger.info(f"\tOptimal conditions found: b={optimal_params.get(utils.COL_B, 'N/A')}, d_norm={optimal_params.get(utils.COL_D_NORM, 'N/A')}")
            return optimal_params, optimal_candidates
        else:
            logger.warning(f"\tNo suitable optimal conditions found for {signal_name} meeting criteria.")
            return None

    except KeyError as e:
        logger.error(f"\tOptima finding failed for {signal_name}: Missing expected column in summary data: {e}.", exc_info=False)
        return None
    except Exception as e:
        logger.error(f"\tUnexpected error during optima finding for {signal_name}: {e}", exc_info=True)
        return None

def _simulate_and_save_optimal_step(signal_data: SignalDict, optimal_params: OptimalParamsDict, optimal_output_path: Optional[str], bools: BoolsDict, config: ConfigDict) -> Optional[OptimalSimResultDict]:
    '''
    Runs ASDM encode/decode using the optimal parameters found.
    Calculates metrics for this specific run (can use stable region).
    Saves results (summary, raw data, plots).
    Returns a dictionary containing key results needed for subsequent analyses (t, u_rec, s, N_spikes).

    Inputs
    ------
    - signal_data: dict
        Dictionary containing signal data (t, u, dur, b, dte).
    - optimal_params: dict
        Dictionary containing optimal parameters (bias, d_norm).
    - optimal_output_path: str
        Path to save the results.
    - bools: dict
        Dictionary containing boolean flags for saving and plotting.
    - config: dict

    Outputs
    -------
    - simulation_results: dict or None
        Dictionary containing simulation results (t, u_rec, s, N_spikes) or None if simulation failed.
    
    Raises
    ------
    - ValueError: If required parameters are missing or invalid.
    '''
    signal_name = signal_data.get('name', 'unnamed_signal')
    logger.info(f"--- Simulating Optimal Conditions [{signal_name}] ---")

    # --- Validate Inputs & Setup ---
    if not optimal_output_path:
        logger.warning(f"\tSkipping optimal simulation for {signal_name}: Output path not provided.")
        return None

    try:
        # Extract required data (combined check)
        t = signal_data.get('t')
        u = signal_data.get('u')
        dur = signal_data.get('dur')
        dte = signal_data.get('dte')
        fs = config.get('Default Frequency')
        b = optimal_params.get(utils.COL_B)
        d_norm = optimal_params.get(utils.COL_D_NORM)
        freq_max = signal_data.get('freq_max') 

        required_data = {'t': t, 'u': u, 'dur': dur, 'b': b, 'dte': dte, 'fs': fs, 'd_norm': d_norm, 'freq_max': freq_max}
        missing = [k for k, v in required_data.items() if v is None]
        invalid_numeric = [k for k, v in {'fs': fs, 'd_norm': d_norm, 'b':b, 'freq_max': freq_max}.items() if v is None or v <= 0 or (k == 'b' and abs(v) < 1e-12) ]

        if missing or invalid_numeric:
            logger.error(f"\tSkipping optimal simulation for {signal_name}: Missing ({missing}) or invalid ({invalid_numeric}) data.")
            return None
        if not isinstance(t, np.ndarray) or not isinstance(u, np.ndarray) or t.shape != u.shape or len(t) == 0:
            logger.error(f"\tSkipping optimal simulation for {signal_name}: Invalid 't' or 'u' signal data.")
            return None

    except OSError as e:
        logger.error(f"\tFailed to create Optimal output directory '{optimal_output_path}': {e}. Skipping.", exc_info=False)
        return None
    except Exception as e:
        logger.error(f"\tError during setup for optimal simulation [{signal_name}]: {e}", exc_info=True)
        return None

    logger.info(f"\tRunning simulation with optimal b={b:.4f}, d_norm={d_norm:.4f}...")
    start_time = time.time()
    u_rec = None
    s = None
    metrics_dict = None
    simulation_results: Optional[OptimalSimResultDict] = None
    t_stable, u_stable, u_rec_stable = None, None, None
    start_idx, end_idx = None, None
    try:
        # --- Run Simulation (Encode/Decode) ---
        bw = utils.estimate_decoder_bandwidth(freq_max)
        logger.debug(f"\t(Optimal) Using decoder bandwidth (rad/s): {bw:.2f}")

        # Encode
        s = asdm.asdm_encode(u/b, 1/fs, d_norm, dte)
        if s is None or len(s) < 2:
            logger.error(f"\t(Optimal) Encoding failed or produced insufficient spikes for {signal_name}.")
            return None
        N_spikes = len(s)
        logger.debug(f"\t(Optimal) Encoding produced {N_spikes} spikes.")

        # Decode
        u_rec = asdm.asdm_decode(s, dur, 1/fs, bw)
        if u_rec is None or len(u_rec) == 0:
            logger.warning(f"\t(Optimal) Decoding failed or produced empty signal. Skipping metrics.")
            return None

        # Align length (simple truncation/padding) 
        if len(u_rec) != len(t):
            logger.warning(f"\t(Optimal) Length mismatch: Original ({len(t)}) vs Reconstructed ({len(u_rec)}). Aligning...")
            u_rec = np.zeros_like(t) # Create array of target shape
            common_length = min(len(u_rec), len(t))
            u_rec[:common_length] = u_rec[:common_length] # Copy common part
        else:
            u_rec = u_rec # No alignment needed

        # --- Stable Region & Metrics ---
        stable_percentage = config.get('Stable Region Percentage', 0.9)
        try:
            # Get percentage from config, default to 0.9 (90%)
            stable_percentage = config.get('Stable Region Percentage', 0.9)
            logger.debug(f"Using stable region percentage: {stable_percentage:.2f}")
            start_idx, end_idx = utils.get_stable_region(t, stable_percentage)
            # Check if get_stable_region returned fallback (0, len(t)) due to error/short signal
            if end_idx == len(t) and start_idx == 0 and len(t)>1: # Avoid trigger on len=1 case
                logger.info(f"  Stable region covers full signal [{start_idx}:{end_idx}].")
            else:
                logger.info(f"  Using stable region indices [{start_idx}:{end_idx}] for metrics.")

        except Exception as e:
            # Catch unexpected errors during calculation itself
            logger.error(f"Unexpected error getting stable region for signal '{signal_name}': {e}. Using full signal.", exc_info=True)
            start_idx, end_idx = 0, len(t) # Fallback to full signal
        
        try:
            t_stable = t[start_idx:end_idx]
            u_stable = u[start_idx:end_idx]
            u_rec_stable = u_rec[start_idx:end_idx]
            metrics_dict = metrics.calculate_asdm_metrics(u_stable/b, s, u_rec_stable, start_time) 
            if metrics_dict:
                metrics_dict[utils.COL_STABLE_START_IDX] = start_idx
                metrics_dict[utils.COL_STABLE_END_IDX] = end_idx
                logger.info(f"\tOptimal Run Metrics (Stable Region): MedErr={metrics_dict.get(utils.COL_MED_ERR, 'N/A'):.4f}, NSpikes={N_spikes}, Time={metrics_dict.get(utils.COL_TIME, 'N/A'):.4f}s")
            else:
                logger.warning(f"\tMetric calculation failed for {signal_name} on stable region.")
                metrics_dict = {'error_notes': "Stable region metric calculation failed."}
        except Exception as metric_err:
            logger.error(f"\tError during stable region/metric calculation for {signal_name}: {metric_err}", exc_info=False)
            metrics_dict = {'error_notes': f"Metric calculation error: {metric_err}"}

        # --- Save Summary & Prepare Output ---
        optimal_summary_data = {
            **optimal_params, 
            **metrics_dict, 
            utils.COL_N_SPIKES: N_spikes} 
        df_optimal_summary = pd.DataFrame([optimal_summary_data])
        base_filename = f"optimal_summary_{signal_name}"
        utils.store_df_to_excel(df_optimal_summary, optimal_output_path, base_filename)
        if bools.get('pickle'):
            utils.save_pickle(df_optimal_summary, optimal_output_path, base_filename)
        logger.info(f"\tSaved optimal run summary for {signal_name}.")

        # Prepare results dictionary for subsequent steps and optional raw saving
        simulation_results = {
            't': t, 'u': u, 'u_rec': u_rec, 's': s,
            utils.COL_N_SPIKES: N_spikes, utils.COL_FS: fs, utils.COL_D_NORM: d_norm,
            utils.COL_B: b, utils.COL_DTE: dte, 'metrics': metrics_dict,
            't_stable': t_stable, 'u_stable': u_stable, 'u_rec_stable': u_rec_stable,
            'stable_start_idx': start_idx, 'stable_end_idx': end_idx, 
            'optimal_params_input': optimal_params 
        }

        if bools.get('pickle'):
            raw_pickle_filename = f"optimal_raw_{signal_name}"
            utils.save_pickle(simulation_results, optimal_output_path, raw_pickle_filename)
            logger.info(f"\tSaved optimal raw data pickle for {signal_name}.")

        # --- Plotting (Optional) ---
        if bools.get('plots'):
            logger.info(f"\tPlotting optimal results for {signal_name}...")
            plot_title = f"Optimal {signal_name} (b={b:.4f}, d={d_norm:.4f})"
            plot_base = f"optimal_{signal_name}"

            # Decide what to plot based on valid stable region data
            plot_t, plot_u, plot_u_rec, plot_suf = (t_stable, u_stable, u_rec_stable, " (Stable)") if t_stable is not None else (t, u, u_rec, " (Full)")

            if plot_t is not None: # Check if we have *any* data to plot
                plotting.plot_process(plot_t, plot_u/b, plot_u_rec, plot_title + plot_suf, os.path.join(optimal_output_path, f"{plot_base}_process.png"))
                plotting.plot_with_spikes(plot_t, plot_u/b, plot_u_rec, s, plot_title + plot_suf, os.path.join(optimal_output_path, f"{plot_base}_spikes.png"))
            else:
                 logger.warning(f"\tSkipping optimal plots for {signal_name} due to missing plottable data.")


        logger.info(f"\tOptimal simulation completed successfully for {signal_name}.")
        return simulation_results

    except Exception as e:
        logger.error(f"\tCritical error during optimal simulation/processing for {signal_name}: {e}", exc_info=True)
        return None 


def perform_optima_study(signal_data: SignalDict, results_for_this_signal: Optional[ParametricResultsDict], optima_path: str, bools: BoolsDict, config: Optional[ConfigDict] = None) -> Tuple[Optional[OptimalParamsDict], Optional[OptimalSimResultDict]]:
    '''
    Performs the optimal simulation study for a given signal.

    Inputs
    ------
    - signal_data: SignalDict
        Dictionary containing signal data including time and signal values.
    - resuls_for_this_signal: ParametricResultsDict
        Dictionary containing parametric study results for the signal.
    - optima_path: str
        Path to save Optima analysis results.
    - bools: BoolsDict
        Dictionary of boolean flags for workflow control.
    - config: Optional[ConfigDict]
        Configuration dictionary. Default is None.

    Outputs
    -------
    - optimal_params: Optional[OptimalParamsDict]
        Dictionary containing optimal parameters (bias, d_norm) or None if not found.
    - optima_sim_result: Optional[OptimalSimResultDict]
        DataFrame containing Optima analysis results or None if errors occurred.
    '''
    signal_name = signal_data.get('name', 'unnamed_signal')
    run_id = f"Optima_{signal_name}"
    logger.info(f"--- Running {run_id} ---")

    # 1. Find Optimal Parameters    
    optimal_params: Optional[OptimalParamsDict] = None
    optimal_params, optimal_candidates = _find_signal_optima_step(results_for_this_signal, config, signal_name)
    if optimal_params is None:
        logger.warning(f"Skipping optimal simulation for {signal_name}: No suitable parameters found.")
        return None, None
    if optimal_candidates is not None:
        base_filename = f"optimal_candidates_{signal_name}"
        utils.store_df_to_excel(optimal_candidates, optima_path, base_filename)
        logger.info(f"Optimal candidates for {signal_name} saved to {base_filename}.")

    
    # 2. Simulate with Optimal Parameters (if optima found and needed)
    optimal_sim_result: Optional[OptimalSimResultDict] = None
    optimal_sim_result = _simulate_and_save_optimal_step(signal_data, optimal_params, optima_path, bools, config)
    if optimal_sim_result:
        N_spikes_optimal = optimal_sim_result.get(utils.COL_N_SPIKES) 
        logger.info(f"Optimal simulation for {signal_name} completed. N_spikes = {N_spikes_optimal}")
    else:
        logger.error(f"Optimal simulation FAILED for {signal_name}. Subsequent analyses might fail.")
        return optimal_params, None

    return optimal_params, optimal_sim_result
