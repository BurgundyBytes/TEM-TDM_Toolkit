import os
import sys
import pandas as pd
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging


logger = logging.getLogger(__name__)

# custom modules
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, src_path)
from src.analysis import nyquist, optima, fourier
from src.utilities import metrics
from src.utilities import asdm, plotting, utils


# --- Data Types ---
ConfigDict = Dict[str, Any]
BoolsDict = Dict[str, bool]
OutputPathsDict = Dict[str, Optional[str]]
SignalDict = Dict[str, Any]
SignalResultsDict = Dict[str, Optional[pd.DataFrame]]
ParametricResultsDict = Dict[str, SignalResultsDict]
OptimalParamsDict = Dict[str, Any] 
OptimalSimResultDict = Dict[str, Any] 
FFTResultDict = Dict[str, Dict[str, Any]]


# --- Helper: Load Parametric Data ---
def _load_summary_data(output_paths: OutputPathsDict, load_from_pickle: bool) -> Optional[ParametricResultsDict]:
    '''
    Loads previously saved parametric study SUMMARY results from files for all signals.

    Inputs
    ------
    - output_paths: dict
        Dictionary containing paths for output directories for different study types.
    - load_from_pickle: bool
        Flag indicating whether to load from pickle files (True) or Excel files (False).
    
    Outputs
    -------
    - loaded_summary_results: dict or None
        Dictionary containing loaded parametric summary results for each signal.

    Raises
    ------
    - OSError: If there are issues accessing the output folders or files.   
    - FileNotFoundError: If the specified folder does not exist or is invalid.
    - Exception: For any unexpected errors during the loading process.    
    '''
    logger.info(f"Attempting to load parametric summary data (from {'Pickle' if load_from_pickle else 'Excel'}).")
    loaded_summary_results: ParametricResultsDict = {}
    study_map = {
        "bias":       ('param_bias', 'parametric_bias_summary'),
        "delta":      ('param_delta', 'parametric_delta_summary'),
        "bias_delta": ('biparametric', 'parametric_biasDelta_summary')
    }
    file_ext = ".pkl" if load_from_pickle else ".xlsx"
    load_func = utils.load_pickle if load_from_pickle else utils.load_df_from_excel
    found_any = False

    for study_type, (path_key, base_filename) in study_map.items():
        folder = output_paths.get(path_key)
        if not folder or not os.path.isdir(folder):
            logger.debug(f"Output folder for study '{study_type}' ('{folder}') not found or invalid. Skipping.")
            continue

        prefix = f"{base_filename}_"
        try:
            files = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(file_ext)]
            if not files:
                logger.debug(f"No '{prefix}*{file_ext}' summary files found in '{folder}'.")
                continue

            logger.info(f"Found {len(files)} potential {study_type} summary file(s) in '{folder}'.")
            for filename in files:
                filepath = os.path.join(folder, filename)
                try:
                    signal_name = filename[len(prefix):-len(file_ext)]
                    if not signal_name:
                        logger.warning(f"Could not extract signal name from '{filename}'. Skipping.")
                        continue

                    logger.debug(f"\tLoading {study_type} summary for '{signal_name}' from '{filename}'...")
                    df = load_func(filepath)

                    if isinstance(df, pd.DataFrame) and not df.empty:
                        if signal_name not in loaded_summary_results:
                            loaded_summary_results[signal_name] = {"bias": None, "delta": None, "bias_delta": None}
                        loaded_summary_results[signal_name][study_type] = df
                        logger.debug(f"\tSuccessfully loaded summary for '{signal_name}', study '{study_type}'. Shape: {df.shape}")
                        found_any = True
                    else:
                        logger.warning(f"\tLoading from '{filename}' failed or returned empty/invalid data. Skipping.")

                except Exception as e:
                    logger.error(f"\tError processing file '{filename}': {e}. Skipping.", exc_info=False) # Keep exc_info=False for brevity unless debugging needed

        except OSError as e:
            logger.warning(f"Error accessing folder '{folder}': {e}. Skipping loading for {study_type}.", exc_info=False)
            continue

    if not found_any:
        logger.warning("Loading complete: No valid parametric summary data loaded.")
        return None
    else:
        logger.info("Parametric summary data loading process complete.")
        return loaded_summary_results



# --- Main Manager ---
def manager(config: ConfigDict, bools: BoolsDict, signals: List[SignalDict], output_paths: OutputPathsDict, parametric_summary_results: Optional[ParametricResultsDict] = None) -> None:
    '''
    Manages the post-processing analysis workflow (Optima, Optimal Simulation, Nyquist, Fourier).
    Orchestrates finding optimal parameters, running simulation with them, and performing evaluations.

    Inputs
    ------
    - config: dict
        Dictionary containing configuration settings for parametric studies.
    - bools: dict
        Dictionary of boolean flags to control execution, logging, and output options.
    - signals: list of dicts
        List of input signals, each represented as a dictionary with relevant parameters.
    - output_paths: dict
        Dictionary containing paths for output directories for different study types.
    - parametric_summary_results: dict or None
        Dictionary containing parametric summary results for each signal, if available.

    Raises
    ------
    - KeyError: If expected keys are missing in the input dictionaries.
    - Exception: For any unexpected errors during the analysis workflow.
    '''
    logger.info("====== Results Handler Manager ======")

    # --- Get Parametric Summary Data ---
    # Decide if loading is needed based on which analyses are requested
    load_needed = bools.get('optima', False) or bools.get('fourier', False) or bools.get('nyquist', False)

    study_summary_data: Optional[ParametricResultsDict] = {}
    if parametric_summary_results is not None:
        logger.info("Using live parametric summary results passed from the execution flow.")
        study_summary_data = parametric_summary_results
    elif load_needed:
        logger.info("No live parametric results provided. Attempting to load saved summary data...")
        # Determine if loading from pickle based on bools, default to False (Excel)
        load_pickle = bools.get('pickle', False)
        study_summary_data = _load_summary_data(output_paths, load_pickle)
        if study_summary_data is None:
            logger.warning("Failed to load any saved parametric summary data. Analyses requiring optimal parameters will likely fail.")
            # Decide whether to continue or exit? For now, continue and let steps fail individually.
    else:
        logger.info("Parametric summary data not required or provided live. Loading skipped.")

    # Initialize Results Storage 
    # This is optional, just in case we want to keep track of all results in one place
    # (e.g., for saving to a single file later)
    all_optimal_params: Dict[str, Optional[OptimalParamsDict]] = {}
    all_optimal_sim_results: Dict[str, Optional[OptimalSimResultDict]] = {}
    all_nyquist_results: Dict[str, Optional[OptimalSimResultDict]] = {}
    all_fft_results: Dict[str, Optional[FFTResultDict]] = {}

    # Get specific output paths for each analysis type
    nyquist_path = output_paths.get('nyquist')
    optima_path = output_paths.get('optima')
    fourier_path = output_paths.get('fourier')

    # --- Iterate Through Signals ---
    for signal_data in signals:
        signal_name = signal_data.get('name')
        if not signal_name:
            logger.warning("Skipping signal entry with no 'name' key.")
            continue
        logger.info(f"\n>>> Processing Results for Signal: [{signal_name}] <<<")

        # Get parametric results specific to this signal
        results_for_this_signal: Optional[ParametricResultsDict] = {}
        if study_summary_data and signal_name in study_summary_data:
            results_for_this_signal = study_summary_data[signal_name]
        elif load_needed:
            logger.warning(f"No parametric summary data found or loaded for signal '{signal_name}'. Optimal analysis will fail.")


        # --- Workflow Execution ---
        optimal_params: Optional[OptimalParamsDict] = None
        optimal_sim_result: Optional[OptimalSimResultDict] = None
        nyquist_sim_result: Optional[OptimalSimResultDict] = None
        fft_result: Optional[FFTResultDict] = None

        # 1. Find Optimal Parameters (Required for all subsequent steps)
        # Check if *any* analysis is enabled, as they all depend on optima
        run_optima_search = bools.get('optima', False) or bools.get('nyquist', False) or bools.get('fourier', False)
        if run_optima_search:
            optimal_params, optimal_sim_result = optima.perform_optima_study(signal_data, results_for_this_signal, optima_path, bools, config)
            all_optimal_params[signal_name] = optimal_params
            all_optimal_sim_results[signal_name] = optimal_sim_result
        else:
            logger.info(f"Skipping Optima search for {signal_name} (no dependent analyses enabled).")

        # 2. Run Nyquist Analysis (if enabled and optimal sim succeeded)
        if bools.get('nyquist', False):
            if optimal_params and optimal_sim_result:
                nyquist_sim_result = nyquist.perform_nyquist_study(signal_data, optimal_params, optimal_sim_result, nyquist_path, bools, config)
                all_nyquist_results[signal_name] = nyquist_sim_result
            else:
                logger.warning(f"Skipping Nyquist analysis for {signal_name}: Required optimal simulation results not available (simulation likely failed or was skipped).")
        else:
            logger.info(f"Skipping Nyquist analysis for {signal_name} (disabled).")

        # 3. Run Fourier Analysis (if enabled and optimal sim succeeded)
        if bools.get('fourier', False):
            if optimal_params and optimal_sim_result and nyquist_sim_result:
                # The idea is to be able to run Fourier only even if nyquist fails, to just need optimal sim result. But i haven't developed this yet
                fft_result = fourier.perform_fourier_study(signal_data, optimal_sim_result, nyquist_sim_result, fourier_path, bools)
                all_fft_results[signal_name] = fft_result
            else:
                logger.warning(f"Skipping Fourier analysis for {signal_name}: Required optimal simulation results not available (simulation likely failed or was skipped).")
        else:
            logger.info(f"Skipping Fourier analysis for {signal_name} (disabled).")


    all_results = {
        "optimal_params": all_optimal_params,
        "optimal_sim_results": all_optimal_sim_results,
        "nyquist_results": all_nyquist_results,
        "fft_results": all_fft_results
    }

    logger.info("Completed processing all signals.")
    logger.info("====== Results Handler Finished ======")

    return all_results