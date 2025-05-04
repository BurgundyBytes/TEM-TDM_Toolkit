import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple 
import logging

logger = logging.getLogger(__name__)

# custom modules
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, src_path)
from src.parametric import studies
from src.utilities import utils
from src.utilities import plotting

# --- Data Types ---
ConfigDict = Dict[str, Any]
BoolsDict = Dict[str, bool]
OutputPathsDict = Dict[str, Optional[str]]
SignalDict = Dict[str, Any]
SignalResultsDict = Dict[str, Optional[pd.DataFrame]]
ParametricResultsDict = Dict[str, SignalResultsDict]



def run_studies(config: ConfigDict, signals: List[SignalDict], bools: BoolsDict, output_paths: OutputPathsDict) -> ParametricResultsDict:
    '''
    Run the configured parametric studies for each input signal.
    Calculates stable region indices once per signal and passes them down.

    Each study type (bias, delta, biparametric) is run if the corresponding output path is provided.
    The results are aggregated in a dictionary, which is returned.

    Inputs
    ------
    - config: dict
        Dictionary containing configuration settings for parametric studies.
    - signals: list of dicts
        List of input signals, each represented as a dictionary with relevant parameters.
    - bools: dict
        Dictionary of boolean flags to control execution, logging, and output options.
    - output_paths: dict
        Dictionary containing paths for output directories for different study types.
    
    Outputs
    -------
    - all_results: dict or None
        Dictionary containing the results of the executed parametric studies, or None if no studies were run.
        Each key corresponds to a signal name, and the value is another dictionary with study types as keys
        and DataFrames as values.

    Raises
    ------
    - KeyError: If a required key is missing in the configuration or output paths.
    - ValueError: If the input signals are not provided or are invalid.
    '''
    all_results: ParametricResultsDict = {}

    run_bias_study = output_paths.get('param_bias') is not None
    run_delta_study = output_paths.get('param_delta') is not None
    run_biparam_study = output_paths.get('biparametric') is not None

    if not (run_bias_study or run_delta_study or run_biparam_study):
        logger.warning("No parametric studies configured (no output folders enabled/created).")
        return None

    for signal_data in signals:
        signal_name = signal_data.get('name', 'unnamed_signal')
        logger.info(f"\nProcessing signal: {signal_name}...")
        t = signal_data.get('t')
        u = signal_data.get('u')
        dur = signal_data.get('dur')
        c = signal_data.get('c')
        dte = signal_data.get('dte', 0.0)
        freq_max = signal_data.get('freq_max', 0.0)

        if t is None or u is None or dur is None:
            logger.warning(f"Skipping signal '{signal_name}' due to missing t, u, or dur.")
            continue

        # --- Calculate Stable Region Indices ONCE per signal ---
        try:
            # Get percentage from config, default to 0.9 (90%)
            stable_percentage = config.get('Stable Region Percentage', 0.9)
            logger.debug(f"Using stable region percentage: {stable_percentage:.2f}")
            start_idx, end_idx = utils.get_stable_region(t, stable_percentage)
            bias = utils.estimate_bias_range(c)

            # Check if get_stable_region returned fallback (0, len(t)) due to error/short signal
            if end_idx == len(t) and start_idx == 0 and len(t)>1: # Avoid trigger on len=1 case
                logger.info(f"  Stable region covers full signal [{start_idx}:{end_idx}].")
            else:
                logger.info(f"  Using stable region indices [{start_idx}:{end_idx}] for metrics.")

        except Exception as e:
            # Catch unexpected errors during calculation itself
            logger.error(f"Unexpected error getting stable region for signal '{signal_name}': {e}. Using full signal.", exc_info=True)
            start_idx, end_idx = 0, len(t) # Fallback to full signal

        current_signal_results: SignalResultsDict = {}

        # --- Call specific study runners, passing indices ---
        if run_bias_study:
            logger.info(f"\tRunning Bias study for {signal_name}...")
            try:
                df_bias = run_bias(config, u, t, dur, bias, dte, freq_max,
                                        start_idx, end_idx,
                                        bools, signal_name, output_paths['param_bias'])
                if df_bias is not None and not df_bias.empty:
                    current_signal_results["bias"] = df_bias
                    logger.info(f"\tBias study for {signal_name} completed.")

                    if bools.get('plots', False):
                        filename = f"param_bias_summary_{signal_name}.png"
                        file_path = os.path.join(output_paths['param_bias'], filename)
                        plotting.plot_parametric(df_bias, 'PB', "Bias Study Summary", file_path)
                else:
                    logger.warning(f"\tBias study for {signal_name} did not return valid results.")
            except Exception as e:
                logger.error(f"\tUnhandled exception during Bias study run for {signal_name}", exc_info=True)


        if run_delta_study:
            logger.info(f"\tRunning Delta study for {signal_name}...")
            try:
                b = config.get('Default Bias')
                df_delta = run_delta(config, u, t, dur, b, dte, freq_max,
                                    start_idx, end_idx, 
                                    bools, signal_name, output_paths['param_delta'])
                if df_delta is not None and not df_delta.empty:
                    current_signal_results["delta"] = df_delta
                    logger.info(f"\tDelta study for {signal_name} completed.")

                    if bools.get('plots', False):
                        filename = f"param_delta_summary_{signal_name}.png"
                        file_path = os.path.join(output_paths['param_delta'], filename)
                        plotting.plot_parametric(df_delta, 'PD', "Delta Study Summary", file_path)
                else:
                    logger.warning(f"\tDelta study for {signal_name} did not return valid results.")
            except Exception as e:
                 logger.error(f"\tUnhandled exception during Delta study run for {signal_name}", exc_info=True)


        if run_biparam_study:
            logger.info(f"\tRunning Biparametric study for {signal_name}...")
            try:
                df_2d = run_biparametric(config, u, t, dur, bias, dte, freq_max,
                                        start_idx, end_idx, 
                                        bools, signal_name, output_paths['biparametric'])
                if df_2d is not None and not df_2d.empty:
                    current_signal_results["bias_delta"] = df_2d
                    logger.info(f"\tBiparametric study for {signal_name} completed.")

                    if bools.get('plots', False):
                        filename = f"param_biasDelta_summary_{signal_name}.png"
                        file_path = os.path.join(output_paths['biparametric'], filename)
                        plotting.plot_biparametric(df_2d, "Biparametric Study Summary", file_path)
                else:
                    logger.warning(f"\tBiparametric study for {signal_name} did not return valid results.")
            except Exception as e:
                logger.error(f"\tUnhandled exception during Biparametric study run for {signal_name}", exc_info=True)


        if current_signal_results:
            all_results[signal_name] = current_signal_results
        else:
            # This logs if none of the studies for *this signal* yielded results
            logger.warning(f"No valid parametric study results generated for signal: {signal_name}")

    return all_results if all_results else None


# Runner functions for each study type
def run_delta(config: ConfigDict, u: np.ndarray, t: np.ndarray, dur: float, b: float, dte: float, freq_max: float, start_idx: int, end_idx: int, bools: BoolsDict, signal_name: str, output_dir: Optional[str]) -> Optional[pd.DataFrame]:
    '''
    Run the normalized threshold parametric study for a given signal.
    This function sets up the parameters for the delta study, including the delta range and default encoder bias.
    It then calls the `parametric_delta` function from the `studies` module to perform the actual computation.
    '''
    if output_dir is None:
        logger.debug(f"Output directory for Delta study ({signal_name}) is None. Skipping run.")
        return None

    # Get delta range
    delta_range_tuple = config.get('Delta Range')
    if delta_range_tuple is None:
        logger.error(f"'Delta Range' not found or invalid in config for signal {signal_name}.")
        return None
    try:
        start_delta, end_delta, step_delta = delta_range_tuple
        deltas = np.arange(start_delta, end_delta + step_delta * 0.5, step_delta)
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid delta range format during study setup: {delta_range_tuple}. {e}", exc_info=True)
        return None

    # Get default frequency
    fs = config.get('Default Frequency')
    if fs is None:
        logger.error(f"'Default Frequency' not found in config for signal {signal_name}.")
        return None

    log_filename = f'parametric_delta-{signal_name}.txt'

    logger.info(f"\t\tParams: Delta Range={delta_range_tuple}, Default Bias={b:.4f} Hz")
    logger.info(f"\t\tSaving outputs to: {output_dir}")

    try:
        df_delta = studies.parametric_delta(
            u, t, dur, fs, deltas, b, dte, freq_max,
            start_idx, end_idx, 
            log_filename, output_dir, signal_name,
            bools.get('logs', False), bools.get('plots', False), bools.get('pickle', False)
        )
        return df_delta
    except Exception as e:
        logger.error(f"ERROR calling studies.parametric_delta for {signal_name}: {e}", exc_info=True)
        return None


def run_bias(config: ConfigDict, u: np.ndarray, t: np.ndarray, dur: float, bias: float, dte: float, freq_max: float, start_idx: int, end_idx: int, bools: BoolsDict, signal_name: str, output_dir: Optional[str]) -> Optional[pd.DataFrame]:
    '''
    Run the encoder bias parametric study for a given signal.
    This function sets up the parameters for the bias study, including the bias range and default normalized threshold.
    It then calls the `parametric_bias` function from the `studies` module to perform the actual computation.
    '''
    if output_dir is None:
        logger.debug(f"Output directory for Bias study ({signal_name}) is None. Skipping run.")
        return None

    # Get default frequency
    fs = config.get('Default Frequency')
    if fs is None:
        logger.error(f"'Default Frequency' not found in config for signal {signal_name}.")
        return None

    # Get default normalized threshold
    d_norm = config.get('Default Delta')
    if fs is None:
        logger.error(f"'Default Delta' not found in config for signal {signal_name}.")
        return None
    
    log_filename = f'parametric_bias-{signal_name}.txt'

    logger.info(f"\t\tParams:   Bias Range=({bias[0]:.4f}, {(bias[1]-bias[0]):.4f}, {bias[-1]:.4f}),  Default d_norm={d_norm:.4f} Hz")
    logger.info(f"\t\tSaving outputs to: {output_dir}")

    try:
        df_bias = studies.parametric_bias(
            u, t, dur, fs, d_norm, bias, dte, freq_max,
            start_idx, end_idx, 
            log_filename, output_dir, signal_name,
            bools.get('logs', False), bools.get('plots', False), bools.get('pickle', False)
        )
        return df_bias
    except Exception as e:
        logger.error(f"ERROR calling studies.parametric_bias for {signal_name}: {e}", exc_info=True)
        return None


def run_biparametric(config: ConfigDict, u: np.ndarray, t: np.ndarray, dur: float, bias: float, dte: float, freq_max: float, start_idx: int, end_idx: int, bools: BoolsDict, signal_name: str, output_dir: Optional[str]) -> Optional[pd.DataFrame]:
    '''
    Run the biparametric study for a given signal.
    This function sets up the parameters for the biparametric study, including the bias and delta ranges.
    It then calls the `parametric_bias_delta` function from the `studies` module to perform the actual computation.
    '''
    if output_dir is None:
        logger.debug(f"Output directory for Biparametric study ({signal_name}) is None. Skipping run.")
        return None

    # --- Delta Range ---
    delta_range_tuple = config.get('Delta Range')
    if delta_range_tuple is None:
        logger.error(f"'Delta Range' not found or invalid for biparametric ({signal_name}).")
        return None
    try:
        start_delta, end_delta, step_delta = delta_range_tuple
        deltas = np.arange(start_delta, end_delta + step_delta * 0.5, step_delta)
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid delta range for biparametric: {delta_range_tuple}. {e}", exc_info=True)
        return None

    # Get default frequency
    fs = config.get('Default Frequency')
    if fs is None:
        logger.error(f"'Default Frequency' not found in config for signal {signal_name}.")
        return None
    
    log_filename = f'parametric_biasDelta-{signal_name}.txt'

    logger.info(f"\t\tBias range: {bias[0]:.4f} to {bias[-1]:.4f}")
    logger.info(f"\t\tDelta range: {deltas[0]:.4f} to {deltas[-1]:.4f}")
    logger.info(f"\t\tSaving outputs to: {output_dir}")

    try:
        # Pass indices to studies.parametric_bias_delta
        df_2d = studies.parametric_bias_delta(
            u, t, dur, fs, deltas, bias, dte, freq_max,
            start_idx, end_idx, # Pass indices
            log_filename, output_dir, signal_name,
            bools.get('logs', False), bools.get('plots', False), bools.get('pickle', False)
        )
        return df_2d
    except Exception as e:
        logger.error(f"ERROR calling studies.parametric_bias_delta for {signal_name}: {e}", exc_info=True)
        return None


def save_results(summary_results: ParametricResultsDict, bools: BoolsDict, output_paths: OutputPathsDict) -> None:
    ''''
    Save the results of the parametric studies to specified output paths.
    This function handles the saving of summary results for each signal and study type.
    '''
    if not summary_results:
        logger.info("No parametric summary results provided to save.")
        return
    logger.info("--- Saving Parametric Study Summary Results ---")
    study_map = { # Mapping internal study key to (output_path_key, filename_base)
        "bias":       ('param_bias', 'parametric_bias_summary'),
        "delta":      ('param_delta', 'parametric_delta_summary'),
        "bias_delta": ('biparametric', 'parametric_biasDelta_summary')
    }

    for signal_name, study_dataframes in summary_results.items():
        logger.info(f"\tSaving summary results for signal: {signal_name}")
        for study_type, df_summary in study_dataframes.items():
            if df_summary is None or df_summary.empty:
                logger.debug(f"\t\tSkipping summary for '{study_type}' (DataFrame is None or empty).")
                continue

            if study_type in study_map:
                path_key, base_filename = study_map[study_type]
                target_folder = output_paths.get(path_key)

                if target_folder:
                    # Excel saving
                    try:
                        excel_filename_no_ext = f"{base_filename}_{signal_name}"
                        logger.info(f"\t\tSaving SUMMARY '{study_type}' metrics to Excel: {excel_filename_no_ext}.xlsx")
                        utils.store_df_to_excel(df_summary, target_folder, excel_filename_no_ext)
                    except Exception as e:
                        logger.error(f"\t\tERROR saving summary Excel for {study_type} of {signal_name}: {e}", exc_info=True)

                    # # Pickle saving (optional)
                    # if bools.get('pickle', False):
                    #     try:
                    #         pickle_filename_no_ext = f"{base_filename}_{signal_name}"
                    #         logger.info(f"\t\tSaving SUMMARY '{study_type}' metrics to Pickle: {pickle_filename_no_ext}.pkl")
                    #         utils.save_pickle(df_summary, target_folder, pickle_filename_no_ext)
                    #     except Exception as e:
                    #         logger.error(f"\t\tERROR saving summary Pickle for {study_type} of {signal_name}: {e}", exc_info=True)
                else:
                     logger.warning(f"\t\tOutput path for study type '{study_type}' (key: '{path_key}') not found. Cannot save summary for {signal_name}.")
            else:
                logger.warning(f"\t\tUnknown study type key '{study_type}' in results for {signal_name}. Cannot save summary.")
    logger.info("--- Summary Results Saving Complete ---")


def manager(config: ConfigDict, bools: BoolsDict, signals: List[SignalDict], output_paths: OutputPathsDict) -> ParametricResultsDict:
    '''
    Main function to manage the execution ('execute' workflow) of parametric studies.
    Orchestrates running studies and saving the aggregated results.

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
    
    Outputs
    -------
    - results: dict or None
        Dictionary containing the results of the executed parametric studies, or None if no studies were run.
        Each key corresponds to a signal name, and the value is another dictionary with study types as keys
        and DataFrames as values.
    
    Raises
    ------
    - KeyError: If a required key is missing in the configuration or output paths.
    - ValueError: If the input signals are not provided or are invalid.
    '''
    logger.info("=== Parametric Handler Manager ===")
    if not signals:
        logger.warning("Parametric handler: No input signals provided. Exiting.")
        return None

    # run_studies now calculates indices and calls specific runners
    results: ParametricResultsDict = {}
    results = run_studies(config, signals, bools, output_paths)

    # save_results handles the aggregated DataFrames returned by run_studies
    if results:
        save_results(results, bools, output_paths)
    else:
        logger.warning("Parametric handler: No parametric studies were run or none returned results.")

    logger.info("=== Parametric Handler Finished ===\n")
    return results