import pandas as pd 
import os
import sys
import numpy as np 
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# custom modules
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, src_path)
from src.utilities import utils

# --- Helper Functions ---
def initialize_log_file(filepath: str, header_lines: List[str]) -> None:
    '''
    Initializes a log file by writing header lines (overwrites existing file).
    '''
    try:
        output_dir = os.path.dirname(filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(filepath, "w") as f:
            f.write(f"# Log Initialized: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("#" + "-"*70) 
            for line in header_lines:
                f.write(f"# {line}") 
            f.write("#" + "-"*70 + "\n")
    except Exception as e:
        logger.error(f"Error initializing log file {filepath}: {e}", exc_info=True)

# --- Public Logging Functions ---
def log_study_header(filepath: str, study_type: str, param_ranges: Union[np.ndarray, Dict[str, np.ndarray]], fixed_params: Dict[str, Any]) -> None:
    '''Writes a header section for a parametric study log file.'''
    header = [
        f"\n--- Parametric Study Log ---",
        f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\nStudy Type: {study_type}",
        f"\nFixed Parameters:",
    ]
    for key, value in fixed_params.items():
        # Use utils constants if key matches for consistency in log
        if key == utils.COL_FS: header.append(f"\t{utils.COL_FS}: {value}")
        elif key == utils.COL_D_NORM: header.append(f"\t{utils.COL_D_NORM}: {value}")
        elif key == utils.COL_B: header.append(f"\t{utils.COL_B}: {value}")
        elif key == utils.COL_DTE: header.append(f"\t{utils.COL_DTE}: {value}")
        else: header.append(f"\t{key}: {value}") # Fallback for other keys

    header.append("\nVaried Parameters:")
    if isinstance(param_ranges, dict): # For biparametric
        for key, values in param_ranges.items():
            param_label = key # Default
            if key == 'freqs': param_label = utils.COL_FS
            elif key == 'deltas': param_label = utils.COL_D_NORM
            if len(values) > 0:
                 header.append(f"\t{param_label}: Min={np.min(values)}, Max={np.max(values)}, N={len(values)}")
            else:
                 header.append(f"\t{param_label}: (Empty Range)")
    elif isinstance(param_ranges, np.ndarray): # For 1D studies
        param_label = "Values" # Default label if not identifiable
        # Try to identify based on fixed params (less robust)
        if utils.COL_FS not in fixed_params: param_label = utils.COL_FS
        elif utils.COL_D_NORM not in fixed_params: param_label = utils.COL_D_NORM

        if len(param_ranges) > 0:
            header.append(f"\t{param_label}: Min={np.min(param_ranges)}, Max={np.max(param_ranges)}, N={len(param_ranges)}")
        else:
            header.append(f"\t{param_label}: (Empty Range)")
    else:
         header.append(f"\tRange Info: {param_ranges}") # Fallback

    header.append("-" * 70 + "\n") # Use same length as separator above
    initialize_log_file(filepath, header) # Use initialize to overwrite/create


def log_parametric_run(filepath: str, metrics_dict: Dict[str, Any]) -> None:
    '''
    Appends formatted metrics for a single parametric run to a text log file.
    Assumes parameters like fs, d_norm are already included in metrics_dict.
    '''
    if not filepath:
        logger.warning("No filepath provided for run logging.")
        return
    if not metrics_dict:
        logger.warning("Empty metrics_dict provided for run logging.")
        return

    log_lines = []
    log_lines.append("--- Run Start ---")

    # Log Parameters (using utils constants)
    params_keys = [utils.COL_FS, utils.COL_D_NORM, utils.COL_B, utils.COL_DTE]
    logged_params = False
    log_lines.append("\tParameters:")
    for key in params_keys:
        if key in metrics_dict:
            value = metrics_dict[key]
            # Basic formatting, adjust precision as needed
            if isinstance(value, (float, np.floating)):
                 log_lines.append(f"\t\t{key:<25}: {value:.4g}") # Use general format for params
            else:
                 log_lines.append(f"\t\t{key:<25}: {value}")
            logged_params = True
    if not logged_params: log_lines.pop() # Remove header if no params found


    # Log Metrics (using utils constants where possible)
    log_lines.append("\tMetrics:")
    # Define order, using constants from utils and specific keys from metrics.py
    metric_keys_ordered = [
        'input_length_stable', # Key from metrics.py
        utils.COL_N_SPIKES,
        utils.COL_TIME,
        utils.COL_MAX_ERR,
        utils.COL_MED_ERR,
        utils.COL_MEAN_ERR,
        utils.COL_MSE,
        utils.COL_RMSE,
        utils.COL_NRMSE_STD,
        utils.COL_R2,
        utils.COL_MODE_ERR,
        utils.COL_ERROR_MIDPOINT,
    ]

    # Add stable indices if present
    if utils.COL_STABLE_START_IDX in metrics_dict:
        metric_keys_ordered.append(utils.COL_STABLE_START_IDX)
    if utils.COL_STABLE_END_IDX in metrics_dict:
        metric_keys_ordered.append(utils.COL_STABLE_END_IDX)


    for key in metric_keys_ordered:
         value = metrics_dict.get(key) # Use .get() for safety
         if value is not None:
             # Format based on metric type/key
             if isinstance(value, (int, np.integer)):
                 log_lines.append(f"\t\t{key:<25}: {value}") # Integer formatting
             elif isinstance(value, (float, np.floating)):
                 # More specific formatting based on constant/key
                 if key in [utils.COL_MAX_ERR, utils.COL_MED_ERR, utils.COL_MEAN_ERR, utils.COL_MSE, utils.COL_RMSE, utils.COL_MODE_ERR, utils.COL_ERROR_MIDPOINT]:
                     log_lines.append(f"\t\t{key:<25}: {value:.6e}") # Scientific notation for errors
                 elif key == utils.COL_NRMSE_STD:
                      log_lines.append(f"\t\t{key:<25}: {value:.6f}") # NRMSE as standard float
                 elif key == utils.COL_R2:
                      log_lines.append(f"\t\t{key:<25}: {value:.6f}") # R2 as standard float
                 elif key == utils.COL_TIME:
                      log_lines.append(f"\t\t{key:<25}: {value:.4f} s") # Time
                 else:
                      log_lines.append(f"\t\t{key:<25}: {value:.4g}") # General float formatting
             else: # Handle other types (e.g., strings if any)
                  log_lines.append(f"\t\t{key:<25}: {value}")

    log_lines.append("--- Run End ---") # Removed extra newline here, added in loop below

    # Append to file
    try:
        output_dir = os.path.dirname(filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        # Append with newline after each line, including the final "Run End"
        with open(filepath, "a") as f:
            for line in log_lines:
                f.write(line + "\n")
    except Exception as e:
        logger.error(f"Error appending to run log file {filepath}: {e}", exc_info=True)