import os
import pandas as pd
import numpy as np
import pickle
from scipy.fft import rfft, rfftfreq
from typing import Tuple, Any, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

# --- Constants for DataFrame Columns ---
COL_IN_LEN = "Input Length"
COL_N_SPIKES = "Encoded Length"
COL_FS = "Sampling Frequency (Hz)"
COL_D_NORM = "Normalized Threshold"
COL_B = "Bias" 
COL_DTE = "Encoder Resolution (dte)" 
COL_BW = "Decoder Bandwidth (rad/s)"
COL_TIME = "Elapsed Time (s)"
COL_STABLE_START_IDX = "Stable Start Index" 
COL_STABLE_END_IDX = "Stable End Index"   
COL_N_POINTS = "N_Points" 
COL_MAX_ERR = "Max Error"
COL_MED_ERR = "Median Error"
COL_MEAN_ERR = "Mean Error"
COL_R2 = "R^2"
COL_MSE = "Mean Squared Error"
COL_RMSE = "Root Mean Squared Error"
COL_NRMSE_STD = "Normalized RMSE (std)"
COL_MODE_ERR = "Mode Error"
COL_ERROR_MIDPOINT = "Error Midpoint"


def get_stable_region(time: np.ndarray, stable_percentage: float = 0.8) -> Tuple[int, int]:
    '''
    Calculates the indices defining the stable central region of a time vector.

    Inputs
    ------
    - time: np.ndarray
        The time vector (assumed to be monotonically increasing).
    - stable_percentage: float
        The percentage of the signal duration (centered) to consider stable.
        Defaults to 0.8 (80%).

    Outputs
    -------
    - Tuple[int, int]
        A tuple containing (start_index, end_index) of the stable region.
        Returns (0, len(time)) if calculation fails or time vector is too short,
        along with a printed warning.

    Raises
    ------
    - ValueError: If the time vector is not monotonically increasing or has invalid values.
    - Exception: For any other unexpected errors during calculation.
    '''
    n = len(time)
    if n < 2:
        logger.warning("(get_stable_region): Time vector too short (< 2 points). Returning full range.")
        return 0, n

    try:
        t_min, t_max = time[0], time[-1]
        total_duration = t_max - t_min
        if total_duration <= 0:
            logger.warning("(get_stable_region): Total duration is non-positive. Returning full range.") 
            return 0, n

        stable_duration = total_duration * stable_percentage
        margin = (total_duration - stable_duration) / 2.0
        start_time = t_min + margin
        end_time = t_max - margin

        # Find the indices closest to the calculated start and end times
        start_index = np.searchsorted(time, start_time, side='left')
        end_index = np.searchsorted(time, end_time, side='right')

        # Ensure indices are valid
        start_index = max(0, start_index)
        end_index = min(n, end_index)

        if start_index >= end_index:
            logger.warning(f"(get_stable_region): Calculated indices invalid ({start_index}, {end_index}). Returning full range.") 
            return 0, n

        # Ensure stable region has at least a minimum number of points (e.g., 2)
        if end_index - start_index < 2:
            logger.warning(f"(get_stable_region): Calculated stable region too small ({end_index - start_index} points). Returning full range.")
            return 0, n

        return start_index, end_index

    except Exception as e:
        logger.warning(f"(get_stable_region): Error calculating stable region: {e}. Returning full range.")
        return 0, n

def estimate_bias_range(c: float = 1.0):
    '''
    Estimates a reasonable range of encoder bias values 'b' to test for a given max signal amplitude 'c'.

    Inputs
    ------
    c : float
        Maximum absolute amplitude of the input signal (|x(t)|_max).

    Outputs
    -------
    bias_values: np.ndarray
        A NumPy array of bias values to test, or None if input parameters are invalid
        or result in an empty range.

    Raises
    ------
    - ValueError: If input parameters are invalid (e.g., start_factor <= 1.0, step <= 0).
    - TypeError: If c is not a valid number.
    '''
    # --- Hardcoded Defaults ---
    start_factor: float = 1.01
    end_value: float = 2.0
    step: float = 0.001

    # --- Input Validation ---
    if not isinstance(c, (int, float)) or c < 0:
        logger.error(f"Invalid input 'c' ({c}). Must be a non-negative number.")
        return None

    # --- Calculate Start and End ---
    b_start = start_factor * c

    # Check if start is already beyond the fixed end
    if b_start >= end_value:
        logger.warning(f"Calculated start bias ({b_start:.4g}) is >= fixed end value ({end_value:.4g}). No valid range possible.")
        return None
    try:
        bias_values = np.arange(b_start, end_value + step * 0.5, step)

        if bias_values.size == 0:
            logger.warning(f"Generated bias range is empty (start={b_start:.4g}, end={end_value:.4g}, step={step:.4g}).")
            return None

        logger.info(f"Estimated bias range: Start={bias_values[0]:.4g}, End={bias_values[-1]:.4g}, Step={step:.4g}, NumPoints={len(bias_values)}")
        return bias_values

    except Exception as e:
        logger.error(f"Unexpected error generating bias range: {e}", exc_info=True)
        return None


def store_df_to_excel(df: pd.DataFrame, output_dir: str, filename_no_ext: str) -> None:
    '''
    Stores a pandas DataFrame to an Excel file (.xlsx).

    Inputs
    ------
    - df: pd.DataFrame
        The DataFrame to store.
    - output_dir: str   
        The directory to save the Excel file in.
    - filename_no_ext: str
        The name of the Excel file without the .xlsx extension.
    
    Raises
    ------
    - ValueError: If the input DataFrame is not valid.
    - Exception: For any other unexpected errors during saving.
    '''
    if not isinstance(df, pd.DataFrame):
        logger.error(f"(store_df_to_excel): Expects a pandas DataFrame, got {type(df)}.")
        return
    filepath = os.path.join(output_dir, f"{filename_no_ext}.xlsx")
    try:
        os.makedirs(output_dir, exist_ok=True)
        # Added engine for potentially better compatibility/performance
        df.to_excel(filepath, index=False, engine='openpyxl')
    except ImportError: # Handle case where openpyxl is not installed
        try:
            logger.warning("(store_df_to_excel): openpyxl not found. Trying default engine.")
            df.to_excel(filepath, index=False)
        except Exception as e_fallback:
            logger.error(f'(store_df_to_excel) storing DataFrame {filename_no_ext} to Excel (fallback): {e_fallback}')
    except Exception as e:
        logger.error(f'Error storing DataFrame {filename_no_ext} to Excel: {e}', exc_info=True)

def load_df_from_excel(filepath: str) -> Optional[pd.DataFrame]:
    '''
    Loads data from an Excel file into a pandas DataFrame.
    Does NOT perform column validation or type conversion - caller must handle.

    Inputs
    ------
    - filepath: str
        Full path to the Excel file (.xlsx or .xls).
    
    Outputs
    -------
    - pd.DataFrame or None
        The loaded DataFrame, or None if loading fails.
    
    Raises
    ------
    - FileNotFoundError: If the file does not exist.
    - Exception: For any other unexpected errors during loading.
    '''
    if not os.path.exists(filepath):
        return None
    try:
        # Added engine for potentially better compatibility/performance
        df = pd.read_excel(filepath, engine='openpyxl')
        return df
    except ImportError:
        try:
            logger.warning(f"(load_df_from_excel): openpyxl not found for {os.path.basename(filepath)}. Trying default engine.")
            df = pd.read_excel(filepath)
            return df
        except Exception as e_fallback:
            logger.error(f'(load_df_from_excel) loading {os.path.basename(filepath)} (fallback): {e_fallback}')
            return None
    except Exception as e:
        logger.error(f'(load_df_from_excel) loading {os.path.basename(filepath)}: {e}')
        return None


def save_pickle(data: Any, output_dir: str, filename_no_ext: str) -> None:
    '''
    Saves data to a pickle file (.pkl).

    Inputs
    ------
    - data: Any
        The Python object to save.
    - output_dir: str
        The directory to save the pickle file in.
    - filename_no_ext: str
        The name of the pickle file without the .pkl extension.

    Raises
    ------
    - ValueError: If the data is None or invalid.
    - Exception: For any other unexpected errors during saving.
    '''
    filepath = os.path.join(output_dir, f"{filename_no_ext}.pkl")
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        logger.error(f"(save_pickle) saving data to {filename_no_ext}.pkl: {e}")


def load_pickle(filepath: str) -> Optional[Any]:
    '''
    Loads data from a pickle file.

    Inputs
    ------
    - filepath: str
        The full path to the pickle file.
    
    Outputs
    -------
    - Any or None
        The data loaded from the pickle file, or None if loading fails.
    
    Raises
    ------
    - FileNotFoundError: If the file does not exist.
    - ModuleNotFoundError: If a required module is not found during unpickling. 
    - pickle.UnpicklingError: If the file is corrupted or incompatible.
    - Exception: For any other unexpected errors during loading.
    '''
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            return data
    except ModuleNotFoundError as mnf_err:
        logger.error(f"(load_pickle) {os.path.basename(filepath)}: Module not found ({mnf_err}). Check dependencies.")
        return None
    except pickle.UnpicklingError as up_err:
        logger.error(f"(load_pickle) {os.path.basename(filepath)}: Unpickling error ({up_err}). File corrupted/incompatible?")
        return None
    except Exception as e:
        logger.error(f"(load_pickle) {os.path.basename(filepath)}: {e}")
        return None


def fft(signal: np.ndarray, sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Computes the real FFT of a signal.

    Inputs
    ------
    - signal: np.ndarray
        The input signal array (real-valued).
    - sampling_rate: float
        The sampling rate of the signal (Hz).
    
    Outputs
    ------- 
    - Tuple[np.ndarray, np.ndarray]
        A tuple containing:
            - frequencies: np.ndarray of frequencies (Hz) for the positive spectrum.
            - complex_spectrum: np.ndarray of complex FFT coefficients corresponding
                               to the positive frequencies.
                               Magnitude can be obtained via np.abs(),
                               Phase via np.angle().

    Raises
    ------
    - ValueError: If the sampling rate is non-positive.
    - Exception: For any other unexpected errors during FFT computation.
    '''
    if sampling_rate <= 0:
        logger.error("Sampling rate must be positive for FFT.")
        raise ValueError("Sampling rate must be positive.")
    if len(signal) == 0:
        return np.array([]), np.array([])

    N = len(signal)
    complex_spectrum = rfft(signal)
    frequencies = rfftfreq(N, 1.0 / sampling_rate)
    return frequencies, complex_spectrum

def estimate_decoder_bandwidth(freq_max: float) -> float:
    '''
    Estimate decoder bandwidth to be slightly larger than twice the maximum frequency of the signal.
    '''
    bw = 2.5 * np.pi * freq_max # Bandwidth (rad/s)

    return bw