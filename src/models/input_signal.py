import numpy as np
import os
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional 
import logging

logger = logging.getLogger(__name__)

# --- Data Types ---
SignalDict = Dict[str, Any]

def generate_sum_of_sines(frequencies: List[float], amplitudes: List[float], duration: float, sampling_rate: float) -> Tuple[np.ndarray, np.ndarray, float]:
    '''
    Generates a composite signal consisting of the sum of multiple sine waves.

    Inputs
    -------
    - frequencies: list
        Frequencies for each sine wave [Hz].
    - amplitudes: list
        Amplitudes for each sine wave.
    - duration: float
        Duration of the signal [s].
    - sampling_rate: float
        Sampling rate [Hz].
    
    Outputs
    -------
    - time: np.ndarray
        Time vector.
    - signal: np.ndarray
        Composite signal.
    - max_freq_hz: float
        Maximum frequency in the signal [Hz].

    Raises
    ------
    - ValueError: If the lengths of frequencies and amplitudes do not match.
    - ValueError: If any frequency is negative.
    '''
    if len(frequencies) != len(amplitudes):
        raise ValueError("Length of frequencies and amplitudes must match.")
    time = np.arange(0, duration, 1/sampling_rate)
    signal = np.zeros_like(time)
    for freq, amp in zip(frequencies, amplitudes):
        if freq < 0: raise ValueError("Frequencies must be non-negative.") 
        signal += amp * np.sin(2 * np.pi * freq * time)
    max_freq_hz = max(frequencies) if frequencies else 0.0
    logger.debug(f"Generated sum_of_sines with max frequency: {max_freq_hz:.2f} Hz")

    return time, signal, max_freq_hz 

def generate_multifreq_signal(frequencies: List[float] = [11, 3, 11], amplitudes: List[float] = [1, 1, 1], duration: float = 5, sampling_rate: float = 1000) ->  Tuple[np.ndarray, np.ndarray, float]:
    '''
    Generates a composite signal with 5 sections: Constant - Freq 1 - Freq 2 - Constant - Freq 3.
    Uses the first three frequencies and amplitudes provided.

    Inputs
    -------
    - frequencies: list
        Frequencies for the sine wave sections [Hz]. Needs at least 3 values.
    - amplitudes: list
        Amplitudes for the sine wave sections. Needs at least 3 values.
    - duration: float
        Duration of the signal [s].
    - sampling_rate: float
        Sampling rate [Hz].
    
    Outputs
    -------
    - time: np.ndarray 
        Time vector.
    - signal: np.ndarray
        Composite signal.
    - max_freq_hz: float
        Maximum frequency in the signal [Hz].

    Raises
    ------
    - ValueError: If fewer than 3 frequencies or amplitudes are provided.
    '''
    if len(frequencies) < 3 or len(amplitudes) < 3:
        raise ValueError("Multi-frequency signal requires at least 3 frequencies and amplitudes.")

    time = np.arange(0, duration, 1/sampling_rate)
    signal = np.zeros_like(time)
    n_points = len(time)

    # Define the boundaries of each section (as indices)
    fractions = np.round(np.linspace(0, 1, 6) * n_points).astype(int)    
    section1_end = fractions[1] 
    section2_end = fractions[2]
    section3_end = fractions[3]
    section4_end = fractions[4]
    section5_end = fractions[5]

    # Section 1: Constant
    constant_value = 0.5 # TODO: We could make this configurable
    logger.debug(f"Generating multifreq signal with constant value: {constant_value}")
    signal[:section1_end] = constant_value

    # Section 2: First frequency sine wave
    signal[section1_end:section2_end] = amplitudes[0] * np.sin(2 * np.pi * frequencies[0] * time[section1_end:section2_end])

    # Section 3: Second frequency sine wave
    signal[section2_end:section3_end] = amplitudes[1] * np.sin(2 * np.pi * frequencies[1] * time[section2_end:section3_end])

    # Section 4: Constant
    signal[section3_end:section4_end] = constant_value

    # Section 5: Third frequency sine wave
    signal[section4_end:] = amplitudes[2] * np.sin(2 * np.pi * frequencies[2] * time[section4_end:])

    max_freq_hz = max(frequencies)
    logger.debug(f"Generated multifreq signal with max frequency: {max_freq_hz:.2f} Hz")

    return time, signal, max_freq_hz 

def generate_paper_signal(sampling_rate: float) ->  Tuple[np.ndarray, np.ndarray, float]:
    '''
    Generates the specific signal used in the example from the paper
    'Perfect Recovery and Sensitivity Analysis of Time-Encoded Band-limited signals'.

    Inputs
    -------
    - sampling_rate: float
        Sampling rate [Hz] for the output signal representation.
    
    Outputs
    -------
    - time: np.ndarray
        Time vector.
    - signal: np.ndarray
        Input signal from the paper's example.
    - fs: float
        Sampling frequency [Hz], assumed to be the max frequency in the signal >>>> THIS IS A GUESS.

    Raises
    ------
    - ValueError: If sampling_rate is non-positive.
    '''
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive.")

    # --- Parameters from the paper (or interpretation thereof) ---
    # TODO: Verify these parameters and the sinc function usage against the paper.
    # The paper states Bandwidth B = 20kHz. Standard Nyquist sampling period T = 1/(2B).
    # However, they seem to use frequency = 40kHz? And T = 1/f.
    fs = 40000  # Hz - Corresponds to B=20kHz if T = 1/(2*B)? Or is this 2*B?
    T = 1 / fs # Sampling period corresponding to the samples below

    samples = np.array([
        -0.1961, 0.186965, 0.207271, 0.0987736, -0.275572, 0.0201665,
        0.290247, 0.138374, -0.067588, -0.145661, -0.11133, -0.291498
    ])
    num_samples = len(samples)

    # --- Time vector for output signal ---
    # Paper evaluates from -2T to 15T.
    t_min = -2 * T
    t_max = 15 * T

    # Check if requested sampling rate is adequate relative to the paper's presumed period T
    if 1/fs >= T / 2: # Nyquist criterion
        logger.warning(f"Warning: Requested sampling rate ({fs} Hz) might be low "
              f"compared to the paper's signal characteristics (T={fs:.2e} s). "
              f"Ensure it's high enough for desired accuracy.")

    time = np.arange(t_min, t_max, 1/fs)

    # --- Sinc function definition used in this implementation ---
    g = lambda t_val: np.sinc(2 * fs * t_val)  

    # --- Generate the signal ---
    signal = np.zeros_like(time)
    for k in range(num_samples):
        kT = k * T  
        signal += samples[k] * g(time - kT)

    return time, signal, fs


def load_experiment_signals(file_path: str, file_name: str, config: Dict[str, Any]) -> List[SignalDict]:
    '''
    Loads signals from a .csv or .xlsx file based on column names.

    Inputs
    -------
    - file_path: str
        Path to the .csv or .xlsx file.
    - file_name: str
        Name of the file (used for naming the signals).
    - config: dict
        The main configuration dictionary (used for fs, b, dte).
    
    Outputs
    -------
    - signals: list
        A list of dictionaries, where each dictionary represents a signal.
        Returns an empty list if the file is not found, unsupported, or has read errors.

    Raises
    ------
    - FileNotFoundError: If the file does not exist.
    - pd.errors.EmptyDataError: If the file is empty.
    - pd.errors.ParserError: If the file cannot be parsed.
    - Exception: For any other unexpected errors.
    '''
    # Extract parameters from config 
    fs = config.get('Experiment Sampling Rate')
    b = config.get('Encoder Bias')
    dte = config.get('Encoder Resolution') 

    # Validate required parameters
    if fs is None:
        logger.error(f"'Experiment Sampling Rate' not found in config for file {file_name}.")
        return []
    if b is None:
        logger.warning(f"'Encoder Bias' not found in config, using default 0 for {file_name}.")
        b = 0.0
    if dte is None:
        logger.warning(f"'Encoder Resolution' not found in config, using default 0 for {file_name}.")
        dte = 0.0

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []

    signals: List[SignalDict] = []
    data: Optional[pd.DataFrame] = None

    try:
        if file_path.lower().endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.lower().endswith('.xlsx'):
            data = pd.read_excel(file_path)
        else:
            logger.error(f"Unsupported file type: {file_path}. Only .csv and .xlsx are supported.")
            return []

        if data is None or data.empty:
            logger.warning(f"Warning: No data loaded from file: {file_path}")
            return []

        # Define search terms (case-insensitive)
        pressure_term = 'pres'
        temp_term = 'temp'

        # Extract signals based on column names
        for col_name in data.columns:
            signal_type = None
            if pressure_term in col_name.lower():
                signal_type = 'pressure'
            elif temp_term in col_name.lower():
                signal_type = 'temperature'

            if signal_type:
                u = data[col_name].to_numpy()
                if pd.api.types.is_numeric_dtype(u): # Check if data is numeric
                    # Generate time vector
                    num_points = len(u)
                    duration_est = num_points / fs # Estimated duration
                    t = np.linspace(0, duration_est, num_points, endpoint=False)

                    # Create unique signal name
                    signal_name = f"{file_name}_{col_name.replace(' ', '_')}" # Use column name for uniqueness

                    signals.append({
                        't': t,
                        'u': u,
                        'name': signal_name,
                        'fs': fs,
                        'dur': duration_est,
                        'b': b,
                        'dte': dte,
                        'source_file': file_path,
                        'source_column': col_name
                    })
                else:
                    logger.warning(f"Column '{col_name}' in {file_path} is not numeric. Skipping.")


    except FileNotFoundError: # Should be caught by os.path.exists, but just in case
        logger.error(f"File not found during read: {file_path}", exc_info=True)
        return []
    except pd.errors.EmptyDataError:
        logger.error(f"File is empty: {file_path}", exc_info=True)
        return []
    except pd.errors.ParserError:
        logger.error(f"Could not parse file: {file_path}. Check format.", exc_info=True)
        return []
    except Exception as e:
        # Catch other potential errors (e.g., memory errors, unexpected issues)
        logger.error(f'An unexpected error occurred when loading data from {file_path}: {e}', exc_info=True)
        return []

    if not signals:
        logger.warning(f"No signals matching '{pressure_term}' or '{temp_term}' found in {file_path}.")

    return signals



def main():
    tf, uf, fmaxf = generate_multifreq_signal([6, 20, 13], [1,1,1], 5, 1000)
    ts, us, fmaxs = generate_sum_of_sines([6, 20, 13], [0.5, 0.3, 0.15], 1, 20000)

    import os
    import sys
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, src_path)
    import src.utilities.plotting as plotting
    plotting.plot_signal(tf, uf, title="Multi-Frequency Signal", filepath="multifreq_signal.png")
    plotting.plot_signal(ts, us, title="Sum of Sines Signal", filepath="sum_of_sines_signal.png")


if __name__ == "__main__":
    main()