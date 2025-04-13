import os
import sys
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# custom modules
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, src_path)
from src.models import input_signal


# --- Data Types ---
SignalData = Dict[str, Any]
InputSignals = List[SignalData]
ConfigDict = Dict[str, Any]


def generate_input_signal(config: ConfigDict) -> InputSignals:
    '''
    Generates the preferred input signal based on configuration.

    Inputs
    ------
    - config: dict  
        Dictionary with the fully parsed configuration.
    
    Outputs
    -------
    - signals: list of dict
        List containing a single dictionary representing the generated signal.
        
    Raises
    ------
    - KeyError: If required keys are missing in the configuration.
    - ValueError: If there are issues with the signal generation (e.g., mismatched lengths).
    - Exception: For any other unexpected errors during signal generation.
    '''
    signals: InputSignals = []
    try:
        gen_type = config['Generated Input Type']
        fs = config['Signal Sampling Rate']
        dur = config['Signal Duration']
        b = config.get('Encoder Bias', 0.0)
        dte = config.get('Encoder Resolution', 0.0)
        logger.info(f"Generating signal type: {gen_type} (Fs={fs} Hz, Dur={dur} s)")

        t: Optional[np.ndarray] = None
        u: Optional[np.ndarray] = None
        freq_max: Optional[float] = None

        # Choose the signal generation method based on the type specified in the config
        if gen_type == 'multisin':
            freqs = config['Frequencies']
            amps = config['Amplitudes']
            logger.debug(f"Generating multisin with Freqs={freqs}, Amps={amps}")
            t, u, freq_max = input_signal.generate_sum_of_sines(freqs, amps, dur, fs)

        elif gen_type == 'multifreq':
            freqs = config['Frequencies']
            amps = config['Amplitudes']
            logger.debug(f"Generating multifreq with Freqs={freqs}, Amps={amps}")
            t, u, freq_max = input_signal.generate_multifreq_signal(freqs, amps, dur, fs)

        elif gen_type == 'paper':  # >>> I'm not sure i correctly interpreted the paper signal generation 
            logger.debug("Generating paper signal")
            t, u, freq_max = input_signal.generate_paper_signal(fs)

        else:
            logger.error(f"Error: Invalid 'Generated Input Type' in configuration: {gen_type}")
            return [] 

        # Package the result
        if t is not None and u is not None:
            signal_name = f"generated_{gen_type}"
            signals.append({
                't': t,
                'u': u,
                'freq_max': freq_max,
                'name': signal_name,
                'fs': fs,
                'dur': dur,
                'b': b,
                'dte': dte,
                'source': 'generated' # hardcoded for now -> to be changed later, when excel import is fully developed
            })
            logger.debug(f"Generated signal '{signal_name}' with {len(t)} points.")
        else:
            logger.error(f"Signal generation for type '{gen_type}' failed to return data.")
            return []
        
    except KeyError as e:
        logger.error(f"Missing required key in 'generated' config for signal generation: {e}", exc_info=True)
        return []
    except ValueError as e:
        logger.error(f"Value error during signal generation: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Unexpected error during generated signal processing: {e}", exc_info=True)
        return []

    return signals


def get_experiment_signals(config: ConfigDict) -> InputSignals:
    '''
    Loads input signals from experiment files specified in the configuration.
    
    >>> Note:
        This functionality is not fully implemented yet and is a placeholder for future work.
        If the input source is 'excel', it will attempt to load signals from the specified files. However,
        the actual loading logic is not yet reviewed or implemented.
        Currently, it will log a warning and return an empty list.

    Inputs
    ------
    - config: dict  
        Dictionary with the fully parsed configuration.

    Outputs
    -------
    - signals: list of dict
        List of dictionaries representing the loaded signals.

    Raises
    ------
    - KeyError: If required keys are missing in the configuration.
    - Exception: For any other unexpected errors during signal loading.
    '''
    all_signals: InputSignals = []
    try:
        file_paths = config.get('Filepaths', []) # Get list of paths, default to empty
        file_names = config.get('Filenames', []) # Get corresponding names

        if not file_paths:
            logger.warning("¡'Input Source' is 'excel' but no 'Input File' paths are specified in config.")
            return []

        if len(file_paths) != len(file_names):
            logger.warning("Mismatch between number of Filepaths and Filenames in config. Using paths for naming.")
            # Adjust filenames or handle as appropriate, here we'll just use a generic name based on path index
            file_names = [f"file_{i}" for i in range(len(file_paths))] # Example fallback

        logger.info(f"Loading experiment signals from {len(file_paths)} file(s)...")
        for file_path, file_name in zip(file_paths, file_names):
            logger.info(f"\tProcessing file: {file_path} (as '{file_name}')")
            # Pass the main config dict to the loading function
            signals_from_file = input_signal.load_experiment_signals(file_path, file_name, config)
            if signals_from_file:
                all_signals.extend(signals_from_file)
                logger.info(f"\t\t-> Found {len(signals_from_file)} signal(s).")
            else:
                logger.info(f"\t\t-> No signals loaded or error processing file.")

    except KeyError as e:
        # Should be caught by .get, but defensive check
        logger.error(f"Error: Missing key in 'excel' input configuration: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred during experiment signal processing setup: {e}", exc_info=True)
        return []

    if not all_signals:
        logger.info("No experiment signals were successfully loaded.")

    return all_signals


def manager(config: ConfigDict) -> InputSignals:
    '''
    Main function to manage the extraction or generation of input data based on config.

    Inputs
    ------
    - config: dict  
        Dictionary with the fully parsed configuration.
    
    Outputs
    -------
    - signals: list of dict
        List of dictionaries representing the input signals.

    Raises
    ------
    - Exception: For any unexpected errors during input handling.
    '''
    signals: InputSignals = []
    try:
        input_source = config.get('Input Source')
        if input_source is None:
            logger.error("'Input Source' not specified in configuration.")
            return []

        logger.info(f"Input Source: {input_source}")

        if input_source == 'excel':
            # signals = get_experiment_signals(config) # >>> Keep if implemented
            logger.warning("Excel input source handling not fully reviewed/implemented yet.")
            return [] 
        elif input_source == 'generated':
            signals = generate_input_signal(config)
        else:
            # Should be caught by config validation, but defensive check
            logger.error(f"Invalid 'Input Source': {input_source}. Must be 'excel' or 'generated'.")
            return []

    except Exception as e:
        logger.error(f"FATAL Error in input handler manager: {e}", exc_info=True)
        return []

    if not signals:
        logger.warning("Input handler finished: No signals were generated or loaded.")
    else:
        logger.info(f"Input handler finished: Produced {len(signals)} signal(s).")

    return signals