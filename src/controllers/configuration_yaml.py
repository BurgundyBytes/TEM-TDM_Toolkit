import os
import sys
import yaml 
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import math

logger = logging.getLogger(__name__)

# --- Data Types ---
ConfigDict = Dict[str, Any]
BoolsDict = Dict[str, bool]
OutputPathsDict = Dict[str, Optional[str]]

# --- Parsing helpers ---
def _parse_bool(value: Any) -> bool:
    """Safely parses a loaded value into a boolean."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == 'true'

def _parse_float(value: Any, key_name: str) -> float: 
    """Safely parses a loaded value into a float."""
    try:
        # Check for None before converting
        if value is None:
             raise ValueError(f"Value for '{key_name}' cannot be None/null")
        return float(value)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid float value for '{key_name}': '{value}'")

def _parse_int(value: Any, key_name: str) -> int: 
    """Safely parses a loaded value into an integer."""
    try:
        if value is None:
             raise ValueError(f"Value for '{key_name}' cannot be None/null")
        return int(value)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid integer value for '{key_name}': '{value}'")

def _parse_float_list(value: Any, key_name: str) -> List[float]: 
    """Safely parses a loaded list value into a list of floats."""
    if not isinstance(value, list):
        raise ValueError(f"Expected a list for '{key_name}', got: {type(value)}")
    try:
        return [float(item) for item in value]
    except (ValueError, TypeError):
        raise ValueError(f"Invalid float value within list for '{key_name}': '{value}'")

def _parse_string_list(value: Any, key_name: str) -> List[str]: 
    """Safely parses a loaded list value into a list of strings."""
    if not isinstance(value, list):
         raise ValueError(f"Expected a list for '{key_name}', got: {type(value)}")
    return [str(item) for item in value] # Convert items to string

def _validate_range_list(value: Any, key_name: str) -> Optional[Tuple[float, float, float]]:
    """Validates a list loaded from YAML represents a valid range [start, end, step]."""
    # This helper remains mostly the same, operating on the list loaded by YAML
    if value is None:
         raise ValueError(f"Range value for '{key_name}' is missing or null.")
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"Invalid range format for '{key_name}'. Expected list of 3 numbers [start, end, step], got: {value}")
    try:
        start, end, step = [float(v) for v in value]
    except (ValueError, TypeError) as e:
         raise ValueError(f"Invalid numeric value within range for '{key_name}': {value}. Error: {e}")

    if step == 0: raise ValueError(f"Range step cannot be zero for '{key_name}'.")
    if step > 0 and start > end: raise ValueError(f"Range start ({start}) > end ({end}) with positive step for '{key_name}'.")
    if step < 0 and start < end: raise ValueError(f"Range start ({start}) < end ({end}) with negative step for '{key_name}'.")
    return (start, end, step)

# --- Loading and Flattening Function ---
def load_config(filename: str = 'config.yaml') -> ConfigDict:
    '''
    Reads the configuration file ONCE, parses all sections, performs type
    conversions, and returns a single dictionary containing all configuration parameters.

    Inputs
    ------
    - filename: str
        Name of the configuration file to read.
        Default is 'config.txt'.
    
    Outputs
    -------
    - config: dict
        Dictionary containing all parsed configuration values.

    Raises
    ------  
    - FileNotFoundError: If the configuration file is not found.
    - IOError: If there's an error reading the file.
    - KeyError: If a required configuration key is missing.
    - ValueError: If a configuration value has an invalid format or type.
    '''
    try:
        with open(filename, 'r') as file:
            # Load the raw nested structure from YAML
            nested_config = yaml.safe_load(file)
            if nested_config is None:
                 raise ValueError(f"Configuration file '{filename}' is empty or invalid.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{filename}' not found.")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file '{filename}': {e}")
    except Exception as e:
        raise IOError(f"Error reading configuration file '{filename}': {e}")

    # --- Flatten the nested dictionary and map keys ---
    flat_config: ConfigDict = {}
    key_map = {
        # general_settings -> Flat Key
        ('general_settings', 'input_folder'): 'Input Folder',
        ('general_settings', 'output_folder'): 'Output Folder',
        ('general_settings', 'execution_flow'): 'Execution Flow',
        ('general_settings', 'results_flow'): 'Results Flow',
        ('general_settings', 'log_to_file'): 'Log to file',
        ('general_settings', 'store_in_pickle'): 'Store in pickle',
        ('general_settings', 'plot_output'): 'Plot Output',

        # input_signal -> Flat Key
        ('input_signal', 'source'): 'Input Source',
        ('input_signal', 'mode'): 'Mode',
        ('input_signal', 'generated_type'): 'Generated Input Type',
        ('input_signal', 'signal_duration'): 'Signal Duration',
        ('input_signal', 'encoder_resolution'): 'Encoder Resolution',
        ('input_signal', 'sampling_rate'): 'Signal Sampling Rate',
        ('input_signal', 'frequencies'): 'Frequencies',
        ('input_signal', 'amplitudes'): 'Amplitudes',
        ('input_signal', 'filepaths'): 'Input File', 
        ('input_signal', 'experiment_sampling_rate'): 'Experiment Sampling Rate',
        ('input_signal', 'experiment_duration'): 'Experiment Duration',

        # output_folder_names -> Flat Key
        ('output_folder_names', 'parametric_bias'): 'Parametric Bias Folder',
        ('output_folder_names', 'parametric_delta'): 'Parametric Delta Folder',
        ('output_folder_names', 'biparametric'): 'Biparametric Folder',
        ('output_folder_names', 'nyquist'): 'Nyquist Analysis Folder',
        ('output_folder_names', 'optimal_conditions'): 'Optimal Conditions Folder',
        ('output_folder_names', 'fourier'): 'Fourier Analysis Folder',

        # parametric_studies -> Flat Key
        ('parametric_studies', 'run_parametric_bias'): 'Run Parametric Bias',
        ('parametric_studies', 'run_parametric_delta'): 'Run Parametric Delta',
        ('parametric_studies', 'run_biparametric'): 'Run Biparametric',
        ('parametric_studies', 'bias_range'): 'Bias Range', # Add if needed
        ('parametric_studies', 'delta_range'): 'Delta Range',
        ('parametric_studies', 'default_bias'): 'Default Bias',
        ('parametric_studies', 'default_delta'): 'Default Delta',

        # analysis -> Flat Key
        ('analysis', 'run_optimal'): 'Run Optimal',
        ('analysis', 'run_nyquist'): 'Run Nyquist',
        ('analysis', 'run_fourier'): 'Run Fourier',
        ('analysis', 'error_threshold'): 'Error threshold',
        ('analysis', 'elapsed_time_threshold'): 'Elapsed time threshold',
        # Add encoder analysis keys if needed
    }

    # Iterate through the mapping to flatten the structure
    for (section, key), flat_key in key_map.items():
        if section in nested_config and key in nested_config[section]:
            flat_config[flat_key] = nested_config[section][key]

    # --- Perform Type Conversion and Validation on the *FLAT* dictionary ---
    final_config: ConfigDict = {}
    try:
        # Settings
        final_config['Input Folder'] = str(flat_config['Input Folder']) 
        final_config['Output Folder'] = str(flat_config['Output Folder'])
        final_config['Execution Flow'] = _parse_bool(flat_config['Execution Flow'])
        final_config['Results Flow'] = _parse_bool(flat_config['Results Flow'])

        # Input Signal (conditional logic based on source)
        final_config['Input Source'] = str(flat_config['Input Source']).lower()
        final_config['Mode'] = str(flat_config.get('Mode', 'single')).lower() 

        if final_config['Input Source'] == 'excel':
            # Note: 'Input File' now holds the LIST from YAML 'filepaths'
            filepaths_list = flat_config['Input File']
            if not isinstance(filepaths_list, list):
                raise TypeError("Expected 'input_signal.filepaths' in YAML to be a list")
            final_config['Filepaths'] = _parse_string_list(filepaths_list, 'Filepaths') 
            final_config['Input File'] = ", ".join(final_config['Filepaths']) 
            final_config['Filenames'] = [os.path.basename(f).split(".")[0] for f in final_config['Filepaths']]
            final_config['Experiment Sampling Rate'] = _parse_float(flat_config['Experiment Sampling Rate'], 'Experiment Sampling Rate')
            final_config['Experiment Duration'] = _parse_float(flat_config['Experiment Duration'], 'Experiment Duration')

        elif final_config['Input Source'] == 'generated':
            gen_type_raw = str(flat_config['Generated Input Type']).lower()
            allowed_gen_types = ['multisin', 'paper', 'multifreq']
            if gen_type_raw not in allowed_gen_types:
                raise ValueError(f"Invalid 'Generated Input Type': {gen_type_raw}. Must be one of {allowed_gen_types}")
            final_config['Generated Input Type'] = gen_type_raw
            final_config['Signal Duration'] = _parse_float(flat_config['Signal Duration'], 'Signal Duration')
            final_config['Encoder Resolution'] = _parse_float(flat_config['Encoder Resolution'], 'Encoder Resolution')
            final_config['Signal Sampling Rate'] = _parse_float(flat_config['Signal Sampling Rate'], 'Signal Sampling Rate')
            # Parse lists loaded by YAML
            final_config['Frequencies'] = _parse_float_list(flat_config['Frequencies'], 'Frequencies')
            final_config['Amplitudes'] = _parse_float_list(flat_config['Amplitudes'], 'Amplitudes')
        else:
             raise ValueError(f"Invalid 'Input Source': {final_config['Input Source']}")

        # Output Folders (just store names as strings)
        folder_keys = ['Parametric Bias Folder', 'Parametric Delta Folder', 'Biparametric Folder',
                       'Nyquist Analysis Folder', 'Optimal Conditions Folder', 'Fourier Analysis Folder']
        for key in folder_keys:
             if key in flat_config: # Handle potentially missing keys if studies disabled
                  final_config[key] = str(flat_config[key])
             else:
                  logger.warning(f"Optional folder name key '{key}' not found in flattened config (likely corresponding study disabled).")
                  final_config[key] = key.replace(" Folder", "").replace(" ", "_").lower() # Generate default name

        # Output Control Flags
        final_config['Plot Output'] = _parse_bool(flat_config['Plot Output'])
        final_config['Log to file'] = _parse_bool(flat_config['Log to file'])
        final_config['Store in pickle'] = _parse_bool(flat_config['Store in pickle'])

        # Parametric Studies Flags
        final_config['Run Parametric Bias'] = _parse_bool(flat_config['Run Parametric Bias'])
        final_config['Run Parametric Delta'] = _parse_bool(flat_config['Run Parametric Delta'])
        final_config['Run Biparametric'] = _parse_bool(flat_config['Run Biparametric'])

        # Parametric Studies Ranges & Defaults
        # Validate range *if* run flag is true and key exists
        if final_config['Run Parametric Bias'] and 'Bias Range' in flat_config:
            final_config['Bias Range'] = _validate_range_list(flat_config['Bias Range'], 'Bias Range')
        if final_config['Run Parametric Delta'] and 'Delta Range' in flat_config:
             final_config['Delta Range'] = _validate_range_list(flat_config['Delta Range'], 'Delta Range')

        if 'Default Bias' in flat_config: final_config['Default Bias'] = _parse_float(flat_config['Default Bias'], 'Default Bias')
        if 'Default Delta' in flat_config: final_config['Default Delta'] = _parse_float(flat_config['Default Delta'], 'Default Delta')

        # Analysis Flags
        final_config['Run Optimal'] = _parse_bool(flat_config['Run Optimal'])
        final_config['Run Nyquist'] = _parse_bool(flat_config['Run Nyquist'])
        final_config['Run Fourier'] = _parse_bool(flat_config['Run Fourier'])

        # Analysis Thresholds
        final_config['Error threshold'] = _parse_float(flat_config['Error threshold'], 'Error threshold') if 'Error threshold' in flat_config and flat_config['Error threshold'] is not None else None
        final_config['Elapsed time threshold'] = _parse_float(flat_config['Elapsed time threshold'], 'Elapsed time threshold') if 'Elapsed time threshold' in flat_config and flat_config['Elapsed time threshold'] is not None else None

    except KeyError as e:
        raise KeyError(f"Missing required configuration key in flattened dictionary: {e}. Check YAML file and key_map.")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Configuration validation error after flattening: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during configuration post-processing: {e}", exc_info=True)
        raise

    logger.info(f"YAML configuration loaded and flattened successfully from '{filename}'.")
    return final_config


# --- Workflow Flag Extraction ---
def get_workflow_flags(config: ConfigDict) -> BoolsDict:
    '''
    Extracts the boolean flags controlling the main execution workflows
    from the loaded (flattened) configuration dictionary.
    '''
    bools: BoolsDict = {}
    try:
        # Access keys directly from the flat dictionary
        bools['execute'] = config['Execution Flow']
        bools['results'] = config['Results Flow']
        bools['optima'] = config['Run Optimal']
        bools['nyquist'] = config['Run Nyquist']
        bools['fourier'] = config['Run Fourier']
        bools['plots'] = config['Plot Output']
        bools['logs'] = config['Log to file']
        bools['pickle'] = config['Store in pickle']
    except KeyError as e:
        # Catch missing *required* keys
        raise KeyError(f"Missing required key when extracting boolean flags: {e}.")
    except Exception as e:
        logger.error(f"Unexpected error extracting workflow flags: {e}", exc_info=True)
        raise

    logger.debug("Workflow control flags extracted.")
    return bools


# --- Output Folder Setup ---
def setup_output_folders(config: ConfigDict, bools: BoolsDict) -> OutputPathsDict:
    ''' Creates output folders based on the loaded (flattened) config and flags. '''
    output_paths: OutputPathsDict = {}
    try:
        # Access keys directly from the flat dictionary
        base_output_folder = config['Output Folder']
        input_source = config['Input Source']

        subcase_name = None
        if input_source == 'generated':
            subcase_name = config.get('Generated Input Type', 'unknown_gen_type')
        elif input_source == 'excel':
            # Use 'Filenames' derived during loading
            filenames = config.get('Filenames')
            subcase_name = filenames[0] if filenames else 'excel_default'
        else:
            subcase_name = 'unknown_case'

        full_subcase_path = os.path.abspath(os.path.join(base_output_folder, input_source, subcase_name))
        logger.info(f"Base output path for this run: {full_subcase_path}")
        os.makedirs(full_subcase_path, exist_ok=True)
        output_paths['base_subcase'] = full_subcase_path

        # --- Create Parametric Folders ---
        if bools.get('execute', False):
            logger.info("Execution Flow active. Checking parametric study flags.")
            param_folders_to_check = {
                'param_bias': ('Run Parametric Bias', 'Parametric Bias Folder'),
                'param_delta': ('Run Parametric Delta', 'Parametric Delta Folder'),
                'biparametric': ('Run Biparametric', 'Biparametric Folder')
            }
            for path_key, (bool_key, folder_key) in param_folders_to_check.items():
                if config.get(bool_key, False): # Check flag in flat config
                    folder_name = config.get(folder_key)
                    if not folder_name: raise KeyError(f"Missing folder name key '{folder_key}' when '{bool_key}' is true.")
                    path = os.path.join(full_subcase_path, folder_name)
                    logger.info(f"\tCreating/Verifying {path_key} folder: {path}")
                    os.makedirs(path, exist_ok=True)
                    output_paths[path_key] = path
                else:
                    logger.info(f"\tSkipping {path_key} folder ('{bool_key}' is False).")
                    output_paths[path_key] = None
        else:
            logger.info("Execution Flow inactive. Skipping all parametric folders.")
            output_paths['param_bias'] = None
            output_paths['param_delta'] = None
            output_paths['biparametric'] = None

        # --- Create Analysis Folders ---
        if bools.get('results', False):
             logger.info("Results Flow active. Checking analysis flags.")
             analysis_folders_to_check = {
                'nyquist': ('nyquist', 'Nyquist Analysis Folder'),
                'optima': ('optima', 'Optimal Conditions Folder'),
                'fourier': ('fourier', 'Fourier Analysis Folder')
             }
             for path_key, (bool_key, folder_key) in analysis_folders_to_check.items():
                 if bools.get(bool_key, False): # Use flag from BoolsDict
                    folder_name = config.get(folder_key)
                    if not folder_name: raise KeyError(f"Missing folder name key '{folder_key}' when '{bool_key}' flag is true.")
                    path = os.path.join(full_subcase_path, folder_name)
                    logger.info(f"\tCreating/Verifying {path_key} folder: {path}")
                    os.makedirs(path, exist_ok=True)
                    output_paths[path_key] = path
                 else:
                    logger.info(f"\tSkipping {path_key} folder ('{bool_key}' flag is False).")
                    output_paths[path_key] = None
        else:
             logger.info("Results Flow inactive. Skipping all analysis folders.")
             output_paths['nyquist'] = None
             output_paths['optima'] = None
             output_paths['fourier'] = None

    except KeyError as e:
        logger.error(f"Missing configuration key during folder setup: {e}. Check YAML/flat config.", exc_info=True)
        raise
    except OSError as e:
        logger.error(f"OS error creating directory: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during folder setup: {e}", exc_info=True)
        raise

    logger.debug("Output folder setup complete.")
    return output_paths