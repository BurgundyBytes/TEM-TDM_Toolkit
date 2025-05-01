import os
import sys
from typing import Dict, Any, List, Optional, Tuple 
import logging

logger = logging.getLogger(__name__) 

# --- Data Types ---
ConfigDict = Dict[str, Any]
BoolsDict = Dict[str, bool]
OutputPathsDict = Dict[str, Optional[str]]


# --- Helper Functions for Parsing ---
def _parse_bool(value: str) -> bool:
    '''Safely parses a string into a boolean.'''
    return value.strip().lower() == 'true'

def _parse_float(value: str, key_name: str) -> float:
    '''Safely parses a string into a float.'''
    try:
        return float(value.strip())
    except ValueError:
        raise ValueError(f"Invalid float value for '{key_name}': '{value}'")

def _parse_int(value: str, key_name: str) -> int:
    '''Safely parses a string into an integer.'''
    try:
        return int(value.strip())
    except ValueError:
        raise ValueError(f"Invalid integer value for '{key_name}': '{value}'")

def _parse_float_list(value: str, key_name: str) -> List[float]:
    '''Safely parses a comma-separated string into a list of floats.'''
    try:
        return [float(f.strip()) for f in value.split(',')]
    except ValueError:
        raise ValueError(f"Invalid float list value for '{key_name}': '{value}'")

def _parse_string_list(value: str, key_name: str) -> List[str]:
    '''Safely parses a comma-separated string into a list of strings.'''
    return [s.strip() for s in value.split(',')]

def _parse_range(value: str, key_name: str) -> Optional[Tuple[float, float, float]]:
    '''Safely parses a comma-separated string 'start, end, step' into a tuple of floats.'''
    try:
        parts = [float(f.strip()) for f in value.split(',')]
        if len(parts) != 3:
            raise ValueError("Range must have 3 values (start, end, step)")
        start, end, step = parts
        # Add validation
        if step == 0:
            raise ValueError("Step cannot be zero")
        if step > 0 and start > end:
            raise ValueError(f"Start ({start}) cannot be greater than end ({end}) with positive step ({step})")
        if step < 0 and start < end:
            raise ValueError(f"Start ({start}) cannot be less than end ({end}) with negative step ({step})")
        return tuple(parts) 
    except ValueError as e:
        raise ValueError(f"Invalid range value for '{key_name}': '{value}'. Error: {e}")


# --- Main Configuration Loading ---
def load_config(filename: str = 'config.txt') -> ConfigDict:
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
    raw_config = {}
    try:
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#'):
                    if ':' in line:
                        key, value_comment = line.split(':', 1)
                        value = value_comment.split('#')[0].strip()
                        raw_config[key.strip()] = value
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{filename}' not found.")
    except Exception as e:
        raise IOError(f"Error reading configuration file '{filename}': {e}")

    config: ConfigDict = {}

    try:
        # ----------- Settings parameters -----------
        config['Input Folder'] = raw_config['Input Folder']
        config['Output Folder'] = raw_config['Output Folder']
        config['Execution Flow'] = _parse_bool(raw_config['Execution Flow'])
        config['Results Flow'] = _parse_bool(raw_config['Results Flow'])

        # ----------- Input signal parameters -----------
        config['Input Source'] = raw_config['Input Source'].lower()
        config['Mode'] = raw_config.get('Mode', 'single').lower() # Default to single if missing

        if config['Input Source'] == 'excel':
            config['Input File'] = raw_config['Input File'] 
            config['Filepaths'] = _parse_string_list(config['Input File'], 'Input File')
            config['Filenames'] = [os.path.basename(f).split(".")[0] for f in config['Filepaths']]
            config['Experiment Sampling Rate'] = _parse_float(raw_config['Experiment Sampling Rate'], 'Experiment Sampling Rate')
            config['Experiment Duration'] = _parse_float(raw_config['Experiment Duration'], 'Experiment Duration')
        
        elif config['Input Source'] == 'generated':
            # Check allowed types
            gen_type_raw = raw_config['Generated Input Type'].lower()
            allowed_gen_types = ['multisin', 'paper', 'multifreq']
            if gen_type_raw not in allowed_gen_types:
                raise ValueError(f"Invalid 'Generated Input Type': {gen_type_raw}. Must be one of {allowed_gen_types}")
            config['Generated Input Type'] = gen_type_raw
            config['Signal Duration'] = _parse_float(raw_config['Signal Duration'], 'Signal Duration')
            config['Encoder Resolution'] = _parse_float(raw_config['Encoder Resolution'], 'Encoder Resolution')
            config['Signal Sampling Rate'] = _parse_float(raw_config['Signal Sampling Rate'], 'Signal Sampling Rate')
            config['Frequencies'] = _parse_float_list(raw_config['Frequencies'], 'Frequencies')
            config['Amplitudes'] = _parse_float_list(raw_config['Amplitudes'], 'Amplitudes')
        else:
            raise ValueError(f"Invalid 'Input Source': {config['Input Source']}. Must be 'excel' or 'generated'.")

        # ----------- Output parameters -----------
        config['Parametric Bias Folder'] = raw_config['Parametric Bias Folder']
        config['Parametric Delta Folder'] = raw_config['Parametric Delta Folder']
        config['Biparametric Folder'] = raw_config['Biparametric Folder']
        config['Nyquist Analysis Folder'] = raw_config['Nyquist Analysis Folder']
        config['Optimal Conditions Folder'] = raw_config['Optimal Conditions Folder']
        config['Fourier Analysis Folder'] = raw_config['Fourier Analysis Folder']
        config['Plot Output'] = _parse_bool(raw_config['Plot Output'])
        config['Log to file'] = _parse_bool(raw_config['Log to file'])
        config['Store in pickle'] = _parse_bool(raw_config['Store in pickle'])

        # ----------- Parametric Studies parameters -----------
        config['Run Parametric Bias'] = _parse_bool(raw_config['Run Parametric Bias'])
        config['Run Parametric Delta'] = _parse_bool(raw_config['Run Parametric Delta'])
        config['Run Biparametric'] = _parse_bool(raw_config['Run Biparametric'])
        config['Delta Range'] = _parse_range(raw_config.get('Delta Range'), 'Delta Range') 
        config['Default Bias'] = _parse_float(raw_config['Default Bias'], 'Default Bias')
        config['Default Delta'] = _parse_float(raw_config['Default Delta'], 'Default Delta')

        # ----------- Analysis parameters -----------
        config['Run Optimal'] = _parse_bool(raw_config['Run Optimal'])
        config['Run Nyquist'] = _parse_bool(raw_config['Run Nyquist'])
        config['Run Fourier'] = _parse_bool(raw_config['Run Fourier'])
        config['Error threshold'] = _parse_float(raw_config['Error threshold'], 'Error threshold') if 'Error threshold' in raw_config else None
        config['Elapsed time threshold'] = _parse_float(raw_config['Elapsed time threshold'], 'Elapsed time threshold') if 'Elapsed time threshold' in raw_config else None

    except KeyError as e:
        raise KeyError(f"Missing required configuration key: {e}")

    logger.info(f"Configuration loaded successfully from '{filename}'.")
    return config


# --- Workflow Flag Extraction ---
def get_workflow_flags(config: ConfigDict) -> BoolsDict:
    '''
    Extracts the boolean flags controlling the main execution workflows
    from the loaded configuration dictionary.
    This function assumes that the configuration has been loaded and parsed correctly.

    Inputs
    ------
    - config: dict
        Dictionary containing all parsed configuration values.
        This should be the output of load_config.
    
    Outputs
    -------
    - bools: dict
        Dictionary containing boolean flags for the execution workflows.
        Keys include 'execute', 'results', 'optima', 'nyquist', 'fourier',
        'plots', 'logs', and 'pickle'.

    Raises
    ------
    - KeyError: If expected keys are missing in config (should not happen if load_config is correct).
    '''
    bools: BoolsDict = {}
    try:
        bools['execute'] = config['Execution Flow'] 
        bools['results'] = config['Results Flow']  
        bools['optima'] = config['Run Optimal']  
        bools['nyquist'] = config['Run Nyquist']   
        bools['fourier'] = config['Run Fourier']   
        bools['plots'] = config['Plot Output']     
        bools['logs'] = config['Log to file']   
        bools['pickle'] = config['Store in pickle'] 
    except KeyError as e:
        # This should ideally not happen if load_config succeeded
        raise KeyError(f"Missing key when extracting boolean flags: {e}. Check load_config.")

    logger.debug("Workflow control flags extracted.")
    return bools


# --- Output Folder Setup ---
def setup_output_folders(config: ConfigDict, bools: BoolsDict) -> OutputPathsDict:
    '''
    Creates output folders based on the loaded configuration and workflow flags.

    Folder Creation Logic:
    - Determines base path based on Input Source and specific type/filename.
    - Parametric folders are created ONLY IF bools['execute'] is True AND the
      corresponding 'Run Parametric ...' flag in the config is also True.
    - Analysis folders are created based on the corresponding flags in `bools`
      ('nyquist', 'optima', 'fourier').
    
    Inputs
    ------
    - config: dict
        Dictionary containing all parsed configuration values.
        This should be the output of load_config.
    - bools: dict
        Dictionary containing boolean flags for the execution workflows.
        This should be the output of get_workflow_flags.
    
    Outputs
    -------
    - output_paths: dict
        Dictionary mapping descriptive keys ('base_subcase', 'param_bias', etc.)
        to the absolute paths of created directories, or None if not created.

    Raises
    ------
    - OSError: If directory creation fails for reasons other than existing.
    - KeyError: If expected keys are missing in config (should not happen if load_config is correct).
    '''
    output_paths: OutputPathsDict = {} 
    
    try:
        base_output_folder = config['Output Folder']
        case_folder = config['Input Source'] # 'generated' or 'excel'

        # Determine subcase name
        subcase_name = None
        if case_folder == 'generated':
            subcase_name = config['Generated Input Type']
        elif case_folder == 'excel':
            # Use the first filename if available, otherwise a default
            filenames = config.get('Filenames')
            subcase_name = filenames[0] if filenames else 'excel_default'
        else:
            # This case should be caught by load_config, but defensively:
            subcase_name = 'unknown_case'

        # Construct base path for this subcase
        full_subcase_path = os.path.abspath(os.path.join(base_output_folder, case_folder, subcase_name))
        logger.info(f"Base output path for this run: {full_subcase_path}")
        os.makedirs(full_subcase_path, exist_ok=True)
        output_paths['base_subcase'] = full_subcase_path

        # --- Create Parametric Folders ---
        if bools.get('execute', False):
            logger.info("Execution Flow active. Checking parametric study flags.")
            
            if config.get('Run Parametric Bias', False):
                path = os.path.join(full_subcase_path, config['Parametric Bias Folder'])
                logger.info(f"\tCreating/Verifying Parametric Bias folder: {path}")
                os.makedirs(path, exist_ok=True)
                output_paths['param_bias'] = path
            else:
                logger.info("\tSkipping Parametric Bias folder (Run Parametric Bias is False).")
                output_paths['param_bias'] = None
            
            if config.get('Run Parametric Delta', False):
                path = os.path.join(full_subcase_path, config['Parametric Delta Folder'])
                logger.info(f"\tCreating/Verifying Parametric Delta folder: {path}")
                os.makedirs(path, exist_ok=True)
                output_paths['param_delta'] = path
            else:
                logger.info("\tSkipping Parametric Delta folder (Run Parametric Delta is False).")
                output_paths['param_delta'] = None

            if config.get('Run Biparametric', False):
                path = os.path.join(full_subcase_path, config['Biparametric Folder'])
                logger.info(f"\tCreating/Verifying Biparametric folder: {path}")
                os.makedirs(path, exist_ok=True)
                output_paths['biparametric'] = path
            else:
                logger.info("\tSkipping Biparametric folder (Run Biparametric is False).")
                output_paths['biparametric'] = None
        else:
            logger.info("Execution Flow inactive. Skipping all parametric folders.")
            output_paths['param_bias'] = None
            output_paths['param_delta'] = None
            output_paths['biparametric'] = None

        # --- Create Analysis Folders (using specific flags from bools) ---
        if bools.get('results', False):
            logger.info("Results Flow active. Checking analysis flags.")
            
            if bools.get('nyquist', False):
                path = os.path.join(full_subcase_path, config['Nyquist Analysis Folder'])
                logger.info(f"\tCreating/Verifying Nyquist folder: {path}")
                os.makedirs(path, exist_ok=True)
                output_paths['nyquist'] = path
            else:
                logger.info("\tSkipping Nyquist folder ('nyquist' flag is False).")
                output_paths['nyquist'] = None

            if bools.get('optima', False):
                path = os.path.join(full_subcase_path, config['Optimal Conditions Folder'])
                logger.info(f"\tCreating/Verifying Optima folder: {path}")
                os.makedirs(path, exist_ok=True)
                output_paths['optima'] = path 
            else:
                logger.info("\tSkipping Optima folder ('optima' flag is False).")
                output_paths['optima'] = None

            if bools.get('fourier', False):
                path = os.path.join(full_subcase_path, config['Fourier Analysis Folder'])
                logger.info(f"\tCreating/Verifying Fourier folder: {path}")
                os.makedirs(path, exist_ok=True)
                output_paths['fourier'] = path
            else:
                logger.info("\tSkipping Fourier folder ('fourier' flag is False).")
                output_paths['fourier'] = None
        else:
            logger.info("Results Flow inactive. Skipping all analysis folders.")
            output_paths['nyquist'] = None
            output_paths['optima'] = None
            output_paths['fourier'] = None


    except KeyError as e:
        logger.error(f"Missing configuration key during folder setup: {e}. Check config file and load_config.", exc_info=True) 
        raise
    except OSError as e:
        logger.error(f"OS error creating directory: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during folder setup: {e}", exc_info=True)
        raise

    logger.debug("Output folder setup complete.")
    return output_paths