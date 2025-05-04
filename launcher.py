import sys
from typing import Dict, Any, Optional, List
import logging
import pandas as pd

# Configure logging (basic setup)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# custom modules
import src.controllers.configuration as configuration
import src.controllers.input_handler as input_handler 
import src.controllers.parametric_handler as parametric_handler 
import src.controllers.results_handler as results_handler

# --- Data Types ---
ConfigDict = Dict[str, Any]
BoolsDict = Dict[str, bool]
OutputPathsDict = Dict[str, Optional[str]]
SignalData = Dict[str, Any]
InputSignals = List[SignalData]
SignalResultsDict = Dict[str, Optional[pd.DataFrame]]
ParametricResultsDict = Dict[str, SignalResultsDict]


def main(config_filename: str = 'config.txt') -> None:
    '''
    Main execution function for the signal processing plant.
    This function orchestrates the loading of configuration, setting up output folders,
    generating/loading input signals, executing parametric studies, and handling results.

    Inputs
    -------
    - config_filename: str
        Path to the configuration file. Default is 'config.txt'.

    Raises
    -------
    - SystemExit: If any critical error occurs during the execution process.
        This will terminate the program with a non-zero exit code.
    '''
    logger.info("--- Signal Processing Plant Startup ---")
    
    # 1. Load Configuration
    config: ConfigDict = {}
    try:
        config = configuration.load_config(config_filename)
    except (FileNotFoundError, IOError, KeyError, ValueError) as e:
        logger.error(f"FATAL ERROR loading configuration '{config_filename}': {e}", exc_info=True)
        sys.exit(1)

    # 2. Get Workflow Control Flags
    # >> Note: get_workflow_flags expects config to be valid if load_config succeeded
    bools: BoolsDict = {}
    try:
        bools = configuration.get_workflow_flags(config)
        logger.info('Workflow control flags:')
        for key, value in bools.items():
            logger.info(f"\t{key}: {value}")
    except KeyError as e:
        logger.error(f"FATAL ERROR extracting workflow flags: {e}. Check config and get_workflow_flags.", exc_info=True)
        sys.exit(1)

    # 3. Setup Output Folders
    output_paths: OutputPathsDict = {}
    try:
        output_paths = configuration.setup_output_folders(config, bools)
        logger.info('Output folder paths configured:')
        for key, value in output_paths.items():
            logger.info(f"\t{key}: {value if value else '(Not Created/Needed)'}")
        if not output_paths.get('base_subcase'):
            logger.critical("Base output directory could not be determined or created. Exiting.")
            sys.exit(1)
    except (KeyError, OSError) as e:
        logger.error(f"FATAL ERROR setting up output folders: {e}", exc_info=True)
        sys.exit(1)

    # 4. Generate/Load Input Signal(s)
    ipt: InputSignals = {}
    try:
        logger.info("--- Input Handler ---")
        ipt = input_handler.manager(config)
        if not ipt:
            logger.error("Input handler failed to produce signal data. Exiting.")
            sys.exit(1)
        logger.info(f"Input signals generated/loaded: {[signal.get('name', 'Unnamed') for signal in ipt]}")
    except Exception as e:
        logger.error(f"ERROR during input signal handling: {e}", exc_info=True)
        sys.exit(1)

    # 5. Execute Workflow (Parametric Studies)
    parametric_summary_results: ParametricResultsDict = {}
    if bools.get('execute', False):
        logger.info("\n--- Starting Execution Workflow (Parametric Handler) ---")
        try:
            parametric_summary_results = parametric_handler.manager(config, bools, ipt, output_paths)

            if parametric_summary_results is not None:
                logger.info("Parametric handler completed successfully.")
            else:
                logger.warning("Parametric handler ran but returned no results (or None).")
        except Exception as e:
            logger.error(f"ERROR during parametric execution: {e}", exc_info=True)
            logger.warning("Attempting to continue to Results Workflow despite Parametric error (if enabled).")
    else:
        logger.info("\n--- Execution Workflow Skipped (disabled in config) ---")

    # 6. Results Workflow (Analysis)
    if bools.get('results', False):
        logger.info("\n--- Starting Results Workflow (Results Handler) ---")
        try:
            if parametric_summary_results is not None:
                logger.info("Results manager will use data from the current Execution workflow run.")
            else:
                logger.info("Execution workflow did not run or produced no data. Results manager will attempt to load previous results.")

            all_results = results_handler.manager(config, bools, ipt, output_paths, parametric_summary_results)
            logger.info("Results handler completed.")
        except Exception as e:
            logger.error(f"ERROR during results generation: {e}", exc_info=True)
    else:
        logger.info("\n--- Results Workflow Skipped (disabled in config) ---")

    logger.info("\n--- Plant Operations Complete ---")

if __name__ == "__main__":
    config_path = 'config.txt'
    main(config_path)