# Parametric Handler Controller (`src/controllers/parametric_handler.py`)

## Overview

This module acts as the manager for the **Execution Workflow** of the TEM-TDM toolkit. Its primary responsibility is to orchestrate the running of various parametric studies (varying frequency, delta, or both) on the provided input signals based on the configuration settings. It manages the flow of data to the underlying study functions, handles the aggregation of results, and initiates the saving of these results.

## Functionality

1.  **Manager Function (`manager`)**:
    *   Serves as the main entry point for the parametric handler workflow.
    *   Receives global configuration, workflow flags, input signals, and output paths.
    *   Calls `run_studies` to execute the configured parametric sweeps.
    *   Calls `save_results` to persist the aggregated summary dataframes returned by `run_studies`.
    *   Returns the aggregated results dictionary (or `None`).
2.  **Study Orchestration (`run_studies`)**:
    *   Iterates through each provided input signal.
    *   Determines which parametric studies (frequency, delta, biparametric) need to be run based on the existence of corresponding output paths (configured and created by the `configuration` module).
    *   **Calculates Stable Region:** For each signal, it calls `utils.get_stable_region` *once* to determine the time indices corresponding to a stable portion of the signal (used for consistent metric calculation). These indices are passed down to the individual study runners.
    *   Calls the specific runner functions (`run_frequency`, `run_delta`, `run_biparametric`) for each enabled study, passing the signal data, stable region indices, configuration, flags, and the specific output directory.
    *   Aggregates the resulting summary DataFrames from each successful study run into a nested dictionary (`ParametricResultsDict`), keyed first by signal name, then by study type (`'freq'`, `'delta'`, `'freq_delta'`).
    *   Handles potential errors during individual study runs, logging warnings/errors but attempting to continue with other studies/signals.
3.  **Individual Study Runners (`run_frequency`, `run_delta`, `run_biparametric`)**:
    *   Each runner function acts as a setup and dispatch layer for a specific type of parametric study.
    *   Retrieves the necessary parameter ranges (e.g., `Frequency Range`, `Delta Range`) and default values from the `config` dictionary.
    *   Constructs the array of parameter values to be swept (e.g., `np.arange` for frequencies or deltas).
    *   Defines the specific log filename for that study and signal.
    *   Calls the corresponding core computation function from the `src.parametric.studies` module (e.g., `studies.parametric_freq`), passing all required arguments including the stable region indices.
    *   Handles potential errors during the call to the `studies` module.
    *   Returns the resulting summary DataFrame (or `None` if the study was skipped or failed).
4.  **Results Saving (`save_results`)**:
    *   Iterates through the aggregated `ParametricResultsDict`.
    *   For each signal and study type, it calls utility functions (`utils.store_df_to_excel`, `utils.save_pickle`) to save the summary DataFrame to the appropriate output folder (determined via a mapping from study type to output path key).
    *   Handles optional saving to Pickle format based on the `pickle` flag.
    *   Logs the saving process and handles potential file I/O errors.

## Visualization
````mermaid
%%{ init: { 'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'primaryTextColor': '#333', 'lineColor': '#666', 'fontSize': '14px' } } }%%
graph TD

    %% -------------------------------- Styles (Reusable Template Section) --------------------------------
    classDef inputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,color:#333,rx:4,ry:4
    classDef processStyle fill:#fff,stroke:#888,stroke-width:1.5px,color:#000,font-weight:bold,rx:4,ry:4
    classDef outputStyle fill:#e8f5e9,stroke:#a5d6a7,stroke-width:1px,color:#333,rx:4,ry:4
    classDef optionalInputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,stroke-dasharray: 4 4,color:#333,rx:4,ry:4
    classDef helperStyle fill:#f3e5f5,stroke:#ba68c8,stroke-width:1px,color:#333,rx:2,ry:2 %% Style for internal helpers/steps
    classDef externalStyle fill:#ffe0b2,stroke:#ffb74d,stroke-width:1px,color:#333,rx:2,ry:2
    classDef fileOutputStyle fill:#e8f5e9,stroke:#a5d6a7,stroke-width:1px,stroke-dasharray: 3 3,color:#555,rx:4,ry:4 %% Dashed border for file outputs

    %% -------------------------------- Diagram Structure: High-Level parametric_handler.py --------------------------------

    subgraph Inputs
        direction TB
        In1[config: ConfigDict<br/><span style='font-size:0.9em; color:#444'>Global Configuration</span>]:::inputStyle
        In2[bools: BoolsDict<br/><span style='font-size:0.9em; color:#444'>Execution/Output Flags</span>]:::inputStyle
        In3[signals: List of SignalDict<br/><span style='font-size:0.9em; color:#444'>List of Signal Data</span>]:::inputStyle
        In4[output_paths: OutputPathsDict<br/><span style='font-size:0.9em; color:#444'>Paths for Parametric Results</span>]:::inputStyle
    end

    subgraph Process
        direction TB
        P_Main[manager<br/><span style='font-size:0.9em; color:#444'>Main Parametric Orchestrator</span>]:::processStyle

        Step_RunStudies[run_studies<br/><span style='font-size:0.9em; color:#444'>Orchestrates studies per signal</span>]:::helperStyle

        subgraph SignalLoop [Loop Through Each Signal]
            direction TB
            Step_StableRegion[Calculate Stable Region<br/><span style='font-size:0.9em; color:#444'>Calls utils.get_stable_region</span>]:::helperStyle
            Ext_StableUtil[utils.get_stable_region]:::externalStyle

            Step_RunFreq[run_frequency<br/><span style='font-size:0.9em; color:#444'>Runner for Frequency Study</span>]:::helperStyle
            Call_StudyFreq[studies.parametric_freq]:::externalStyle

            Step_RunDelta[run_delta<br/><span style='font-size:0.9em; color:#444'>Runner for Delta Study</span>]:::helperStyle
            Call_StudyDelta[studies.parametric_delta]:::externalStyle

            Step_RunBiParam[run_biparametric<br/><span style='font-size:0.9em; color:#444'>Runner for Biparametric Study</span>]:::helperStyle
            Call_StudyBiParam[studies.parametric_freq_delta]:::externalStyle
        end

        Step_SaveResults[save_results<br/><span style='font-size:0.9em; color:#444'>Saves aggregated summary results</span>]:::helperStyle
        Ext_SaveUtil[utils.store_df_to_excel / utils.save_pickle]:::externalStyle

    end


    subgraph Outputs
        O_ResultsDict[ParametricResultsDict<br/><span style='font-size:0.9em; color:#444'>Primary return: Dictionary of<br/>summary DataFrames per signal/study</span>]:::outputStyle
        O_Files[Parametric Result Files<br/><span style='font-size:0.9em; color:#444'>Generated in output_paths folders:<br/>- Summary Excel/Pickle<br/>- Per-run plots/logs/raw data, by studies module</span>]:::fileOutputStyle
        O_Logs[Log Messages<br/><span style='font-size:0.9em; color:#444'>Progress and status updates</span>]:::outputStyle
    end

    %% -------------------------------- Connections --------------------------------
    Inputs --> P_Main

    P_Main -- Calls --> Step_RunStudies
    Step_RunStudies -- Initiates --> SignalLoop

    %% Inside the loop
    SignalLoop --> Step_StableRegion
    Step_StableRegion -- Calls --> Ext_StableUtil

    Step_StableRegion -- Stable Indices --> Step_RunFreq
    Step_RunFreq -- Calls --> Call_StudyFreq

    Step_StableRegion -- Stable Indices --> Step_RunDelta
    Step_RunDelta -- Calls --> Call_StudyDelta

    Step_StableRegion -- Stable Indices --> Step_RunBiParam
    Step_RunBiParam -- Calls --> Call_StudyBiParam

    %% Collecting Results
    Call_StudyFreq -- Returns Summary DataFrame --> Step_RunStudies
    Call_StudyDelta -- Returns Summary DataFrame --> Step_RunStudies
    Call_StudyBiParam -- Returns Summary DataFrame --> Step_RunStudies

    Step_RunStudies -- Aggregated Results Dict --> P_Main

    P_Main -- Calls --> Step_SaveResults
    Step_SaveResults -- Uses Aggregated Results Dict --> Ext_SaveUtil

    %% Outputs Generation
    P_Main -- Returns --> O_ResultsDict

    %% File Outputs (Side Effects) - Conceptually linked
    Call_StudyFreq -- Generates --> O_Files
    Call_StudyDelta -- Generates --> O_Files
    Call_StudyBiParam -- Generates --> O_Files
    Ext_SaveUtil -- Generates --> O_Files

    %% Logging output
    P_Main -- Generates --> O_Logs
    SignalLoop -- Generates --> O_Logs
````