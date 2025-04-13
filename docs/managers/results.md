# Results Handler Controller (`src/controllers/results_handler.py`)

## Overview

This module manages the **Results Workflow** of the TEM-TDM toolkit. It orchestrates the various analysis tasks performed *after* the parametric studies (or using previously generated parametric results). Its primary functions include identifying optimal encoding parameters, simulating the encoding/decoding process with these optimal parameters, and performing comparative analyses like Nyquist and Fourier comparisons.

## Functionality

1.  **Manager Function (`manager`)**:
    *   Serves as the main entry point for the results analysis workflow.
    *   Receives global configuration, workflow flags, input signals, output paths, and potentially the live summary results from the `parametric_handler` (`parametric_summary_results`).
    *   **Data Acquisition:** Checks if live parametric summary results are available. If not, and if any analysis requiring them is enabled (`optima`, `nyquist`, `fourier`), it calls the internal `_load_summary_data` helper to load previously saved summary results from files (Excel or Pickle).
    *   **Signal Iteration:** Loops through each provided input signal.
    *   **Analysis Orchestration (per signal):**
        *   Retrieves the relevant parametric summary data for the current signal.
        *   Calls `optima.perform_optima_study` (from `src.analysis.optima`) if any dependent analysis is enabled. This step finds the optimal parameters (e.g., best `delta`) based on the parametric data and configuration criteria, and *also* performs the simulation using these optimal parameters.
        *   Calls `nyquist.perform_nyquist_study` (from `src.analysis.nyquist`) if enabled *and* if the optimal simulation step was successful, passing the optimal simulation results.
        *   Calls `fourier.perform_fourier_study` (from `src.analysis.fourier`) if enabled *and* if the optimal simulation step was successful, passing the optimal simulation results.
    *   Handles potential errors during individual analysis steps, logging warnings/errors but attempting to continue with other analyses/signals.
    *   Returns a dictionary containing the results of all performed analyses across all signals (though this return value might be primarily for internal tracking or potential future extensions, as results are mainly saved to files).
2.  **Parametric Data Loading (`_load_summary_data`)**:
    *   An internal helper function called by `manager` if live parametric results are not available but are needed.
    *   Identifies the expected summary filenames based on study type (`freq`, `delta`, `freq_delta`).
    *   Searches the appropriate parametric output folders specified in `output_paths`.
    *   Loads data using `utils.load_pickle` or `utils.load_df_from_excel` based on the `pickle` flag.
    *   Aggregates the loaded DataFrames into a nested dictionary (`ParametricResultsDict`), similar to the structure produced by `parametric_handler`.
    *   Returns the loaded data dictionary (or `None` if no valid data was found).

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
    classDef externalStyle fill:#ffe0b2,stroke:#ffb74d,stroke-width:1px,color:#333,rx:2,ry:2 %% Style for calls to analysis modules
    classDef fileOutputStyle fill:#e8f5e9,stroke:#a5d6a7,stroke-width:1px,stroke-dasharray: 3 3,color:#555,rx:4,ry:4 %% Dashed border for file outputs

    %% -------------------------------- Diagram Structure: High-Level results_handler.py --------------------------------

    subgraph Inputs
        direction TB
        In1[config: ConfigDict<br/><span style='font-size:0.9em; color:#444'>Global Configuration</span>]:::inputStyle
        In2[bools: BoolsDict<br/><span style='font-size:0.9em; color:#444'>Analysis Enable Flags</span>]:::inputStyle
        In3[signals: List of SignalDict<br/><span style='font-size:0.9em; color:#444'>List of Signal Data</span>]:::inputStyle
        In4[output_paths: OutputPathsDict<br/><span style='font-size:0.9em; color:#444'>Paths for Results: Optima, Nyquist, etc.</span>]:::inputStyle
        InOpt1[parametric_summary_results: ParametricResultsDict?<br/><span style='font-size:0.9em; color:#444'>Live Parametric Summaries -Optional-</span>]:::optionalInputStyle
    end

    subgraph Process
        direction TB
        P_Main[manager<br/><span style='font-size:0.9em; color:#444'>Main Results Orchestrator</span>]:::processStyle

        Step_Load[_load_summary_data<br/><span style='font-size:0.9em; color:#444'>Loads saved summaries if needed</span>]:::helperStyle
        Ext_LoadUtils[utils.load_pickle / utils.load_df_from_excel]:::externalStyle

        subgraph SignalLoop [Loop Through Each Signal]
            direction TB
            Step_Optima[Run Optima Search & Simulation<br/><span style='font-size:0.9em; color:#444'>Calls optima.perform_optima_study</span>]:::helperStyle
            Call_Opt[optima.perform_optima_study]:::externalStyle

            Step_Nyquist[Run Nyquist Analysis<br/><span style='font-size:0.9em; color:#444'>Calls nyquist.perform_nyquist_study</span>]:::helperStyle
            Call_Nyq[nyquist.perform_nyquist_study]:::externalStyle

            Step_Fourier[Run Fourier Analysis<br/><span style='font-size:0.9em; color:#444'>Calls fourier.perform_fourier_study</span>]:::helperStyle
            Call_Fou[fourier.perform_fourier_study]:::externalStyle
        end
    end


    subgraph Outputs
        O_Files[Analysis Result Files<br/><span style='font-size:0.9em; color:#444'>Generated in output_paths folders:<br/>- Optima summaries/plots/sim_data<br/>- Nyquist summaries/plots<br/>- Fourier summaries/plots</span>]:::fileOutputStyle
        O_Logs[Log Messages<br/><span style='font-size:0.9em; color:#444'>Progress and status updates</span>]:::outputStyle
        O_Return[Analysis Results Dict<br/><span style='font-size:0.9em; color:#444'>Internal dictionary containing results</span>]:::outputStyle
    end

    %% -------------------------------- Connections --------------------------------
    Inputs --> P_Main

    P_Main -- Checks Inputs, Calls if needed --> Step_Load
    Step_Load -- Calls --> Ext_LoadUtils
    Step_Load -- Returns Loaded Summaries --> P_Main

    P_Main -- Initiates --> SignalLoop

    %% Inside the loop - show dependency chain & calls
    %% Parametric data (live or loaded) is an input to the loop/Optima step
    P_Main -- Parametric Summaries --> Step_Optima
    SignalLoop --> Step_Optima
    Step_Optima -- Calls --> Call_Opt
    Call_Opt -- Optimal Params & Sim Results --> Step_Optima

    Step_Optima -- Optimal Sim Results --> Step_Nyquist
    Step_Nyquist -- Calls --> Call_Nyq

    Step_Optima -- Optimal Sim Results --> Step_Fourier
    Step_Fourier -- Calls --> Call_Fou

    %% Connecting Process Steps to File Outputs (Conceptually)
    %% The actual file creation happens inside the called analysis modules
    Call_Opt -- Generates --> O_Files
    Call_Nyq -- Generates --> O_Files
    Call_Fou -- Generates --> O_Files

    %% Logging output
    P_Main -- Generates --> O_Logs
    SignalLoop -- Generates --> O_Logs

    %% Return Value
    P_Main -- Returns --> O_Return
````