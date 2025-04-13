# Optimal Conditions Analysis Module (`src/analysis/optima.py`)

## Overview

This module is dedicated to analyzing the results of parametric studies (`freq`, `delta`, `freq_delta`) to identify the optimal operating parameters (specifically `fs` - sampling frequency and `d_norm` - normalized threshold) for the ASDM encoder/decoder based on specified criteria (minimizing error within time/error thresholds). It then performs a simulation using these optimal parameters and saves the detailed results and relevant plots.

## Functionality

1.  **Main Orchestrator (`perform_optima_study`)**:
    *   The primary public function called by the `results_handler`.
    *   Takes signal data, parametric study results for that signal, the output path, configuration flags, and the global config as input.
    *   Calls `_find_signal_optima_step` to determine the best `fs` and `d_norm`.
    *   If optimal parameters are found, it calls `_simulate_and_save_optimal_step` to run the ASDM simulation with these parameters.
    *   Returns both the optimal parameters found and the results of the optimal simulation (or `None` if steps fail).
2.  **Signal-Specific Optima Finding (`_find_signal_optima_step`)**:
    *   A helper function called by `perform_optima_study`.
    *   Extracts the relevant DataFrames (`df_freq`, `df_delta`, `df_2d`) from the provided study results dictionary for the specific signal.
    *   Retrieves error and time thresholds from the configuration.
    *   Calls `_find_optimal_conditions` to perform the core logic of finding the best parameters based on the available data and thresholds.
    *   Returns the dictionary of optimal parameters (`OptimalParamsDict`) or `None`.
3.  **Core Optima Selection (`_find_optimal_conditions`)**:
    *   Takes the summary DataFrames from the parametric studies and thresholds as input.
    *   Implements a prioritized search:
        *   **Priority 1:** Analyzes the 2D sweep (`df_2d`) using `_filter_and_select`.
        *   **Priority 2:** If no suitable candidate is found in 2D, analyzes the 1D frequency sweep (`df_freq`), using the `Default Delta` from the config. Calls `_filter_and_select`.
        *   **Priority 3:** If still no candidate, analyzes the 1D delta sweep (`df_delta`), using the `Default Frequency` from the config. Calls `_filter_and_select`.
    *   Returns the best candidate found (as a dictionary `OptimalParamsDict` including the source study type) or `None`.
4.  **Filtering and Selection Logic (`_filter_and_select`)**:
    *   A low-level helper called by `_find_optimal_conditions`.
    *   Takes a single DataFrame and thresholds as input.
    *   Cleans the DataFrame (drops NaNs).
    *   Applies the optional `error_threshold` and `time_threshold` filters.
    *   If candidates remain, sorts them first by the error column (ascending) and then by the time column (ascending) as a tie-breaker.
    *   Selects and returns the top row (best candidate) as a pandas Series, or `None` if no candidates meet the criteria.
5.  **Optimal Simulation (`_simulate_and_save_optimal_step`)**:
    *   Takes the signal data and the *found* optimal parameters (`fs`, `d_norm`) as input.
    *   Performs a single ASDM encode/decode cycle using `asdm.asdm_encode` and `asdm.asdm_decode`.
    *   Aligns the reconstructed signal length.
    *   Calculates metrics on the stable region of the simulation using `metrics.calculate_asdm_metrics`.
    *   Saves a summary DataFrame of this optimal run (parameters + metrics) to Excel and optionally Pickle using `utils.store_df_to_excel` and `utils.save_pickle`.
    *   Saves the raw simulation results (signals, spikes, detailed metrics) to a Pickle file if enabled.
    *   Generates plots (process, spikes) for the optimal run using `plotting.plot_process` and `plotting.plot_with_spikes` if enabled.
    *   Returns a dictionary (`OptimalSimResultDict`) containing key results needed by other analysis modules (e.g., `u_rec`, `s`, `N_spikes`).

## Visualization: module level
````mermaid
%%{ init: { 'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'primaryTextColor': '#333', 'lineColor': '#666', 'fontSize': '14px' } } }%%
graph TD

    %% -------------------------------- Styles (Reusable Template Section) --------------------------------
    classDef inputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,color:#333,rx:4,ry:4
    classDef processStyle fill:#fff,stroke:#888,stroke-width:1.5px,color:#000,font-weight:bold,rx:4,ry:4
    classDef outputStyle fill:#e8f5e9,stroke:#a5d6a7,stroke-width:1px,color:#333,rx:4,ry:4
    classDef optionalInputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,stroke-dasharray: 4 4,color:#333,rx:4,ry:4
    classDef helperStyle fill:#f3e5f5,stroke:#ba68c8,stroke-width:1px,color:#333,rx:2,ry:2 %% Style for internal helpers
    classDef externalStyle fill:#ffe0b2,stroke:#ffb74d,stroke-width:1px,color:#333,rx:2,ry:2 %% Style for calls to other modules
    classDef fileOutputStyle fill:#e8f5e9,stroke:#a5d6a7,stroke-width:1px,stroke-dasharray: 3 3,color:#555,rx:4,ry:4 %% Dashed border for file outputs

    %% -------------------------------- Diagram Structure: High-Level optima.py --------------------------------

    subgraph Inputs
        direction TB
        In1[signal_data: SignalDict]:::inputStyle
        In2[results_for_this_signal: ParametricResultsDict?]:::optionalInputStyle
        In3[optima_path: str]:::inputStyle
        In4[bools: BoolsDict]:::inputStyle
        In5[config: ConfigDict?]:::optionalInputStyle
    end

    subgraph Process
        direction TB
        P_Main[perform_optima_study<br/><span style='font-size:0.9em; color:#444'>Main Orchestrator</span>]:::processStyle
        H_FindOpt[_find_signal_optima_step<br/><span style='font-size:0.9em; color:#444'>Finds best fs, d_norm</span>]:::helperStyle
        H_Simulate[_simulate_and_save_optimal_step<br/><span style='font-size:0.9em; color:#444'>Runs simulation with best params</span>]:::helperStyle
    end

    subgraph Core_Logic ["Lower-Level Helpers"]
         direction TB
         H_Select[_find_optimal_conditions<br/><span style='font-size:0.9em; color:#444'>Prioritized Selection Logic</span>]:::helperStyle
         H_Filter[_filter_and_select<br/><span style='font-size:0.9em; color:#444'>Core Filtering & Sorting</span>]:::helperStyle
    end

    subgraph External_Modules ["Called Modules"]
         direction TB
         ExtASDM[asdm: encode/decode]:::externalStyle
         ExtMetrics[metrics: calculate]:::externalStyle
         ExtPlotting[plotting: plot process/spikes]:::externalStyle
         ExtUtils[utils: save/load, stable_region]:::externalStyle
    end

    subgraph Outputs
        direction TB
        Out1[OptimalParamsDict?<br/><span style='font-size:0.9em; color:#444'>Optimal parameters found</span>]:::outputStyle
        Out2[OptimalSimResultDict?<br/><span style='font-size:0.9em; color:#444'>Results of optimal simulation</span>]:::outputStyle
        OutFiles[Output Files<br/><span style='font-size:0.9em; color:#444'>Optima Summary,<br/>Raw Data, Plots</span>]:::fileOutputStyle
        OutLogs[Log Messages]:::outputStyle
    end

    %% -------------------------------- Connections --------------------------------
    Inputs --> P_Main

    P_Main -- Calls --> H_FindOpt
    H_FindOpt -- Calls --> H_Select
    H_Select -- Calls --> H_Filter

    H_Filter -- Returns Best Row --> H_Select
    H_Select -- Returns Opt Dict --> H_FindOpt
    H_FindOpt -- Optimal Params --> P_Main

    P_Main -- If Opt Found, Calls --> H_Simulate
    H_Simulate -- Calls --> ExtASDM
    H_Simulate -- Calls --> ExtMetrics
    H_Simulate -- Calls --> ExtPlotting
    H_Simulate -- Calls --> ExtUtils

    ExtPlotting -- Generates --> OutFiles
    ExtUtils -- Generates --> OutFiles
    H_Simulate -- Returns Sim Results --> P_Main

    %% Final Outputs
    P_Main -- Returns --> Out1
    P_Main -- Returns --> Out2
    P_Main -- Generates --> OutLogs
    H_FindOpt -- Generates --> OutLogs
    H_Simulate -- Generates --> OutLogs
````
## Visualization: function level
```python
perform_optima_study()
```
The main entry point for the optima analysis workflow for a single signal.

````mermaid
    %%{ init: { 'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'primaryTextColor': '#333', 'lineColor': '#666', 'fontSize': '14px' } } }%%
graph LR
    %% Style Definitions ... (copy from above)
    classDef inputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,color:#333,rx:4,ry:4
    classDef processStyle fill:#fff,stroke:#888,stroke-width:1.5px,color:#000,font-weight:bold,rx:4,ry:4
    classDef outputStyle fill:#e8f5e9,stroke:#a5d6a7,stroke-width:1px,color:#333,rx:4,ry:4
    classDef optionalInputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,stroke-dasharray: 4 4,color:#333,rx:4,ry:4
    classDef helperStyle fill:#f3e5f5,stroke:#ba68c8,stroke-width:1px,color:#333,rx:2,ry:2
    classDef externalStyle fill:#ffe0b2,stroke:#ffb74d,stroke-width:1px,color:#333,rx:2,ry:2
    classDef fileOutputStyle fill:#e8f5e9,stroke:#a5d6a7,stroke-width:1px,stroke-dasharray: 3 3,color:#555,rx:4,ry:4
    classDef collectorStyle fill:none,stroke:none

    subgraph Inputs
        direction LR
        In1[signal_data: SignalDict]:::inputStyle
        In2[results_for_this_signal: ParametricResultsDict?]:::optionalInputStyle
        In3[optima_path: str]:::inputStyle
        In4[bools: BoolsDict]:::inputStyle
        In5[config: ConfigDict?]:::optionalInputStyle
    end

    subgraph Proces
        direction TB
        P[perform_optima_study]:::processStyle
        Call_Find[_find_signal_optima_step]:::helperStyle
        Call_Sim[_simulate_and_save_optimal_step]:::helperStyle
    end

    subgraph Outputs
        direction LR
        Out1[OptimalParamsDict?]:::outputStyle
        Out2[OptimalSimResultDict?]:::outputStyle
    end

    %% Connections
    Inputs--> P
    P -- Calls --> Call_Find
    Call_Find -- Returns Optimal Params --> P
    P -- If Opt Params Found, Calls --> Call_Sim
    Call_Sim -- Returns Sim Results --> P
    P -- Returns Tuple --> Outputs
````

```python
_find_signal_optima_step()
```
Helper to coordinate finding the optimal parameters for a single signal.

````mermaid
%%{ init: { 'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'primaryTextColor': '#333', 'lineColor': '#666', 'fontSize': '14px' } } }%%
graph LR
    %% Style Definitions ...
    classDef inputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,color:#333,rx:4,ry:4
    classDef processStyle fill:#fff,stroke:#888,stroke-width:1.5px,color:#000,font-weight:bold,rx:4,ry:4
    classDef outputStyle fill:#e8f5e9,stroke:#a5d6a7,stroke-width:1px,color:#333,rx:4,ry:4
    classDef optionalInputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,stroke-dasharray: 4 4,color:#333,rx:4,ry:4
    classDef helperStyle fill:#f3e5f5,stroke:#ba68c8,stroke-width:1px,color:#333,rx:2,ry:2
    classDef externalStyle fill:#ffe0b2,stroke:#ffb74d,stroke-width:1px,color:#333,rx:2,ry:2
    classDef fileOutputStyle fill:#e8f5e9,stroke:#a5d6a7,stroke-width:1px,stroke-dasharray: 3 3,color:#555,rx:4,ry:4
    classDef collectorStyle fill:none,stroke:none

    subgraph Inputs
        direction LR
        In1[study_results_for_signal: ParametricResultsDict?]:::optionalInputStyle
        In2[config: ConfigDict]:::inputStyle
        In3[signal_name: str]:::inputStyle
    end

    subgraph Process
        P[_find_signal_optima_step]:::processStyle
        Call_FindCond[_find_optimal_conditions]:::helperStyle
    end

    subgraph Outputs
        Out1[OptimalParamsDict?]:::outputStyle
    end

    %% Connections
    Inputs --> P
    P -- Extracts DFs, Calls --> Call_FindCond
    Call_FindCond -- Returns Opt Params --> P
    P -- Returns --> Out1
````

```python
_find_optimal_conditions()
```
Contains the core logic for selecting the best parameters based on prioritized study results.

````mermaid
%%{ init: { 'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'primaryTextColor': '#333', 'lineColor': '#666', 'fontSize': '14px' } } }%%
graph LR
    %% Style Definitions ...
    classDef inputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,color:#333,rx:4,ry:4
    classDef processStyle fill:#fff,stroke:#888,stroke-width:1.5px,color:#000,font-weight:bold,rx:4,ry:4
    classDef outputStyle fill:#e8f5e9,stroke:#a5d6a7,stroke-width:1px,color:#333,rx:4,ry:4
    classDef optionalInputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,stroke-dasharray: 4 4,color:#333,rx:4,ry:4
    classDef helperStyle fill:#f3e5f5,stroke:#ba68c8,stroke-width:1px,color:#333,rx:2,ry:2
    classDef externalStyle fill:#ffe0b2,stroke:#ffb74d,stroke-width:1px,color:#333,rx:2,ry:2
    classDef fileOutputStyle fill:#e8f5e9,stroke:#a5d6a7,stroke-width:1px,stroke-dasharray: 3 3,color:#555,rx:4,ry:4
    classDef collectorStyle fill:none,stroke:none

    subgraph Inputs
        direction LR
        In1[df_freq: pd.DataFrame?]:::optionalInputStyle
        In2[df_delta: pd.DataFrame?]:::optionalInputStyle
        In3[df_2d: pd.DataFrame?]:::optionalInputStyle
        In4[error_threshold: float?]:::optionalInputStyle
        In5[time_threshold: float?]:::optionalInputStyle
        In6[config: ConfigDict?]:::optionalInputStyle
    end

    subgraph Process
        P[_find_optimal_conditions<br/><span style='font-size:0.9em; color:#444'>Prioritized Search:<br/>1. 2D<br/>2. Freq<br/>3. Delta</span>]:::processStyle
        Call_Filter[_filter_and_select]:::helperStyle
    end

    subgraph Output
        Out1[OptimalParamsDict?<br/><span style='font-size:0.9em; color:#444'>Best params found or None</span>]:::outputStyle
    end

    %% Connections
    Inputs --> P
    P -- Uses DFs & Thresholds, Calls --> Call_Filter
    Call_Filter -- Returns Candidate Row --> P
    P -- Formats Best Candidate --> Out1
````

```python
_filter_and_select()
```
Core filtering and sorting logic applied to a single DataFrame.

````mermaid
%%{ init: { 'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'primaryTextColor': '#333', 'lineColor': '#666', 'fontSize': '14px' } } }%%
graph LR
    %% Style Definitions ...
    classDef inputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,color:#333,rx:4,ry:4
    classDef processStyle fill:#fff,stroke:#888,stroke-width:1.5px,color:#000,font-weight:bold,rx:4,ry:4
    classDef outputStyle fill:#e8f5e9,stroke:#a5d6a7,stroke-width:1px,color:#333,rx:4,ry:4
    classDef optionalInputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,stroke-dasharray: 4 4,color:#333,rx:4,ry:4
    classDef helperStyle fill:#f3e5f5,stroke:#ba68c8,stroke-width:1px,color:#333,rx:2,ry:2
    classDef externalStyle fill:#ffe0b2,stroke:#ffb74d,stroke-width:1px,color:#333,rx:2,ry:2
    classDef fileOutputStyle fill:#e8f5e9,stroke:#a5d6a7,stroke-width:1px,stroke-dasharray: 3 3,color:#555,rx:4,ry:4
    classDef collectorStyle fill:none,stroke:none


    subgraph Inputs
        direction LR
        In1[df: pd.DataFrame]:::inputStyle
        In2[id_cols: List of str]:::inputStyle
        InOpt1[error_col: str]:::optionalInputStyle
        InOpt2[time_col: str]:::optionalInputStyle
        InOpt3[error_threshold: float?]:::optionalInputStyle
        InOpt4[time_threshold: float?]:::optionalInputStyle
    end

    subgraph Process
        P[_filter_and_select<br/><span style='font-size:0.9em; color:#444'>1. Clean DF<br/>2. Apply Thresholds<br/>3. Sort: Error, Time<br/>4. Select Top Row</span>]:::processStyle
    end

    subgraph Outputs
        O[best_row: pd.Series?<br/><span style='font-size:0.9em; color:#444'>Best candidate row or None</span>]:::outputStyle
    end

    %% Connections
    Inputs --> P
    P -- Returns --> O
````

```python
_simulate_and_save_optimal_step()
```
Runs the simulation with optimal parameters and handles outputs.

````mermaid
%%{ init: { 'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'primaryTextColor': '#333', 'lineColor': '#666', 'fontSize': '14px' } } }%%
graph LR
    %% Style Definitions ...
    classDef inputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,color:#333,rx:4,ry:4
    classDef processStyle fill:#fff,stroke:#888,stroke-width:1.5px,color:#000,font-weight:bold,rx:4,ry:4
    classDef outputStyle fill:#e8f5e9,stroke:#a5d6a7,stroke-width:1px,color:#333,rx:4,ry:4
    classDef optionalInputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,stroke-dasharray: 4 4,color:#333,rx:4,ry:4
    classDef helperStyle fill:#f3e5f5,stroke:#ba68c8,stroke-width:1px,color:#333,rx:2,ry:2
    classDef externalStyle fill:#ffe0b2,stroke:#ffb74d,stroke-width:1px,color:#333,rx:2,ry:2
    classDef fileOutputStyle fill:#e8f5e9,stroke:#a5d6a7,stroke-width:1px,stroke-dasharray: 3 3,color:#555,rx:4,ry:4
    classDef collectorStyle fill:none,stroke:none

    subgraph Inputs
        direction LR
        In1[signal_data: SignalDict]:::inputStyle
        In2[optimal_params: OptimalParamsDict]:::inputStyle
        In3[optimal_output_path: str?]:::optionalInputStyle
        In4[bools: BoolsDict]:::inputStyle
        In5[config: ConfigDict]:::inputStyle
    end

    subgraph Process
        P[_simulate_and_save_optimal_step<br/><span style='font-size:0.9em; color:#444'>1. Encode/Decode<br/>2. Calc Metrics<br/>3. Save Summary<br/>4. Save Raw<br/>5. Plot</span>]:::processStyle
    end

     subgraph Called_Modules ["Called Modules"]
         direction TB
         ExtASDM[asdm: encode/decode]:::externalStyle
         ExtMetrics[metrics: calculate]:::externalStyle
         ExtPlotting[plotting: plot process/spikes]:::externalStyle
         ExtUtils[utils: save/load, stable_region]:::externalStyle
    end

    subgraph Outputs
        direction TB
        Out1[OptimalSimResultDict?<br/><span style='font-size:0.9em; color:#444'>Key Sim Results or None</span>]:::outputStyle
        OutFiles[Output Files<br/><span style='font-size:0.9em; color:#444'>Optima Summary,<br/>Raw Data, Plots</span>]:::fileOutputStyle
    end

    %% Connections
    Inputs --> P
    P -- Calls --> ExtASDM
    P -- Calls --> ExtMetrics
    P -- Calls --> ExtPlotting
    P -- Calls --> ExtUtils

    ExtPlotting -- Generates --> OutFiles
    ExtUtils -- Generates --> OutFiles

    P -- Returns --> Out1
````