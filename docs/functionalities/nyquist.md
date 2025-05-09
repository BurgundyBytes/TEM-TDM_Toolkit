# Nyquist Analysis Module (`src/analysis/nyquist.py`)

## Overview

This module performs a comparative analysis based on the principles of the Nyquist-Shannon sampling theorem, but adapted for comparison against the ASDM's performance. It samples the original signal at `N` evenly spaced indices, reconstructs the signal using cubic spline interpolation, and calculates reconstruction error metrics for various values of `N`. The range of `N` typically goes from a minimum required for reconstruction (Nyquist-Shannon sampling theorem) up to the number of spikes generated by the ASDM in the optimal simulation, allowing for a direct comparison of reconstruction quality for a similar number of data points (samples vs. spikes).

## Functionality

1.  **Main Orchestrator (`perform_nyquist_study`)**:
    *   The primary public function called by the `results_handler`.
    *   Takes the original signal data, results from the optimal ASDM simulation (especially `N_spikes`), the output path, configuration flags, and global config as input.
    *   Calls `_determine_n_range` to establish the range of sample points `N` to test (typically from a minimum up to `N_spikes`).
    *   **Looping:** Iterates through each value of `N` in the determined range.
        *   Calls `_sample_signal_by_index` to get `N` uniformly spaced samples (t, u) from the original signal.
        *   Calls `_reconstruct_with_spline` to interpolate the sampled data back onto the original time vector `t_orig` using cubic splines.
        *   Calls `metrics.calculate_nyquist_metrics` to compute error metrics (Max Error, Median Error, RMSE, NRMSE_std) between the original signal and the spline-reconstructed signal for the current `N`.
        *   Optionally calls plotting functions (`plotting.plot_nyquist_reconstruction`, `plotting.plot_nyquist_error_profile`) to visualize the reconstruction and error for the current `N`.
    *   **Aggregation & Saving:** Collects the metrics for all tested `N` values into a pandas DataFrame.
    *   Saves the summary DataFrame to Excel and optionally Pickle using `utils.store_df_to_excel` and `utils.save_pickle`.
    *   Optionally calls `plotting.plot_nyquist_summary` to generate a plot of error metrics vs. `N`.
    *   Returns a dictionary containing the summary DataFrame and other relevant results like the reconstructed signal at `N = N_spikes`.
2.  **N Range Determination (`_determine_n_range`)**:
    *   Helper function to calculate the `range` object for the number of sample points `N`.
    *   Uses the signal's `freq_max` to determine a minimum number of points (e.g., `2 * freq_max`) and goes up to the provided `n_samples_max` (typically `N_spikes`).
    *   Ensures the minimum is at least `MIN_POINTS_FOR_CUBIC`.
3.  **Index-Based Sampling (`_sample_signal_by_index`)**:
    *   Helper function to perform uniform sampling based on array *indices* rather than time.
    *   Calculates `N` evenly spaced integer indices between 0 and the length of the original signal minus 1.
    *   Returns the time and signal values corresponding to these selected indices.
4.  **Spline Reconstruction (`_reconstruct_with_spline`)**:
    *   Helper function to reconstruct the signal using `scipy.interpolate.CubicSpline`.
    *   Takes the sampled time (`t_sampled`), sampled signal (`u_sampled`), and the original full time vector (`t_target`) as input.
    *   Handles potential issues with non-unique sample times before fitting the spline.
    *   Evaluates the fitted spline at all points in `t_target`.
    *   Returns the fully reconstructed signal vector (`u_reconstructed`).

## Visualization: module level
````mermaid
%%{ init: { 'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'primaryTextColor': '#333', 'lineColor': '#666', 'fontSize': '14px' } } }%%
graph TD

    %% -------------------------------- Styles (Simplified Template) --------------------------------
    classDef inputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,color:#333,rx:4,ry:4
    classDef processStyle fill:#fff,stroke:#888,stroke-width:1.5px,color:#000,font-weight:bold,rx:4,ry:4
    classDef outputStyle fill:#e8f5e9,stroke:#a5d6a7,stroke-width:1px,color:#333,rx:4,ry:4
    classDef optionalInputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,stroke-dasharray: 4 4,color:#333,rx:4,ry:4
    classDef helperStyle fill:#f3e5f5,stroke:#ba68c8,stroke-width:1px,color:#333,rx:2,ry:2
    classDef externalStyle fill:#ffe0b2,stroke:#ffb74d,stroke-width:1px,color:#333,rx:2,ry:2
    classDef fileOutputStyle fill:#e8f5e9,stroke:#a5d6a7,stroke-width:1px,stroke-dasharray: 3 3,color:#555,rx:4,ry:4

    %% -------------------------------- Diagram Structure: High-Level nyquist.py (Simplified) --------------------------------

    subgraph Inputs [Module Inputs]
        direction TB
        In1[signal_data: SignalDict]:::inputStyle
        In2[optimal_params: OptimalParamsDict]:::inputStyle
        In3[optimal_sim_results: OptimalSimResultDict]:::inputStyle
        In4[nyquist_path: str]:::inputStyle
        In5[bools: BoolsDict]:::inputStyle
        InOpt1[config: ConfigDict?]:::optionalInputStyle
    end

    subgraph Processing [Nyquist Study Execution Flow]
        direction TB
        P_Main[perform_nyquist_study]:::processStyle
        H_Range[_determine_n_range]:::helperStyle
        Loop{Loop over N points};
        H_Sample[_sample_signal_by_index]:::helperStyle
        H_Spline[_reconstruct_with_spline]:::helperStyle
        Ext_Metrics[metrics.calculate_nyquist_metrics]:::externalStyle
        Agg[Aggregate Results]:::processStyle
    end

     subgraph External_Modules ["Called Modules"]
         direction TB
         Ext_Plotting[plotting.*]:::externalStyle
         Ext_Utils[utils.store/save]:::externalStyle
     end

    subgraph Outputs [Module Outputs & Side Effects]
        direction TB
        O_ResultDict[nyquist_sim_results: Dict<br/><span style='font-size:0.9em; color:#444'>Summary Dict</span>]:::outputStyle
        O_Files[Output Files<br/><span style='font-size:0.9em; color:#444'>Plots, Summary Excel/Pickle</span>]:::fileOutputStyle
        OutLogs[Log Messages]:::outputStyle
    end

    %% -------------------------------- Connections --------------------------------
    Inputs --> P_Main

    P_Main -- Calls --> H_Range
    H_Range -- Returns N Range --> P_Main

    P_Main -- Initiates --> Loop

    %% Loop Internals
    Loop -- Calls --> H_Sample
    H_Sample -- Sampled Data --> H_Spline
    H_Spline -- Reconstructed Data --> Ext_Metrics
    Ext_Metrics -- Metrics --> Loop
    Loop -- Calls (Opt) --> Ext_Plotting

    Loop -- Completed, Metrics List --> Agg
    Agg -- Summary DataFrame --> Ext_Utils
    Agg -- Summary DataFrame --> Ext_Plotting
    Agg -- Returns Dict --> P_Main

    %% Outputs Generation
    Ext_Plotting -- Generates --> O_Files
    Ext_Utils -- Creates --> O_Files
    P_Main -- Returns --> O_ResultDict
    P_Main -- Generates --> OutLogs
    Loop -- Generates --> OutLogs
````

## Visualization: function level
````python
perform_nyquist_study()
````
Main orchestrator for the Nyquist analysis for a single signal.
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

    subgraph Inputs
        direction LR
        In1[signal_data: SignalDict]:::inputStyle
        In2[optimal_params: OptimalParamsDict]:::inputStyle
        In3[optimal_sim_results: OptimalSimResultDict?]:::optionalInputStyle
        In4[nyquist_path: str]:::inputStyle
        In5[bools: BoolsDict]:::inputStyle
        InOpt1[config: ConfigDict?]:::optionalInputStyle
    end

    subgraph Process
        P[perform_nyquist_study<br/><span style='font-size:0.9em; color:#444'>Orchestrates Nyquist analysis:<br/>N range, loops N,<br/>calls helpers & externals,<br/>handles plotting & saving</span>]:::processStyle
        Call_Range[_determine_n_range]:::helperStyle
        Loop{Loop over N};
        Call_Sample[_sample_signal_by_index]:::helperStyle
        Call_Spline[_reconstruct_with_spline]:::helperStyle
        Call_Metrics[metrics.calculate_nyquist_metrics]:::externalStyle
        Call_PlotLoop[plotting.*, per N]:::externalStyle
        Aggregate[Aggregate Results];
        Call_Save[utils.store/save]:::externalStyle
        Call_PlotSum[plotting.plot_nyquist_summary]:::externalStyle
    end

    subgraph Outputs
        O[nyquist_sim_results: Dict?<br/><span style='font-size:0.9em; color:#444'>Summary Dict or None</span>]:::outputStyle
        %% Note: Side effects (plots, excel, pickle files) are also outputs
    end

    %% Connections
    Inputs --> P
    P -- Calls --> Call_Range
    P -- Uses Range, Initiates --> Loop
    Loop -- Calls --> Call_Sample
    Loop -- Calls --> Call_Spline
    Loop -- Calls --> Call_Metrics
    Loop -- Calls (Opt) --> Call_PlotLoop
    Loop -- Completed --> Aggregate
    Aggregate -- Calls --> Call_Save
    Aggregate -- Calls (Opt) --> Call_PlotSum
    Aggregate -- Returns Dict --> P
    P -- Returns --> O
````

````python
_determine_n_range()
````
Calculates the range of N values to iterate over.
````mermaid
%%{ init: { 'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'primaryTextColor': '#333', 'lineColor': '#666', 'fontSize': '14px' } } }%%
graph LR
    %% Style Definitions ...
    classDef inputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,color:#333,rx:4,ry:4
    classDef processStyle fill:#fff,stroke:#888,stroke-width:1.5px,color:#000,font-weight:bold,rx:4,ry:4
    classDef outputStyle fill:#e8f5e9,stroke:#a5d6a7,stroke-width:1px,color:#333,rx:4,ry:4
    classDef optionalInputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,stroke-dasharray: 4 4,color:#333,rx:4,ry:4

    subgraph Inputs
        direction LR
        In1[signal_data: SignalDict<br/><span style='font-size:0.9em; color:#444'>Needs freq_max</span>]:::inputStyle
        In2[n_samples_max: int<br/><span style='font-size:0.9em; color:#444'>Max N</span>]:::inputStyle
    end

    subgraph Process
        P[_determine_n_range<br/><span style='font-size:0.9em; color:#444'>Calculates N range based<br/>on freq_max and max value,<br/>ensuring minimum points</span>]:::processStyle
    end

    subgraph Outputs
        O[n_range: range?<br/><span style='font-size:0.9em; color:#444'>Range object for N values,<br/>or None if invalid</span>]:::outputStyle
    end

    %% Connections
    Inputs --> P
    P -- Resulting Range --> O
````

````python
_reconstruct_with_spline()
````
Reconstructs the signal from samples using cubic spline interpolation.
````mermaid
%%{ init: { 'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'primaryTextColor': '#333', 'lineColor': '#666', 'fontSize': '14px' } } }%%
graph LR
    %% Style Definitions ...
    classDef inputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,color:#333,rx:4,ry:4
    classDef processStyle fill:#fff,stroke:#888,stroke-width:1.5px,color:#000,font-weight:bold,rx:4,ry:4
    classDef outputStyle fill:#e8f5e9,stroke:#a5d6a7,stroke-width:1px,color:#333,rx:4,ry:4
    classDef externalStyle fill:#ffe0b2,stroke:#ffb74d,stroke-width:1px,color:#333,rx:2,ry:2


    subgraph Inputs
        direction LR
        In1[t_sampled: np.ndarray]:::inputStyle
        In2[u_sampled: np.ndarray]:::inputStyle
        In3[t_target: np.ndarray]:::inputStyle
        In4[n_id_log: str]:::inputStyle
    end

    subgraph Process
        P[_reconstruct_with_spline<br/><span style='font-size:0.9em; color:#444'>Creates Cubic Spline & evaluates</span>]:::processStyle
        Ext[scipy.interpolate.CubicSpline]:::externalStyle
    end

    subgraph Outputs
        O[u_reconstructed: np.ndarray?<br/><span style='font-size:0.9em; color:#444'>Reconstructed signal or None</span>]:::outputStyle
    end

    %% Connections
    Inputs --> P
    P -- Calls --> Ext
    Ext -- Spline Object --> P
    P -- Reconstructed Signal --> O
````

