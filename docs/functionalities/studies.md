# Parametric Studies Module (`src/parametric/studies.py`)

## Overview

This module contains the core logic for executing parametric studies. It systematically runs the ASDM encode/decode simulation (`_run_single_simulation`) across ranges of specified parameters, now focusing on the **encoder bias (`b`)** and the **normalized threshold (`d_norm`)**. It calculates performance metrics for each simulation run, handles optional logging and plotting for individual runs, saves raw simulation data if requested, and aggregates the metrics into summary DataFrames for each study type.

## Functionality

1.  **Single Simulation Runner (`_run_single_simulation`)**:
    *   The workhorse function that performs one complete cycle:
        *   **Input Scaling:** Normalizes the input signal `u` by the current bias `b` (`u/b`).
        *   **Encoding:** Calls `asdm.asdm_encode` with the normalized input (`u/b`), the specified `d_norm`, and the simulation time step (`dte = 1/fs`).
        *   **Decoding:** Calls `asdm.asdm_decode` using the generated spikes (`s`), duration `dur`, `dte`, and estimated bandwidth (`bw`). Note: The decoder implementation used here is assumed to be the δ-insensitive one, operating on the normalized principles.
        *   **Alignment:** Ensures the reconstructed signal (`u_rec`) matches the length of the original time vector (`t`).
        *   **Stable Region Extraction:** Selects the portion of the original signal (`u`), reconstructed signal (`u_rec`), and time vector (`t`) corresponding to the pre-calculated stable region indices (`start_idx`, `end_idx`).
        *   **Metrics Calculation:** Calls `metrics.calculate_asdm_metrics` using the *normalized* stable original signal (`u_stable/b`) and the stable reconstructed signal (`u_rec_stable`).
        *   **Adds Parameters:** Includes the `b` and `d_norm` used for the run into the results dictionary.
        *   **Optional Outputs (if flags enabled):** Logs metrics, generates plots, saves raw data (pickle).
    *   Returns a dictionary (`MetricsDict`) with metrics and parameters for this single run, or `None` if failed.
2.  **Bias Study (`parametric_bias`)**:
    *   Takes an array of bias values (`bias`) and a fixed normalized threshold (`d_norm`) as input.
    *   Initializes the study log file.
    *   Iterates through each `b` in the `bias` array (ensuring `b > c` should be handled by the caller or config).
    *   Calls `_run_single_simulation` for each `b`, passing the fixed `d_norm`.
    *   Collects results and aggregates them into `df_bias`.
    *   Returns the summary DataFrame (or `None`).
3.  **Delta Study (`parametric_delta`)**:
    *   Takes an array of normalized thresholds (`deltas`) and a fixed bias (`b`) as input.
    *   Initializes the study log file.
    *   Iterates through each `d_norm` in `deltas`.
    *   Calls `_run_single_simulation` for each `d_norm`, passing the fixed `b`.
    *   Collects results and aggregates them into `df_delta`.
    *   Returns the summary DataFrame (or `None`).
4.  **Biparametric Study (`parametric_bias_delta`)**:
    *   Takes arrays of bias values (`bias`) and thresholds (`deltas`) as input.
    *   Initializes the study log file.
    *   Uses nested loops to iterate through *all combinations* of `b` and `d_norm`.
    *   Calls `_run_single_simulation` for each (`b`, `d_norm`) pair.
    *   Collects results and aggregates them into `df_2d`.
    *   Returns the summary DataFrame (or `None`).

## Visualization: module level
````mermaid
%%{ init: { 'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'primaryTextColor': '#333', 'lineColor': '#666', 'fontSize': '14px' } } }%%
graph LR

    %% -------------------------------- Styles (Reusable Template Section) --------------------------------
    classDef inputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,color:#333,rx:4,ry:4 %% rx/ry for slightly rounded corners
    classDef processStyle fill:#fff,stroke:#888,stroke-width:1.5px,color:#000,font-weight:bold,rx:4,ry:4
    classDef outputStyle fill:#e8f5e9,stroke:#a5d6a7,stroke-width:1px,color:#333,rx:4,ry:4
    %% Optional input style: Same fill as input, but dashed border for clear visual cue
    classDef optionalInputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,stroke-dasharray: 4 4,color:#333,rx:4,ry:4
    %% Invisible style for grouping nodes without visual clutter
    classDef collectorStyle fill:none,stroke:none
    classDef helperStyle fill:#f3e5f5,stroke:#ba68c8,stroke-width:1px,color:#333,rx:2,ry:2 %% Style for internal helpers/steps
    classDef externalStyle fill:#ffe0b2,stroke:#ffb74d,stroke-width:1px,color:#333,rx:2,ry:2 %% Style for calls to other modules
    classDef fileOutputStyle fill:#e8f5e9,stroke:#a5d6a7,stroke-width:1px,stroke-dasharray: 3 3,color:#555,rx:4,ry:4 %% Dashed border for file outputs


    %% -------------------------------- Diagram Structure: Parametric Study Module --------------------------------

    %% --- Inputs ---
    subgraph Inputs
        direction TB
        InShared[Shared Parameters<br/><span style='font-size:0.9em; color:#444'>u, t, dur, fs, dte, freq_max, start_idx, end_idx,<br/>log_filename, output_dir, signal_name,<br/>log_study, plot_study, save_raw</span>]:::inputStyle
        InBiasRange[bias: np.ndarray<br/><span style='font-size:0.9em; color:#444'>for PB, PBD</span>]:::inputStyle
        InDeltaRange[deltas: np.ndarray<br/><span style='font-size:0.9em; color:#444'>for PD, PBD</span>]:::inputStyle
        InBias[b: float<br/><span style='font-size:0.9em; color:#444'>for PD</span>]:::inputStyle
        InDnorm[d_norm: float<br/><span style='font-size:0.9em; color:#444'>for PB</span>]:::inputStyle
        %% Invisible Collector for inputs
        InputCollector( ):::collectorStyle
    end

    %% --- Processes (Parametric Study Functions) ---
    subgraph Process
        direction TB
        P_Bias[parametric_bias]:::processStyle
        P_Delta[parametric_delta]:::processStyle
        P_BiasDelta[parametric_bias_delta]:::processStyle
    end

    %% --- Helper Process (Core Simulation) ---
    subgraph Core_Simulation_Helper ["Core Simulation Helper"]
        P_Single[_run_single_simulation]:::helperStyle
    end

     %% --- External Dependencies ---
    subgraph Called_Modules ["Called Modules"]
        direction TB
        ExtASDM[asdm: encode/decode]:::externalStyle
        ExtMetrics[metrics: calculate]:::externalStyle
        ExtLogging[project_logging: log run/header]:::externalStyle
        ExtPlotting[plotting: plot process/spikes]:::externalStyle
        ExtUtils[utils: save_pickle, estimate_bw]:::externalStyle
        %% Invisible Collector for dependencies
        DepCollector( ):::collectorStyle
    end

    %% --- Outputs ---
    subgraph Outputs
       direction TB
       O_DF[pd.DataFrame <br/><span style='font-size:0.9em; color:#444'>Aggregated results from<br/><b>parametric_* functions</b></span>]:::outputStyle
       O_Files[Output Files<br/><span style='font-size:0.9em; color:#444'>Logs, Plots,<br/>Raw Data Pickles</span>]:::fileOutputStyle
       %% Invisible Collector for outputs
       OutputCollector( ):::collectorStyle
    end


    %% -------------------------------- Connections --------------------------------
    %% Inputs flow to the invisible input collector
    InShared --> InputCollector; InBiasRange --> InputCollector; InDeltaRange --> InputCollector;
    InBias --> InputCollector; InDnorm --> InputCollector;

    %% Input Collector connects to the specific study functions
    InputCollector -- Specific Inputs --> P_Bias
    InputCollector -- Specific Inputs --> P_Delta
    InputCollector -- Specific Inputs --> P_BiasDelta

    %% Study functions call the core simulation helper within their loops
    P_Bias -- Calls (in loop) --> P_Single
    P_Delta -- Calls (in loop) --> P_Single
    P_BiasDelta -- Calls (in loop) --> P_Single

    %% Core helper calls external dependencies (connected via collector)
    ExtASDM --> DepCollector; ExtMetrics --> DepCollector; ExtLogging --> DepCollector;
    ExtPlotting --> DepCollector; ExtUtils --> DepCollector;
    P_Single -- Calls --> DepCollector

    %% Core helper generates side effect outputs (connected via collector)
    DepCollector -- Generates --> O_Files

    %% Metrics Dict is returned by helper and collected by study functions
    P_Single -- Returns MetricsDict --> Process

    %% Study functions aggregate metrics and return DataFrame
    P_Bias -- Aggregates & Returns --> O_DF
    P_Delta -- Aggregates & Returns --> O_DF
    P_BiasDelta -- Aggregates & Returns --> O_DF
````

## Visualization: funtion level

````python
parametric_bias()
````
Runs a parametric sweep over a range of encoder bias values (`b`) while keeping the normalized threshold (`d_norm`) constant.
````mermaid
%%{ init: { 'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'primaryTextColor': '#333', 'lineColor': '#666', 'fontSize': '14px' } } }%%
graph LR
    %% -------------------------------- Styles (Reusable Template Section) --------------------------------
    classDef inputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,color:#333,rx:4,ry:4 %% rx/ry for slightly rounded corners
    classDef processStyle fill:#fff,stroke:#888,stroke-width:1.5px,color:#000,font-weight:bold,rx:4,ry:4
    classDef outputStyle fill:#e8f5e9,stroke:#a5d6a7,stroke-width:1px,color:#333,rx:4,ry:4
    %% Optional input style: Same fill as input, but dashed border for clear visual cue
    classDef optionalInputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,stroke-dasharray: 4 4,color:#333,rx:4,ry:4
    %% Invisible style for grouping nodes without visual clutter
    classDef collectorStyle fill:none,stroke:none

    %% -------------------------------- Diagram Structure: parametric_bias --------------------------------
    subgraph Input
        direction TB %% Arrange inputs vertically

        %% Compulsory Input Nodes
        In1[u: np.ndarray<br/><span style='font-size:0.9em; color:#444'>Input Signal</span>]:::inputStyle
        In2[t: np.ndarray<br/><span style='font-size:0.9em; color:#444'>Signal Time Vector</span>]:::inputStyle
        In3[dur: float<br/><span style='font-size:0.9em; color:#444'>Signal Duration</span>]:::inputStyle
        In4[fs: float<br/><span style='font-size:0.9em; color:#444'>Fixed Sampling Frequency</span>]:::inputStyle
        In5[d_norm: float<br/><span style='font-size:0.9em; color:#444'>Fixed Norm. Threshold</span>]:::inputStyle
        In6[bias: np.ndarray<br/><span style='font-size:0.9em; color:#444'>Bias values to Test</span>]:::inputStyle
        In7[dte: float<br/><span style='font-size:0.9em; color:#444'>Encoding Sample Time</span>]:::inputStyle
        In8[freq_max: float<br/><span style='font-size:0.9em; color:#444'>Max Signal Freq: for BW</span>]:::inputStyle
        In9[start_idx: int<br/><span style='font-size:0.9em; color:#444'>Analysis Start Index</span>]:::inputStyle
        In10[end_idx: int<br/><span style='font-size:0.9em; color:#444'>Analysis End Index</span>]:::inputStyle
        In11[log_filename: str<br/><span style='font-size:0.9em; color:#444'>Log Filename</span>]:::inputStyle
        In12[output_dir: str<br/><span style='font-size:0.9em; color:#444'>Output Directory Path</span>]:::inputStyle
        In13[signal_name: str<br/><span style='font-size:0.9em; color:#444'>Signal Identifier</span>]:::inputStyle


        %% Optional Input Nodes
        InOpt1[log_study: bool]:::optionalInputStyle
        InOpt2[plot_study: bool]:::optionalInputStyle
        InOpt3[save_raw: bool]:::optionalInputStyle

        %% Invisible Node to collect input arrows
        InputCollector( ):::collectorStyle
    end

    subgraph Process
        P[parametric_bias<br/><span style='font-size:0.9em; color:#444'>Runs simulations across<br/>a bias range: b</span>]:::processStyle
        Loop{Loop over bias};
        Call[_run_single_simulation];
    end

    subgraph Output
        %% Primary Return Value
        O[df_bias: pd.DataFrame<br/><span style='font-size:0.9em; color:#444'>DataFrame of metrics per bias,<br/>or None on error/no results</span>]:::outputStyle
    end

    %% -------------------------------- Connections --------------------------------
    %% Connect all inputs to the invisible collector
    In1 --> InputCollector; In2 --> InputCollector; In3 --> InputCollector;
    In4 --> InputCollector; In5 --> InputCollector; In6 --> InputCollector;
    In7 --> InputCollector; In8 --> InputCollector; In9 --> InputCollector;
    In10 --> InputCollector; In11 --> InputCollector; In12 --> InputCollector;
    In13 --> InputCollector; InOpt1 --> InputCollector; InOpt2 --> InputCollector;
    InOpt3 --> InputCollector;

    %% Connect the collector (representing all inputs) to the process
    InputCollector -- Parameters --> P

    P -- Initiates --> Loop
    Loop -- Calls for each b --> Call
    Call -- Returns MetricsDict --> Loop
    Loop -- Aggregates --> P

    %% Connect the process to the output
    P -- Results DataFrame --> O
````

````python
parametric_delta()
````
Runs a parametric sweep over a range of normalized thresholds (`d_norm`) while keeping the encoder bias (`b`) constant.

````mermaid
%%{ init: { 'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'primaryTextColor': '#333', 'lineColor': '#666', 'fontSize': '14px' } } }%%
graph LR
    %% -------------------------------- Styles (Reusable Template Section) --------------------------------
    classDef inputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,color:#333,rx:4,ry:4 %% rx/ry for slightly rounded corners
    classDef processStyle fill:#fff,stroke:#888,stroke-width:1.5px,color:#000,font-weight:bold,rx:4,ry:4
    classDef outputStyle fill:#e8f5e9,stroke:#a5d6a7,stroke-width:1px,color:#333,rx:4,ry:4
    %% Optional input style: Same fill as input, but dashed border for clear visual cue
    classDef optionalInputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,stroke-dasharray: 4 4,color:#333,rx:4,ry:4
    %% Invisible style for grouping nodes without visual clutter
    classDef collectorStyle fill:none,stroke:none

    %% -------------------------------- Diagram Structure: parametric_delta --------------------------------
    subgraph Input
        direction TB %% Arrange inputs vertically

        %% Compulsory Input Nodes
        In1[u: np.ndarray<br/><span style='font-size:0.9em; color:#444'>Input Signal</span>]:::inputStyle
        In2[t: np.ndarray<br/><span style='font-size:0.9em; color:#444'>Signal Time Vector</span>]:::inputStyle
        In3[dur: float<br/><span style='font-size:0.9em; color:#444'>Signal Duration</span>]:::inputStyle
        In4[fs: float<br/><span style='font-size:0.9em; color:#444'>Fixed Sampling Frequency</span>]:::inputStyle
        In5[deltas: np.ndarray<br/><span style='font-size:0.9em; color:#444'>Thresholds to Test: d_norm</span>]:::inputStyle
        In6[b: float<br/><span style='font-size:0.9em; color:#444'>Fixed Encoder Bias</span>]:::inputStyle
        In7[dte: float<br/><span style='font-size:0.9em; color:#444'>Fixed Encoding Sample Time</span>]:::inputStyle
        In8[start_idx: int<br/><span style='font-size:0.9em; color:#444'>Analysis Start Index</span>]:::inputStyle
        In9[end_idx: int<br/><span style='font-size:0.9em; color:#444'>Analysis End Index</span>]:::inputStyle
        In10[log_filename: str<br/><span style='font-size:0.9em; color:#444'>Log Filename</span>]:::inputStyle
        In11[output_dir: str<br/><span style='font-size:0.9em; color:#444'>Output Directory Path</span>]:::inputStyle
        In12[signal_name: str<br/><span style='font-size:0.9em; color:#444'>Signal Identifier</span>]:::inputStyle
        In13[freq_max: float<br/><span style='font-size:0.9em; color:#444'>Max Signal Freq: for BW</span>]:::inputStyle

        %% Optional Input Nodes
        InOpt1[log_study: bool<br/><span style='font-size:0.9em; color:#444'>Enable/Disable Logging</span>]:::optionalInputStyle
        InOpt2[plot_study: bool<br/><span style='font-size:0.9em; color:#444'>Enable/Disable Plotting</span>]:::optionalInputStyle
        InOpt3[save_raw: bool<br/><span style='font-size:0.9em; color:#444'>Enable/Disable Raw Data Saving</span>]:::optionalInputStyle

        %% Invisible Node to collect input arrows
        InputCollector( ):::collectorStyle
    end

    subgraph Process
        P[parametric_delta<br/><span style='font-size:0.9em; color:#444'>Runs simulations across<br/>a threshold range: d_norm</span>]:::processStyle
        Loop{Loop over deltas};
        Call[_run_single_simulation];
    end

    subgraph Output
        %% Primary Return Value
        O[df_delta: pd.DataFrame <br/><span style='font-size:0.9em; color:#444'>DataFrame of metrics per threshold,<br/>or None on error/no results</span>]:::outputStyle
        %% Note: This function also has side effects (creating files/dirs) if options enabled.
    end

    %% -------------------------------- Connections --------------------------------
    %% Connect all inputs to the invisible collector
    In1 --> InputCollector; In2 --> InputCollector; In3 --> InputCollector;
    In4 --> InputCollector; In5 --> InputCollector; In6 --> InputCollector;
    In7 --> InputCollector; In8 --> InputCollector; In9 --> InputCollector;
    In10 --> InputCollector; In11 --> InputCollector; In12 --> InputCollector;
    In13 --> InputCollector; InOpt1 --> InputCollector; InOpt2 --> InputCollector;
    InOpt3 --> InputCollector;

    %% Connect the collector (representing all inputs) to the process
    InputCollector -- Parameters --> P

    P -- Initiates --> Loop
    Loop -- Calls for each d_norm --> Call
    Call -- Returns MetricsDict --> Loop
    Loop -- Aggregates --> P

    %% Connect the process to the output
    P -- Results DataFrame --> O
````

````python
parametric_bias_delta()
````
Runs a biparametric sweep over combinations of encoder bias (`b`) and normalized threshold (`d_norm`).

````mermaid
%%{ init: { 'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'primaryTextColor': '#333', 'lineColor': '#666', 'fontSize': '14px' } } }%%
graph LR
    %% -------------------------------- Styles (Reusable Template Section) --------------------------------
    classDef inputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,color:#333,rx:4,ry:4 %% rx/ry for slightly rounded corners
    classDef processStyle fill:#fff,stroke:#888,stroke-width:1.5px,color:#000,font-weight:bold,rx:4,ry:4
    classDef outputStyle fill:#e8f5e9,stroke:#a5d6a7,stroke-width:1px,color:#333,rx:4,ry:4
    %% Optional input style: Same fill as input, but dashed border for clear visual cue
    classDef optionalInputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,stroke-dasharray: 4 4,color:#333,rx:4,ry:4
    %% Invisible style for grouping nodes without visual clutter
    classDef collectorStyle fill:none,stroke:none

    %% -------------------------------- Diagram Structure: parametric_bias_delta --------------------------------
    subgraph Input
        direction TB %% Arrange inputs vertically

        %% Compulsory Input Nodes
        In1[u: np.ndarray<br/><span style='font-size:0.9em; color:#444'>Input Signal</span>]:::inputStyle
        In2[t: np.ndarray<br/><span style='font-size:0.9em; color:#444'>Signal Time Vector</span>]:::inputStyle
        In3[dur: float<br/><span style='font-size:0.9em; color:#444'>Signal Duration</span>]:::inputStyle
        In4[fs: float<br/><span style='font-size:0.9em; color:#444'>Fixed Sampling Frequency</span>]:::inputStyle
        In5[deltas: np.ndarray<br/><span style='font-size:0.9em; color:#444'>Thresholds to Test</span>]:::inputStyle
        In6[bias: np.ndarray<br/><span style='font-size:0.9em; color:#444'>Bias values to Test</span>]:::inputStyle
        In7[dte: float<br/><span style='font-size:0.9em; color:#444'>Encoding Sample Time</span>]:::inputStyle
        In8[freq_max: float<br/><span style='font-size:0.9em; color:#444'>Max Signal Freq: for BW</span>]:::inputStyle
        In9[start_idx: int<br/><span style='font-size:0.9em; color:#444'>Analysis Start Index</span>]:::inputStyle
        In10[end_idx: int<br/><span style='font-size:0.9em; color:#444'>Analysis End Index</span>]:::inputStyle
        In11[log_filename: str<br/><span style='font-size:0.9em; color:#444'>Log Filename</span>]:::inputStyle
        In12[output_dir: str<br/><span style='font-size:0.9em; color:#444'>Output Directory Path</span>]:::inputStyle
        In13[signal_name: str<br/><span style='font-size:0.9em; color:#444'>Signal Identifier</span>]:::inputStyle

        %% Optional Input Nodes
        InOpt1[log_study: bool]:::optionalInputStyle
        InOpt2[plot_study: bool]:::optionalInputStyle
        InOpt3[save_raw: bool]:::optionalInputStyle

        %% Invisible Node to collect input arrows
        InputCollector( ):::collectorStyle
    end

    subgraph Process
        P[parametric_bias_delta<br/><span style='font-size:0.9em; color:#444'>Runs simulations across<br/>combinations of encoder bias: b<br/>and threshold: d_norm</span>]:::processStyle
        Loop{Nested Loop over<br/>bias & deltas};
        Call[_run_single_simulation];
    end

    subgraph Output
        %% Primary Return Value
        O[df_2d: pd.DataFrame <br/><span style='font-size:0.9em; color:#444'>DataFrame of metrics per: b, d_norm ,<br/>or None on error/no results</span>]:::outputStyle
    end

    %% -------------------------------- Connections --------------------------------
    %% Connect all inputs to the invisible collector
    In1 --> InputCollector; In2 --> InputCollector; In3 --> InputCollector;
    In4 --> InputCollector; In5 --> InputCollector; In6 --> InputCollector;
    In7 --> InputCollector; In8 --> InputCollector; In9 --> InputCollector;
    In10 --> InputCollector; In11 --> InputCollector; In12 --> InputCollector;
    In13 --> InputCollector; InOpt1 --> InputCollector; InOpt2 --> InputCollector;
    InOpt3 --> InputCollector;

    %% Connect the collector (representing all inputs) to the process
    InputCollector -- Parameters --> P

    P -- Initiates --> Loop
    Loop -- Calls for each (b, d_norm) pair --> Call
    Call -- Returns MetricsDict --> Loop
    Loop -- Aggregates --> P

    %% Connect the process to the output
    P -- Results DataFrame --> O
````