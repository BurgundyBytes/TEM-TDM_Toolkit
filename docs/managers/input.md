# Input Handler Controller (`src/controllers/input_handler.py`)

## Overview

This module is responsible for providing the input signal data required for the subsequent processing steps (parametric studies and analysis). It acts as the **Receiving department**, either generating signals internally based on configuration parameters or (in future development) loading them from external experimental data files.

## Functionality

1.  **Manager Function (`manager`)**:
    *   The main entry point for the input handling process.
    *   Reads the `Input Source` setting from the configuration dictionary (`config`).
    *   Based on the source (`'generated'` or `'excel'`), it calls the appropriate helper function (`generate_input_signal` or `get_experiment_signals`).
    *   Returns a list of signal dictionaries (`InputSignals`), where each dictionary contains the time vector (`t`), signal vector (`u`), signal name, and other relevant metadata (sampling rate `fs`, duration `dur`, max frequency `freq_max`, etc.).
    *   Handles errors if the input source is invalid or if the helper functions fail.
2.  **Signal Generation (`generate_input_signal`)**:
    *   Called by `manager` if `Input Source` is `'generated'`.
    *   Reads specific generation parameters from the `config` dictionary (e.g., `Generated Input Type`, `Signal Sampling Rate`, `Signal Duration`, `Frequencies`, `Amplitudes`).
    *   Calls the appropriate signal generation function within the `src.models.input_signal` module (e.g., `generate_sum_of_sines`, `generate_multifreq_signal`, `generate_paper_signal`) based on `Generated Input Type`.
    *   Packages the returned `t`, `u`, and `freq_max` along with metadata into a signal dictionary and returns it within a list.
    *   Handles errors related to missing configuration keys or failures within the generation functions.
3.  **Experiment Signal Loading (`get_experiment_signals`)**:
    *   Called by `manager` if `Input Source` is `'excel'`.
    *   **(Note: Currently a placeholder/partially implemented)** Reads file paths and related configuration from `config`.
    *   Intended to iterate through specified files and call `input_signal.load_experiment_signals` for each.
    *   Currently logs a warning and returns an empty list, as the loading logic requires further development/review.

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
    classDef fileInputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,stroke-dasharray: 3 3,color:#555,rx:4,ry:4 %% Dashed border for file inputs
    classDef collectorStyle fill:none,stroke:none

    %% -------------------------------- Diagram Structure: input_handler.py (Simplified) --------------------------------

    subgraph Inputs
        direction TB
        In1[config: ConfigDict<br/><span style='font-size:0.9em; color:#444'>Global Configuration</span>]:::inputStyle
    end

    subgraph Process
        direction TB
        P_Main[manager<br/><span style='font-size:0.9em; color:#444'>Main Input Orchestrator</span>]:::processStyle
        Decide{Input Source?<br/><span style='font-size:0.9em; color:#444'>Reads config</span>}
        H_Generate[generate_input_signal]:::helperStyle
        H_Load[get_experiment_signals]:::helperStyle
    end

    subgraph Called_Modules ["Called Modules"]
        direction TB
        Ext_GenSignal[src.models.input_signal<br/><span style='font-size:0.9em; color:#444'>Signal generation functions</span>]:::externalStyle
        Ext_LoadSignal[src.models.input_signal<br/><span style='font-size:0.9em; color:#444'>load_experiment_signals<br/>To be implemented</span>]:::externalStyle
    end

    subgraph Outputs
        direction TB
        Out1[signals: InputSignals<br/><span style='font-size:0.9em; color:#444'>List of Signal Dictionaries</span>]:::outputStyle
        OutLogs[Log Messages<br/><span style='font-size:0.9em; color:#444'>Status, Warnings, Errors</span>]:::outputStyle
    end

    %% -------------------------------- Connections --------------------------------
    In1 -- config --> P_Main

    P_Main --> Decide

    Decide -- "'generated'" --> H_Generate
    Decide -- "'excel'" --> H_Load
    Decide -- "Invalid" --> P_Main --> OutLogs 

    H_Generate -- Calls --> Ext_GenSignal
    H_Load -- Calls (intended) --> Ext_LoadSignal

    Ext_GenSignal -- Returns Data --> H_Generate
    Ext_LoadSignal -- Returns Data (intended) --> H_Load

    H_Generate -- Returns List[SignalDict] --> P_Main
    H_Load -- Returns List[SignalDict] --> P_Main

    %% Outputs
    P_Main -- Returns --> Out1
    P_Main -- Generates --> OutLogs
    H_Generate -- Generates --> OutLogs
    H_Load -- Generates --> OutLogs
````