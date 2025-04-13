# Configuration Controller (`src/controllers/configuration.py`)

## Overview

This module centralizes all operations related to reading, parsing, validating, and managing the toolkit's configuration settings, primarily from the `config.txt` file. It also handles the creation of the output directory structure.

## Functionality

1.  **Parsing Helpers (`_parse_*`)**: Provides internal utility functions to safely convert configuration strings read from the file into specific data types (boolean, float, int, lists, ranges), including basic validation (e.g., non-zero step in ranges).
2.  **Configuration Loading (`load_config`)**:
    *   Reads the specified configuration file (`config.txt` by default).
    *   Ignores comments (`#`) and blank lines.
    *   Splits lines into key-value pairs.
    *   Uses the parsing helpers to convert values to their appropriate types.
    *   Performs essential validation checks (e.g., valid 'Input Source', 'Generated Input Type').
    *   Consolidates all settings into a single configuration dictionary (`ConfigDict`).
    *   Raises specific errors (FileNotFoundError, KeyError, ValueError) if issues are encountered.
3.  **Workflow Flag Extraction (`get_workflow_flags`)**:
    *   Takes the loaded configuration dictionary as input.
    *   Extracts key boolean flags that control the main execution paths (e.g., `Execution Flow`, `Results Flow`, `Run Optimal`, `Run Nyquist`, etc.).
    *   Returns these flags in a separate dictionary (`BoolsDict`) for easy access by the launcher.
4.  **Output Folder Setup (`setup_output_folders`)**:
    *   Takes the configuration dictionary and the workflow flags dictionary as input.
    *   Determines the base output path based on the input source (e.g., `Output/excel/my_signal/` or `Output/generated/sinc/`).
    *   Conditionally creates specific sub-folders for parametric studies and analysis results *only if* the corresponding workflow flags are enabled in both the `config.txt` (e.g., `Run Parametric Delta: true`) and the overall workflow (`Execution Flow: true` or `Results Flow: true`).
    *   Uses `os.makedirs(exist_ok=True)` to safely create directories.
    *   Returns a dictionary (`OutputPathsDict`) mapping folder types (e.g., `param_delta`, `nyquist`) to their absolute paths (or `None` if not created).

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

    %% -------------------------------- Diagram Structure: configuration.py --------------------------------

    subgraph Inputs
        direction TB
        In1[filename: str<br/><span style='font-size:0.9em; color:#444'>Path to config file</span>]:::inputStyle
        In2[ConfigDict: for flags/folders<br/><span style='font-size:0.9em; color:#444'>Loaded Config -Input to others-</span>]:::inputStyle
        In3[BoolsDict: for folders<br/><span style='font-size:0.9em; color:#444'>Workflow Flags -Input to folders-</span>]:::inputStyle
        InFile[config.txt<br/><span style='font-size:0.9em; color:#444'>Physical Configuration File</span>]:::fileInputStyle
    end

    subgraph Process
        direction TB
        P_Load[load_config<br/><span style='font-size:0.9em; color:#444'>Loads, Parses, Validates Config</span>]:::processStyle
        P_Flags[get_workflow_flags<br/><span style='font-size:0.9em; color:#444'>Extracts Boolean Flags</span>]:::processStyle
        P_Folders[setup_output_folders<br/><span style='font-size:0.9em; color:#444'>Creates Output Directories</span>]:::processStyle

        H_Parse[_parse_ Helpers<br/><span style='font-size:0.9em; color:#444'>Type Conversion Utilities</span>]:::helperStyle
        Ext_OS[os module<br/><span style='font-size:0.9em; color:#444'>Path manipulation, makedirs</span>]:::externalStyle

    end

    subgraph Outputs
        direction TB
        Out1[ConfigDict<br/><span style='font-size:0.9em; color:#444'>Parsed Configuration Settings</span>]:::outputStyle
        Out2[BoolsDict<br/><span style='font-size:0.9em; color:#444'>Workflow Control Flags</span>]:::outputStyle
        Out3[OutputPathsDict<br/><span style='font-size:0.9em; color:#444'>Paths to Created Folders</span>]:::outputStyle
        OutFS[Output Folders<br/><span style='font-size:0.9em; color:#444'>Directories Created on Filesystem</span>]:::fileOutputStyle
        OutLogs[Log Messages<br/><span style='font-size:0.9em; color:#444'>Status and Errors</span>]:::outputStyle
    end

    %% -------------------------------- Connections --------------------------------
    InFile --> P_Load
    In1 --> P_Load

    P_Load -- Uses --> H_Parse
    P_Load -- Returns --> Out1

    In2 --> P_Flags
    P_Flags -- Extracts from --> Out1
    P_Flags -- Returns --> Out2

    In2 --> P_Folders
    In3 --> P_Folders
    P_Folders -- Uses --> Out1
    P_Folders -- Uses --> Out2
    P_Folders -- Calls --> Ext_OS
    P_Folders -- Returns --> Out3
    Ext_OS -- Creates --> OutFS

    %% Logging
    P_Load -- Generates --> OutLogs
    P_Flags -- Generates --> OutLogs
    P_Folders -- Generates --> OutLogs
````