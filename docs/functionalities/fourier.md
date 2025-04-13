# Fourier Analysis Module (`src/analysis/fourier.py`)

## Overview

This module performs Fourier analysis to compare the frequency content of the original signal, the signal reconstructed using the optimal ASDM parameters, and optionally, a signal reconstructed using traditional sampling (at a comparable number of points, obtained from the Nyquist analysis). It calculates the Fast Fourier Transform (FFT) for each relevant signal, identifies the principal frequency components, and generates comparison plots.

## Functionality

1.  **Main Orchestrator (`perform_fourier_study`)**:
    *   The primary public function called by the `results_handler`.
    *   Takes original signal data, optimal ASDM simulation results, Nyquist analysis results (containing the traditionally sampled signal), the output path, and configuration flags as input.
    *   **FFT Calculation:** Calls the `_calculate_fft` helper function for:
        *   The original signal (`u_orig`).
        *   The optimally reconstructed signal (`u_rec_opt` from `optimal_sim_results`).
        *   The traditionally reconstructed signal (`u_rec_trad` from `nyquist_sim_results`).
    *   Stores the calculated FFT results (frequency array, complex spectrum array, sampling frequency used, magnitude array) in a dictionary (`fft_results`).
    *   **Principal Frequency Analysis:** Calls `_find_principal_frequencies` for each successfully calculated spectrum to identify the frequencies with the highest magnitudes.
    *   **Plotting (Optional):**
        *   Calls `_plot_full_spectra` to generate a comparative plot of the magnitude spectra of all available signals.
        *   Calls `_plot_principal_freqs` to generate a comparative plot of the identified principal frequencies and their magnitudes.
    *   **Saving Data (Optional):** Saves the `fft_results` dictionary (containing frequency, spectrum, fs, and magnitude arrays) to a pickle file using `utils.save_pickle`.
    *   Returns the `fft_results` dictionary.
2.  **FFT Calculation (`_calculate_fft`)**:
    *   Helper function to compute the FFT for a given signal and time vector.
    *   Estimates the sampling frequency (`fs`) from the median time step (`dt`) of the provided `time_vector`.
    *   Calls the core `utils.fft` function to perform the FFT calculation.
    *   Returns the frequency array, the complex spectrum array, and the estimated sampling frequency used, or `(None, None, None)` on failure.
3.  **Principal Frequency Identification (`_find_principal_frequencies`)**:
    *   Helper function to find the `num_peaks` frequencies with the largest magnitudes in a given spectrum, above a specified `min_freq`.
    *   Calculates magnitudes from the complex spectrum.
    *   Sorts magnitudes (above `min_freq`) and selects the top `num_peaks`.
    *   Returns the corresponding principal frequencies and their magnitudes.
4.  **Plotting Helpers (`_plot_full_spectra`, `_plot_principal_freqs`)**:
    *   Wrapper functions that prepare data and safely call the respective plotting functions (`plotting.plot_fourier_combined_spectrum`, `plotting.plot_principal_frequencies`) located in the `plotting` module. Handle potential errors during plotting calls.

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

    %% -------------------------------- Diagram Structure: High-Level fourier.py (Simplified) --------------------------------

    subgraph Inputs
        direction TB
        In1[signal_data: SignalDict]:::inputStyle
        In2[optimal_sim_results: OptimalSimResultDict?]:::optionalInputStyle
        In3[nyquist_sim_results: OptimalSimResultDict?]:::optionalInputStyle
        In4[fourier_output_path: str?]:::optionalInputStyle
        In5[bools: BoolsDict]:::inputStyle
    end

    subgraph Process
        direction TB
        P_Main[perform_fourier_study]:::processStyle
        H_Calc[_calculate_fft]:::helperStyle
        H_Princ[_find_principal_frequencies]:::helperStyle
        H_Plot1[_plot_full_spectra]:::helperStyle
        H_Plot2[_plot_principal_freqs]:::helperStyle
    end

    subgraph External_Modules ["Called Modules"]
         direction TB
         Ext_FFT[utils.fft]:::externalStyle
         Ext_Plot[plotting.*]:::externalStyle
         Ext_Save[utils.save_pickle]:::externalStyle
    end

    subgraph Outputs
        direction TB
        O_Dict[fft_results: FFTResultDict?<br/><span style='font-size:0.9em; color:#444'>FFT Data Dict or None</span>]:::outputStyle
        O_Files[Output Files<br/><span style='font-size:0.9em; color:#444'>Plots,<br/>FFT Data Pickle</span>]:::fileOutputStyle
        OutLogs[Log Messages]:::outputStyle
    end

    %% -------------------------------- Connections --------------------------------
    Inputs --> P_Main

    %% Main orchestrator calls helpers
    P_Main -- Calls (x3) --> H_Calc
    H_Calc -- FFT Results --> P_Main[perform_fourier_study]
    P_Main -- Calls (x3) --> H_Princ
    H_Princ -- Principal Freqs --> P_Main

    %% Helpers call external modules
    H_Calc -- Calls --> Ext_FFT

    %% Plotting and Saving (Conditional)
    P_Main -- Calls (Opt) --> H_Plot1
    P_Main -- Calls (Opt) --> H_Plot2
    P_Main -- Calls (Opt) --> Ext_Save
    H_Plot1 -- Calls --> Ext_Plot
    H_Plot2 -- Calls --> Ext_Plot

    %% File Outputs
    Ext_Plot -- Creates --> O_Files
    Ext_Save -- Creates --> O_Files

    %% Return Value
    P_Main -- Returns --> O_Dict

    %% Logging
    P_Main -- Generates --> OutLogs
    H_Calc -- Generates --> OutLogs
    H_Princ -- Generates --> OutLogs
````
## Visualization: function level

````python
perform_fourier_study()
````
Main orchestrator for the Fourier analysis comparison.
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
        In2[optimal_sim_results: OptimalSimResultDict?]:::optionalInputStyle
        In3[nyquist_sim_results: OptimalSimResultDict?]:::optionalInputStyle
        In4[fourier_output_path: str?]:::optionalInputStyle
        In5[bools: BoolsDict]:::inputStyle
    end

    subgraph Process
        P[perform_fourier_study<br/><span style='font-size:0.9em; color:#444'>Calculates FFTs,<br/>Finds Principal Freqs,<br/>Calls Plotters & Savers</span>]:::processStyle
        Call_Calc[_calculate_fft]:::helperStyle
        Call_Princ[_find_principal_frequencies]:::helperStyle
        Call_Plot1[_plot_full_spectra]:::helperStyle
        Call_Plot2[_plot_principal_freqs]:::helperStyle
        Call_Save[utils.save_pickle]:::externalStyle
    end

    subgraph Outputs
        O[fft_results: FFTResultDict?<br/><span style='font-size:0.9em; color:#444'>Dict with FFT data or None</span>]:::outputStyle
        %% Side effects: Plots, Pickle file
    end

    %% Connections
    Inputs --> P
    P -- Calls (x3) --> Call_Calc
    P -- Calls (x3) --> Call_Princ
    P -- Calls (Opt) --> Call_Plot1
    P -- Calls (Opt) --> Call_Plot2
    P -- Calls (Opt) --> Call_Save
    P -- Returns --> O
````

````python
_calculate_fft()
````
Helper to calculate FFT for a single signal.
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
        In1[signal: np.ndarray?]:::inputStyle
        In2[time_vector: np.ndarray?]:::inputStyle
        In3[label: str]:::inputStyle
    end

    subgraph Process
        P[_calculate_fft<br/><span style='font-size:0.9em; color:#444'>Estimates Fs,<br/>Calls utils.fft</span>]:::processStyle
        Ext[utils.fft]:::externalStyle
    end

    subgraph Outputs
        O[FFTResult<br/><span style='font-size:0.9em; color:#444'>freq, spec, fs</span>]:::outputStyle
    end

    %% Connections
    Inputs --> P
    P -- Calls --> Ext
    Ext -- FFT data --> P
    P -- Returns --> O
````

````python
_find_principal_frequencies()
````
Helper to identify dominant frequencies in a spectrum.

````mermaid
%%{ init: { 'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'primaryTextColor': '#333', 'lineColor': '#666', 'fontSize': '14px' } } }%%
graph LR
    %% Style Definitions ...
    classDef inputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,color:#333,rx:4,ry:4
    classDef processStyle fill:#fff,stroke:#888,stroke-width:1.5px,color:#000,font-weight:bold,rx:4,ry:4
    classDef outputStyle fill:#e8f5e9,stroke:#a5d6a7,stroke-width:1px,color:#333,rx:4,ry:4

    subgraph Inputs
        direction LR
        In1[freq: np.ndarray?]:::inputStyle
        In2[spec: np.ndarray?]:::inputStyle
        In3[num_peaks: int]:::inputStyle
        In4[min_freq: float]:::inputStyle
    end

    subgraph Process
        P[_find_principal_frequencies<br/><span style='font-size:0.9em; color:#444'>Calculates Magnitudes,<br/>Applies min_freq mask,<br/>Sorts & Selects Top Peaks</span>]:::processStyle
    end

    subgraph Outputs
        O[Tuple?<br/><span style='font-size:0.9em; color:#444'>principal_freqs, principal_mags<br/>or None, None</span>]:::outputStyle
    end

    %% Connections
    Inputs --> P
    P -- Returns --> O
````

````pyhton
_plot_full_spectra() & _plot_principal_freqs()
````
These are simple wrappers around plotting functions.

````mermaid
%%{ init: { 'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'primaryTextColor': '#333', 'lineColor': '#666', 'fontSize': '14px' } } }%%
graph LR
    %% Style Definitions ...
    classDef inputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,color:#333,rx:4,ry:4
    classDef processStyle fill:#fff,stroke:#888,stroke-width:1.5px,color:#000,font-weight:bold,rx:4,ry:4
    classDef externalStyle fill:#ffe0b2,stroke:#ffb74d,stroke-width:1px,color:#333,rx:2,ry:2

    subgraph Inputs
        direction LR
        In1[Plot Data: List of Dict]:::inputStyle
        In2[title: str]:::inputStyle
        In3[save_path: str?]:::inputStyle
    end

    subgraph Process
        P[_plot_full_spectra / _plot_principal_freqs<br/><span style='font-size:0.9em; color:#444'>Wrapper Function</span>]:::processStyle
        Ext[plotting.*<br/><span style='font-size:0.9em; color:#444'>Actual Plotting Function</span>]:::externalStyle
    end

    %% Connections
    Inputs --> P
    P -- Calls --> Ext
````