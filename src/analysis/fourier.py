import numpy as np
import os
import sys
from typing import Optional, Tuple, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# custom modules
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, src_path)
from src.utilities import utils, plotting # Rely on utils for FFT and plotting for visualization

# Consistent type definitions
SignalDict = Dict[str, Any]
OptimalSimResultDict = Dict[str, Any]
BoolsDict = Dict[str, bool]
FFTResult = Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]] # freq, spec, fs_used
FFTResultDict = Dict[str, Dict[str, Any]] # Stores {'freq': ..., 'spec': ..., 'fs': ..., 'mag': ...}


# --- Helper Functions ---
def _calculate_fft(signal: Optional[np.ndarray], time_vector: Optional[np.ndarray], label: str) -> FFTResult:
    '''
    Calculates FFT using utils.fft after determining sampling frequency from the time vector.

    Inputs
    ------
    - signal: array_like of floats
        The signal to analyze.
    - time_vector: array_like of floats
        The corresponding time vector.
    - label: str
        Descriptive label for logging.

    Outputs
    -------
    - Tuple: (frequencies, spectrum, sampling_frequency_used) or (None, None, None) on failure.

    Raises
    ------
    - ValueError: If input arrays are invalid or empty.
    - Exception: For any other unexpected errors during processing.
    '''
    if signal is None or time_vector is None or \
       not isinstance(signal, np.ndarray) or not isinstance(time_vector, np.ndarray) or \
       signal.ndim != 1 or time_vector.ndim != 1 or \
       signal.size != time_vector.size or signal.size < 2:
        logger.warning(f"FFT calculation skipped for '{label}': Invalid signal or time vector.")
        return None, None, None

    try:
        # Estimate sampling frequency from the time vector
        dt_vals = np.diff(time_vector)
        # Filter out non-positive or extremely small dt values that might cause issues
        valid_dt = dt_vals[dt_vals > 1e-12]
        if valid_dt.size == 0:
            logger.warning(f"FFT calculation skipped for '{label}': Cannot determine valid time step (dt).")
            return None, None, None

        median_dt = np.median(valid_dt)
        fs_estimated = 1.0 / median_dt
        logger.debug(f"\tEstimated Fs for '{label}' FFT from median dt: {fs_estimated:.2f} Hz")

        if fs_estimated <= 0:
             logger.warning(f"FFT calculation skipped for '{label}': Invalid estimated sampling frequency ({fs_estimated}).")
             return None, None, None

        # Calculate FFT using the utility function
        freq, spec = utils.fft(signal, fs_estimated)

        if freq is not None and spec is not None:
            logger.info(f"\tSuccessfully calculated FFT for '{label}'.")
            return freq, spec, fs_estimated
        else:
            logger.warning(f"\tFFT calculation failed for '{label}' (utils.fft returned None).")
            return None, None, None

    except Exception as e:
        logger.error(f"\tError during FFT calculation for '{label}': {e}", exc_info=True)
        return None, None, None



def _find_principal_frequencies(freq: np.ndarray, spec: np.ndarray, num_peaks: int = 5, min_freq: float = 0.0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    '''
    Identifies the frequencies with the largest magnitudes in the spectrum.

    Inputs
    ------
    - freq: array_like of floats
        Frequencies from FFT.
    - spec: array_like of complex numbers
        Complex spectrum values from FFT.
    - num_peaks: int
        Number of top peaks to identify.
    - min_freq: float
        Minimum frequency to consider (e.g., to exclude DC).
    
    Outputs
    -------
    - Tuple: (principal_frequencies, principal_magnitudes) or (None, None) on failure.

    Raises
    ------
    - ValueError: If input arrays are invalid or empty.
    - Exception: For any other unexpected errors during processing.
    '''
    if freq is None or spec is None or len(freq) != len(spec) or len(freq) == 0:
        return None, None

    try:
        magnitudes = np.abs(spec)

        # Apply frequency mask
        valid_indices = np.where(freq >= min_freq)[0]
        if len(valid_indices) == 0:
            logger.warning("\tNo frequencies found above min_freq for principal component analysis.")
            return np.array([]), np.array([]) # Return empty arrays

        freq_masked = freq[valid_indices]
        magnitudes_masked = magnitudes[valid_indices]

        # Find the indices of the largest magnitudes
        # If fewer points than num_peaks, take all available
        num_peaks_actual = min(num_peaks, len(magnitudes_masked))
        if num_peaks_actual == 0:
             return np.array([]), np.array([])

        peak_indices_masked = np.argsort(magnitudes_masked)[-num_peaks_actual:][::-1] # Get top N indices

        principal_freqs = freq_masked[peak_indices_masked]
        principal_mags = magnitudes_masked[peak_indices_masked]

        return principal_freqs, principal_mags

    except Exception as e:
        logger.error(f"\tError finding principal frequencies: {e}", exc_info=True)
        return None, None
    

def _plot_full_spectra(spectra_data: List[Dict[str, Any]], title: str, save_path: Optional[str]):
    ''' Safely calls the combined spectrum plotting function. '''
    if not spectra_data:
        logger.warning("\tSkipping full spectra plot: No valid spectra data provided.")
        return
    try:
        logger.info(f"\tGenerating combined spectrum plot: {os.path.basename(save_path) if save_path else 'No save path'}")
        # Assuming plot_fourier_combined_spectrum exists and takes this list format
        plotting.plot_fourier_combined_spectrum(spectra_data, title=title, save_path=save_path, log_y=True)
    except AttributeError:
        logger.error("\tPlotting failed: `plotting.plot_fourier_combined_spectrum` not found.")
    except Exception as e:
        logger.error(f"\tError during combined spectrum plotting: {e}", exc_info=True)


def _plot_principal_freqs(principal_data: List[Dict[str, Any]], title: str, save_path: Optional[str]):
    ''' Safely calls the principal frequencies plotting function. '''
    if not principal_data:
        logger.warning("\tSkipping principal frequencies plot: No valid principal frequency data provided.")
        return
    try:
        logger.info(f"\tGenerating principal frequencies plot: {os.path.basename(save_path) if save_path else 'No save path'}")
        # Assuming plot_principal_frequencies exists and takes this list format
        plotting.plot_principal_frequencies(principal_data, title=title, save_path=save_path)
    except AttributeError:
        logger.error("\tPlotting failed: `plotting.plot_principal_frequencies` not found.")
    except Exception as e:
        logger.error(f"\tError during principal frequencies plotting: {e}", exc_info=True)



def perform_fourier_study(signal_data: SignalDict, optimal_sim_results: Optional[OptimalSimResultDict], nyquist_sim_results: Optional[OptimalSimResultDict], fourier_output_path: Optional[str], bools: BoolsDict) -> Optional[FFTResultDict]:
    '''
    Performs Fourier analysis comparing Original, Optimal ASDM, and Traditional Sampled signals.

    1. Gets the traditionally sampled signal using N_spikes from optimal results.
    2. Calculates FFT for all three signals.
    3. Finds principal frequencies for all three.
    4. Generates comparison plots for full spectra and principal frequencies.
    5. Saves FFT data if requested.

    Inputs
    ------
    - signal_data: dict
        Dictionary containing original signal data.
    - optimal_sim_results: dict
        Dictionary containing optimal simulation results.
    - nyquist_sim_results: dict
        Dictionary containing traditionally sampled signal data.
    - fourier_output_path: str
        Directory to save outputs.
    - bools: dict
        Dictionary of flags for plotting and saving.

    Outputs
    -------
    - fft_results: dict or None
        Dictionary containing FFT results for original, reconstructed optimal, and reconstructed traditional signals.

    Raises
    ------
    - Exception: If any unexpected errors occur during processing.
    '''
    signal_name = signal_data.get('name', 'unnamed_signal')
    run_id = f"Fourier_{signal_name}"
    logger.info(f"--- Running {run_id} ---")

    # --- Extract Base Data ---
    t_orig = signal_data.get('t')
    u_orig = signal_data.get('u')
    duration = signal_data.get('dur')

    if t_orig is None or u_orig is None or duration is None or duration <= 0:
        logger.error(f"{run_id}: Missing essential original signal data (t, u, or dur). Aborting.")
        return None

    fft_results: FFTResultDict = {}
    principal_freq_results: Dict[str, Dict[str, Optional[np.ndarray]]] = {} # Store {'label': {'freqs': ..., 'mags': ...}}

    # --- 1. Process Original Signal ---
    freq_orig, spec_orig, fs_orig = _calculate_fft(u_orig, t_orig, 'Original')
    if freq_orig is not None and spec_orig is not None:
        fft_results['original'] = {'freq': freq_orig, 'spec': spec_orig, 'fs': fs_orig, 'mag': np.abs(spec_orig)}


    # --- 2. Process Optimal Reconstructed Signal ---
    u_rec_opt, t_rec_opt = None, None
    if optimal_sim_results:
        if optimal_sim_results.get('u_rec_stable') is not None and optimal_sim_results.get('t_stable') is not None:
            u_rec_opt = optimal_sim_results.get('u_rec_stable')
            t_rec_opt = optimal_sim_results.get('t_stable')
            logger.info("\tUsing STABLE region for optimal reconstructed signal FFT.")
        else:
            u_rec_opt = optimal_sim_results.get('u_rec')
            t_rec_opt = optimal_sim_results.get('t') 
            logger.info("\tUsing FULL signal for optimal reconstructed signal FFT.")

        # Check alignment again, just in case
        if t_rec_opt is not None and len(t_rec_opt) != len(u_rec_opt):
            logger.warning(f"\tLength mismatch for optimal reconstructed signal ({len(u_rec_opt)}) and its time vector ({len(t_rec_opt)}). Skipping its FFT.")
            u_rec_opt = None 

        if u_rec_opt is not None and t_rec_opt is not None:
            freq_opt, spec_opt, fs_opt = _calculate_fft(u_rec_opt, t_rec_opt, 'Reconstructed_Optimal')
            if freq_opt is not None and spec_opt is not None:
                fft_results['reconstructed_optimal'] = {'freq': freq_opt, 'spec': spec_opt, 'fs': fs_opt, 'mag': np.abs(spec_opt)}
        else:
             logger.warning("\tOptimal reconstructed signal or time data invalid/missing. Skipping its FFT.")
    else:
        logger.warning("\tOptimal simulation results not provided. Skipping optimal reconstruction FFT.")


    # --- 3. Extract Traditionally Sampled Signal ---
    u_rec_trad = nyquist_sim_results.get('u_rec_trad')
    t_trad = nyquist_sim_results.get('t_trad')
    n_spikes_trad = nyquist_sim_results.get('n_nyquist_optimal')
    try:
        # Calculate FFT for the traditional signal
        freq_trad, spec_trad, fs_trad = _calculate_fft(u_rec_trad, t_trad, 'Reconstructed_Traditional')
        if freq_trad is not None and spec_trad is not None:
            fft_results['reconstructed_traditional'] = {'freq': freq_trad, 'spec': spec_trad, 'fs': fs_trad, 'mag': np.abs(spec_trad)}

    except Exception as e:
        logger.error(f"\tError generating or processing traditionally sampled signal: {e}", exc_info=True)
    

    # --- 4. Find Principal Frequencies ---
    num_principal = 5 # HARD-CODED: Number of principal frequencies to find
    min_principal_freq = 0.1 # HARD-CODED: Minimum frequency to consider for principal frequencies (Hz)
    logger.info(f"\tFinding top {num_principal} principal frequencies (min_freq={min_principal_freq} Hz)...")

    for label, data in fft_results.items():
        pfreq, pmag = _find_principal_frequencies(data['freq'], data['spec'], num_principal, min_principal_freq)
        if pfreq is not None and pmag is not None:
            principal_freq_results[label] = {'freqs': pfreq, 'mags': pmag}
            logger.info(f"\t\tFound {len(pfreq)} principal freqs for '{label}'.")
        else:
            logger.warning(f"\t\tFailed to find principal frequencies for '{label}'.")


    # --- 5. Plotting ---
    if bools.get('plots', False) and fourier_output_path:
        plot_base = f"fourier_{signal_name}"

        # a) Full Spectrum Comparison Plot
        spectra_plot_data = []
        for label, data in fft_results.items():
            spectra_plot_data.append({
                'freq': data['freq'],
                'mag': data['mag'], # Use pre-calculated magnitude
                'label': label.replace('_', ' ').title() # Nicer label
            })
        plot_path_full = os.path.join(fourier_output_path, f"{plot_base}_comparison_spectrum.png")
        _plot_full_spectra(spectra_plot_data, title=f"Full Spectrum Comparison ({signal_name})", save_path=plot_path_full)

        # b) Principal Frequencies Comparison Plot
        principal_plot_data = []
        for label, data in principal_freq_results.items():
             if data and data.get('freqs') is not None and data.get('mags') is not None:
                 principal_plot_data.append({
                     'freqs': data['freqs'],
                     'mags': data['mags'],
                     'label': label.replace('_', ' ').title()
                 })
        plot_path_princ = os.path.join(fourier_output_path, f"{plot_base}_principal_freqs.png")
        _plot_principal_freqs(principal_plot_data, title=f"Principal Frequencies Comparison ({signal_name})", save_path=plot_path_princ)

    else:
        logger.info("\tPlotting skipped (disabled or output path missing).")


    # --- 6. Save FFT Data ---
    if bools.get('pickle', False) and fourier_output_path:
        if fft_results: # Only save if we calculated something
            try:
                # Note: fft_results contains complex 'spec', maybe save only freq/mag/fs?
                # Or keep 'spec' if needed later. For now, saving the full dict.
                data_filename = f"fourier_data_{signal_name}"
                utils.save_pickle(fft_results, fourier_output_path, data_filename)
                logger.info(f"\tSaved Fourier FFT data pickle ({data_filename}.pkl).")
            except Exception as save_err:
                logger.error(f"\tError saving Fourier data: {save_err}", exc_info=True)
        else:
            logger.warning("\tSkipping pickle save: No FFT results were generated.")


    logger.info(f"{run_id} complete.")
    return fft_results if fft_results else None 