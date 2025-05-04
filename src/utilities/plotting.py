import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import os
import sys
import pandas as pd
from typing import List, Optional, Any, Dict, Tuple, Sequence
import logging

# custom modules
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, src_path)
import src.utilities.utils as utils

logger = logging.getLogger(__name__)

# --- Determine Style Name ---
_PREFERRED_STYLE = 'seaborn-v0_8-ticks'  # Preferred academic style
_FALLBACK_STYLE = 'default'           # Fallback

# Check availability and set the style name to use globally in this module
if _PREFERRED_STYLE in plt.style.available:
    _STYLE_NAME_TO_USE = _PREFERRED_STYLE
    logger.info(f"Using matplotlib style: '{_STYLE_NAME_TO_USE}'")
else:
    _STYLE_NAME_TO_USE = _FALLBACK_STYLE
    logger.warning(f"Preferred style '{_PREFERRED_STYLE}' not available. Using fallback: '{_STYLE_NAME_TO_USE}'")

# --- Aesthetic Constants ---
BASE_LINEWIDTH = 1.3
BASE_MARKERSIZE = 5
GRID_ALPHA = 0.6
GRID_LINESTYLE = ':'

# Define colors (using common academic choices)
COLOR_ORIGINAL = 'black'
COLOR_OPTIMAL_ASDM = '#d62728' # Red (tab:red)
COLOR_NYQUIST = '#2ca02c'      # Green (tab:green)
COLOR_RECONSTRUCTED = COLOR_OPTIMAL_ASDM # Alias for consistency
COLOR_ERROR = '#8c564b'         # Brown (tab:brown)
COLOR_SAMPLES = COLOR_OPTIMAL_ASDM # Use same color as reconstruction line
COLOR_SPIKES = '#ff7f0e'        # Orange (tab:orange)

# --- Use constants from utils ---
COL_B = utils.COL_B
COL_D_NORM = utils.COL_D_NORM
COL_MAX_ERR = utils.COL_MAX_ERR
COL_MED_ERR = utils.COL_MED_ERR
COL_TIME = utils.COL_TIME
COL_N_SPIKES = utils.COL_N_SPIKES
COL_N_POINTS = utils.COL_N_POINTS
COL_RMSE = utils.COL_RMSE


# --- General Plot Saving Helper ---
def _save_and_close_plot(fig: Optional[plt.Figure], filepath: Optional[str]) -> None:
    '''Helper to save plot if filepath is given and close figure.'''
    if fig is None:
        logger.debug("_save_and_close_plot called with fig=None.")
        return

    if filepath:
        try:
            output_dir = os.path.dirname(filepath)
            if output_dir: 
                os.makedirs(output_dir, exist_ok=True)
            fig.savefig(filepath, bbox_inches='tight', dpi=300)
            logger.debug(f"Plot saved successfully to: {os.path.basename(filepath)}")
        except Exception as e:
            logger.error(f"Failed to save plot to {os.path.basename(filepath)}: {e}", exc_info=False)
    else:
        logger.debug("Filepath not provided, plot not saved.")

    # Always close the figure if it exists and is managed by pyplot
    if plt.fignum_exists(fig.number):
        plt.close(fig)
    elif fig is not None:
        plt.close(fig)


# --- Basic Signal Plots ---
def plot_signal(time: np.ndarray, signal: np.ndarray, title: str, filepath: Optional[str], ylim: Optional[Tuple[float, float]] = None) -> None:
    '''Plots a single signal against time.'''
    fig, ax = None, None 
    try:
        with plt.style.context(_STYLE_NAME_TO_USE):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(time, signal, linewidth=BASE_LINEWIDTH, color=COLOR_ORIGINAL)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title(title)
            if ylim:
                ax.set_ylim(ylim)
            ax.grid(True, linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)
            fig.tight_layout()
    except Exception as e:
        logger.error(f"Error generating plot_signal '{title}': {e}", exc_info=True)
    finally:
        _save_and_close_plot(fig, filepath)


def plot_process(time: np.ndarray, original_signal: np.ndarray, reconstructed_signal: np.ndarray, title: str, filepath: Optional[str]) -> None:
    '''Plots original, reconstructed, and error signals using subplots.'''
    if time is None or original_signal is None or reconstructed_signal is None:
        logger.warning(f"plot_process: Missing input data for '{os.path.basename(filepath) if filepath else title}'. Skipping plot.")
        return
    if len(time) != len(original_signal) or len(time) != len(reconstructed_signal):
        logger.warning(f"plot_process: Length mismatch in inputs for '{os.path.basename(filepath) if filepath else title}'. Skipping plot.")
        return

    fig, axes = None, None
    try:
        with plt.style.context(_STYLE_NAME_TO_USE):
            fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

            # Original
            axes[0].plot(time, original_signal, label="Original", linewidth=BASE_LINEWIDTH, color=COLOR_ORIGINAL)
            axes[0].set_ylabel("Amplitude")
            axes[0].grid(True, linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)
            axes[0].legend(loc='upper right', frameon=False)

            # Reconstructed
            axes[1].plot(time, reconstructed_signal, label="Reconstructed", linewidth=BASE_LINEWIDTH, color=COLOR_RECONSTRUCTED)
            axes[1].set_ylabel("Amplitude")
            axes[1].grid(True, linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)
            axes[1].legend(loc='upper right', frameon=False)

            # Error
            error_signal = original_signal - reconstructed_signal
            axes[2].plot(time, error_signal, label="Error", linewidth=BASE_LINEWIDTH * 0.9, color=COLOR_ERROR)
            axes[2].set_xlabel("Time (s)")
            axes[2].set_ylabel("Error")
            axes[2].grid(True, linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)
            axes[2].legend(loc='upper right', frameon=False)

            fig.suptitle(title) 
            fig.tight_layout(rect=[0, 0, 1, 0.96])

    except Exception as e:
        logger.error(f"Error generating plot_process '{title}': {e}", exc_info=True)
    finally:
        _save_and_close_plot(fig, filepath)


def plot_with_spikes(time: np.ndarray, original_signal: np.ndarray, reconstructed_signal: np.ndarray, spikes: np.ndarray, title: str, filepath: Optional[str]) -> None:
    '''Plots original, spikes (as stem plot), and reconstruction.'''
    if time is None or len(time) == 0:
            logger.warning(f"plot_with_spikes: Empty or None time vector for '{os.path.basename(filepath) if filepath else title}'. Skipping plot.")
            return
    if original_signal is None or reconstructed_signal is None or spikes is None:
         logger.warning(f"plot_with_spikes: Missing signal/spike data for '{os.path.basename(filepath) if filepath else title}'. Skipping plot.")
         return

    fig, axes = None, None
    try:
        with plt.style.context(_STYLE_NAME_TO_USE):
            fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

            # Original
            axes[0].plot(time, original_signal, label="Original", linewidth=BASE_LINEWIDTH, color=COLOR_ORIGINAL)
            axes[0].set_ylabel("Amplitude")
            axes[0].grid(True, linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)
            axes[0].legend(loc='upper right', frameon=False)

            # Spike plot
            axes[1].set_title(f"Encoded Spikes (N={len(spikes)})")
            axes[1].set_ylabel("Spike Event")
            axes[1].set_ylim(0, 1.2)
            axes[1].set_yticks([0, 1])
            axes[1].grid(True, linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)

            if len(spikes) > 0:
                try:
                    t_start = time[0] if len(time) > 0 else 0.0
                    spike_times = np.cumsum(np.asarray(spikes, dtype=float)) + t_start
                    spike_heights = np.ones_like(spike_times)
                    axes[1].stem(spike_times, spike_heights, basefmt=" ", markerfmt="r.", linefmt="r-")
                except ValueError as stem_err:
                    logger.warning(f"\tStem plot failed for spikes in '{os.path.basename(filepath)}' ({stem_err}).")
                except Exception as spike_err:
                    logger.error(f"\tError processing/plotting spikes for '{os.path.basename(filepath)}': {spike_err}", exc_info=True)
            else:
                logger.debug(f"\tEmpty spike train provided for '{os.path.basename(filepath) if filepath else title}'.")
                if len(time) > 0:
                    axes[1].set_xlim(time[0], time[-1])

            # Reconstructed
            if len(time) == len(reconstructed_signal):
                axes[2].plot(time, reconstructed_signal, label="Reconstructed", linewidth=BASE_LINEWIDTH, color=COLOR_RECONSTRUCTED)
            else:
                logger.warning(f"Length mismatch for reconstructed signal in spike plot '{os.path.basename(filepath) if filepath else title}'. Reconstruction not plotted.")
            axes[2].set_ylabel("Amplitude")
            axes[2].set_xlabel("Time (s)")
            axes[2].grid(True, linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)
            axes[2].legend(loc='upper right', frameon=False)

            fig.suptitle(title)
            fig.tight_layout(rect=[0, 0, 1, 0.96])

    except Exception as e:
        logger.error(f"Error generating plot_with_spikes '{title}': {e}", exc_info=True)
    finally:
        _save_and_close_plot(fig, filepath)


# --- Parametric Study Plots ---
def plot_parametric(df: pd.DataFrame, study_id: str, title: str, filepath: Optional[str]) -> None:
    '''
    Plots summary metrics from a 1D parametric study (e.g., vs Bias or vs Delta).
    Hardcoded plots for now (could be customized later). Assembled as 2x3 plots
    '''
    # Define metrics and their plot properties
    # Using utils constants as keys where possible
    plot_config = {
        utils.COL_N_SPIKES: {'label': "Encoded Length", 'scale': 'linear'},
        utils.COL_MAX_ERR: {'label': "Max Error", 'scale': 'log'},
        utils.COL_MED_ERR: {'label': "Median Error", 'scale': 'log'}, 
        utils.COL_NRMSE_STD: {'label': "Normalized RMS Error", 'scale': 'log'},
        utils.COL_R2: {'label': "R^2", 'scale': 'linear'},
        utils.COL_TIME: {'label': "Elapsed Time (s)", 'scale': 'linear'} 
    }
    metrics_to_plot = list(plot_config.keys()) 

    # Determine x-axis based on study_id
    if study_id.lower() == "pb":
        x_data_key = utils.COL_B
        x_label = f"{x_data_key}" 
    elif study_id.lower() == "pd":
        x_data_key = utils.COL_D_NORM
        x_label = f"{x_data_key}"
    else:
        logger.error(f"plot_parametric: Invalid study_id '{study_id}'. Must be 'bias' or 'delta'. Skipping plot.")
        return

    # Input validation
    required_columns = [x_data_key] + metrics_to_plot
    if df is None or df.empty:
        logger.warning(f"plot_parametric: DataFrame for '{title}' is empty or None. Skipping plot.")
        return
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        logger.warning(f"plot_parametric: DataFrame for '{title}' missing columns: {missing_cols}. Skipping plot.")
        return

    # Data preparation
    df_plot, x_data = None, None
    try:
        # Convert only necessary columns to numeric
        df_plot = df[required_columns].copy()
        for col in required_columns:
            df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')
        df_plot = df_plot.dropna().sort_values(by=x_data_key)

        if df_plot.empty:
            logger.warning(f"plot_parametric: DataFrame for '{title}' empty after converting/dropping NaNs. Skipping plot.")
            return
        x_data = df_plot[x_data_key].to_numpy()
    except Exception as e:
        logger.error(f"Error preparing data for plot_parametric '{title}': {e}. Skipping plot.", exc_info=True)
        return

    # Plotting
    nrows = 2 # hardcoded for now
    ncols = 3 # hardcoded for now
    fig, axes = None, None
    if len(metrics_to_plot) != nrows * ncols:
            logger.error(f"plot_parametric: Mismatch between metrics to plot ({len(metrics_to_plot)}) and grid size ({nrows}x{ncols}). Skipping plot.")
            return
    try:
        with plt.style.context(_STYLE_NAME_TO_USE):
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5), squeeze=False)

            for i, metric_key in enumerate(metrics_to_plot):
                # Calculate row and column index for the 2D axes array
                row = i // ncols # Integer division gives the row index (0 or 1)
                col = i % ncols  # Modulo gives the column index (0, 1, or 2)
                ax = axes[row, col]
                cfg = plot_config[metric_key]
                y_data = df_plot[metric_key].to_numpy()

                ax.plot(x_data, y_data, marker='.', linestyle='-', linewidth=BASE_LINEWIDTH*0.8, markersize=BASE_MARKERSIZE)
                ax.set_title(f"{cfg['label']}") # Short title
                ax.set_xlabel(x_label)
                ax.set_ylabel(cfg['label'])

                # Apply log scale carefully
                current_use_log = cfg['scale'] == 'log'
                if current_use_log:
                    try:
                        if np.any(y_data <= 1e-15): 
                            logger.warning(f"\tNon-positive values found in '{metric_key}' for '{title}'. Plotting on linear scale.")
                            current_use_log = False
                        else:
                            ax.set_yscale('log')
                    except ValueError as log_err:
                        logger.warning(f"\tCould not set log scale for '{metric_key}' in '{title}': {log_err}")
                        current_use_log = False

                ax.grid(True, which='both' if current_use_log else 'major', linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)

            fig.suptitle(title, fontsize=14)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    except Exception as e:
        logger.error(f"Error generating plot_parametric '{title}': {e}", exc_info=True)
    finally:
        _save_and_close_plot(fig, filepath)


def plot_biparametric(df: pd.DataFrame, title: str, filepath: Optional[str]) -> None:
    '''Plots results of the bi-parametric study using contour plots.'''
    # Define metrics and their plot properties
    plot_config = {
        utils.COL_N_SPIKES: {'label': "Encoded Length", 'scale': 'linear'},
        utils.COL_MAX_ERR: {'label': "Max Error", 'scale': 'log'},
        utils.COL_MED_ERR: {'label': "Median Error", 'scale': 'log'}, 
        utils.COL_NRMSE_STD: {'label': "Normalized RMS Error", 'scale': 'log'},
        utils.COL_R2: {'label': "R^2", 'scale': 'linear'},
        utils.COL_TIME: {'label': "Elapsed Time (s)", 'scale': 'linear'} 
    }
    metrics_to_plot = list(plot_config.keys()) 
    required_columns = [utils.COL_B, utils.COL_D_NORM] + metrics_to_plot

    # Input validation
    if df is None or df.empty:
        logger.warning(f"plot_biparametric: DataFrame for '{title}' is empty or None. Skipping plot.")
        return
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        logger.warning(f"plot_biparametric: DataFrame for '{title}' missing columns: {missing_cols}. Skipping plot.")
        return

    # Data preparation (pivoting)
    pivot_tables = {}
    B, DELTA = None, None
    try:
        df_plot = df[required_columns].copy()
        for col in required_columns: # Ensure numeric types before pivoting
            df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')
        df_plot = df_plot.dropna()

        if df_plot.empty:
            logger.warning(f"plot_biparametric: DataFrame for '{title}' empty after converting/dropping NaNs. Skipping plot.")
            return

        for metric in metrics_to_plot:
            try:
                pivot_tables[metric] = df_plot.pivot_table(index=utils.COL_D_NORM, columns=utils.COL_B, values=metric)
            except Exception as pivot_err:
                logger.error(f"Error pivoting data for metric '{metric}' in '{title}': {pivot_err}. Skipping plot for this metric.")

        if not pivot_tables:
            logger.warning(f"plot_biparametric: No valid pivot tables could be created for '{title}'. Skipping plot.")
            return

        first_metric = next(iter(pivot_tables))
        delta_values = pivot_tables[first_metric].index.to_numpy()
        b_values = pivot_tables[first_metric].columns.to_numpy()

        if len(delta_values) < 2 or len(b_values) < 2:
            logger.warning(f"plot_biparametric: Need >= 2 unique delta/bias values after pivoting for '{title}'. Skipping contour plot.")
            return

        B, DELTA = np.meshgrid(b_values, delta_values)

    except Exception as e:
        logger.error(f"Error preparing data for plot_biparametric '{title}': {e}. Skipping plot.", exc_info=True)
        return

    # Plotting
    metrics_actually_plotted = [m for m in metrics_to_plot if m in pivot_tables]
    num_plots = len(metrics_actually_plotted)
    if num_plots == 0: return 

    fig, axes = None, None
    try:
        with plt.style.context(_STYLE_NAME_TO_USE):
            fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5), squeeze=False)
            axes = axes.flatten()

            plot_index = 0
            for metric_key in metrics_actually_plotted:
                    cfg = plot_config[metric_key]
                    data = pivot_tables[metric_key].to_numpy()
                    ax = axes[plot_index]

                    # Handle non-finite values and LogNorm safely
                    norm = cfg['norm']
                    if isinstance(norm, LogNorm):
                        # Ensure positive values exist for LogNorm
                        if not np.any(data[np.isfinite(data)] > 0):
                            logger.warning(f"\tLogNorm specified for '{metric_key}' but no positive data found. Using linear norm.")
                            norm = None # Fallback to linear norm
                        else:
                            # Mask non-positive for LogNorm contouring
                            data = np.ma.masked_where(data <= 0, data)

                    # General masking for non-finite
                    if not np.all(np.isfinite(data)):
                        if not isinstance(data, np.ma.MaskedArray): # Avoid double masking
                            data = np.ma.masked_invalid(data)
                        logger.warning(f"\tNon-finite values found in data for '{metric_key}' ({title}). Masking for contour plot.")

                    # Check if all data is masked
                    if hasattr(data, 'mask') and data.mask.all():
                        logger.warning(f"\tAll data is invalid/masked for '{metric_key}' ({title}). Skipping contour.")
                        ax.set_title(f"{cfg['title']}\n(No valid data)")
                    else:
                        try:
                            contour = ax.contourf(B, DELTA, data, cmap=cfg['cmap'], norm=norm, levels=20, antialiased=True)
                            cbar = fig.colorbar(contour, ax=ax, label=cfg['label'])
                            cbar.ax.tick_params(labelsize='small')
                            # Add contour lines for clarity
                            try: # Contour lines might fail on some data
                                contour_lines = ax.contour(B, DELTA, data, colors='white', linewidths=0.5, levels=10, norm=norm, antialiased=True)
                            except ValueError: pass # Ignore if contour lines fail
                        except Exception as contour_err:
                            logger.error(f"\tError generating contour plot for '{metric_key}' ({title}): {contour_err}", exc_info=False)
                            ax.set_title(f"{cfg['title']}\n(Contour Error)")

                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.set_xlabel(f"{utils.COL_B}")
                    ax.set_ylabel(f"{utils.COL_D_NORM}")
                    if ax.get_title() == "": 
                        ax.set_title(cfg['title'])
                    plot_index += 1

            fig.suptitle(title, fontsize=14)
            fig.tight_layout(rect=[0, 0.03, 1, 0.93])

    except Exception as e:
        logger.error(f"Error generating plot_biparametric '{title}': {e}", exc_info=True)
    finally:
        _save_and_close_plot(fig, filepath)


# --- Nyquist Plots ---
def plot_nyquist_reconstruction(t_orig: np.ndarray, u_orig: np.ndarray, t_sampled: np.ndarray, u_sampled: np.ndarray, u_reconstructed: np.ndarray, title: str, save_path: Optional[str] = None) -> None:
    '''Plots original signal, sampled points, and spline reconstruction.'''
    # Input validation
    if t_orig is None or u_orig is None or t_sampled is None or u_sampled is None or u_reconstructed is None:
        logger.warning(f"plot_nyquist_reconstruction: Missing input data for '{title}'. Skipping.")
        return

    fig, ax = None, None
    try:
        with plt.style.context(_STYLE_NAME_TO_USE):
            fig, ax = plt.subplots(figsize=(10, 5)) 
            ax.plot(t_orig, u_orig, label="Original Signal", linewidth=BASE_LINEWIDTH, color=COLOR_ORIGINAL, alpha=0.9, zorder=1)
            ax.plot(t_orig, u_reconstructed, label="Reconstructed (Spline)", linestyle='--', linewidth=BASE_LINEWIDTH, color=COLOR_NYQUIST, zorder=2)
            ax.plot(t_sampled, u_sampled, 'o', label=f"Sampled Points (N={len(t_sampled)})", markersize=BASE_MARKERSIZE, color=COLOR_NYQUIST, markerfacecolor='white', markeredgewidth=0.8, zorder=3)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_title(title)
            ax.legend(frameon=False, loc='best')
            ax.grid(True, linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)
            fig.tight_layout()
    except Exception as e:
        logger.error(f"Error during plot_nyquist_reconstruction ('{title}'): {e}", exc_info=False)
    finally:
        _save_and_close_plot(fig, save_path)


def plot_nyquist_error_profile(t: np.ndarray, errors: np.ndarray, title: str, save_path: Optional[str] = None) -> None:
    '''Plots the reconstruction error profile for Nyquist analysis.'''
    # Input validation
    if t is None or errors is None:
        logger.warning(f"plot_nyquist_error_profile: Missing input data for '{title}'. Skipping.")
        return

    fig, ax = None, None
    try:
        with plt.style.context(_STYLE_NAME_TO_USE):
            fig, ax = plt.subplots(figsize=(10, 4)) 
            ax.plot(t, errors, linewidth=BASE_LINEWIDTH*0.9, color=COLOR_ERROR)
            ax.axhline(0, color='black', linestyle='-', linewidth=0.7, alpha=0.6)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Error (Original - Reconstructed)")
            ax.set_title(title)
            ax.grid(True, linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)
            # Add max/median error text (optional, but helpful)
            if errors.size > 0:
                max_abs_err = np.max(np.abs(errors))
                med_abs_err = np.median(np.abs(errors))
                ax.text(0.98, 0.95, f'Max|Err|: {max_abs_err:.2e}\nMed|Err|: {med_abs_err:.2e}',
                        transform=ax.transAxes, ha='right', va='top', fontsize='small',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
            fig.tight_layout()
    except Exception as e:
        logger.error(f"Error during plot_nyquist_error_profile ('{title}'): {e}", exc_info=False)
    finally:
        _save_and_close_plot(fig, save_path)


def plot_nyquist_summary(df_summary: pd.DataFrame, error_cols_to_plot: Sequence[str] = (COL_MAX_ERR, COL_MED_ERR, COL_RMSE), title: str = "Nyquist Analysis Summary", save_path: Optional[str] = None, log_y: bool = True) -> None:
    '''Plots summary error metrics vs. N_Points from Nyquist analysis.'''
    param_col = COL_N_POINTS
    # Input validation
    if df_summary is None or df_summary.empty: logger.warning(f"plot_nyquist_summary: DataFrame empty. Skipping."); return
    if param_col not in df_summary.columns: logger.warning(f"plot_nyquist_summary: Param col '{param_col}' not found. Skipping."); return
    if not error_cols_to_plot: logger.warning(f"plot_nyquist_summary: No error cols specified. Skipping."); return

    # Data preparation
    plot_cols = [param_col] + [col for col in error_cols_to_plot if col in df_summary.columns]
    missing_cols = [col for col in error_cols_to_plot if col not in df_summary.columns]
    if missing_cols: logger.warning(f"plot_nyquist_summary: Skipping missing error columns: {missing_cols}")
    if len(plot_cols) <= 1: logger.warning(f"plot_nyquist_summary: No valid error columns found. Skipping."); return
    df_plot = None
    try:
        df_plot = df_summary[plot_cols].copy()
        for col in df_plot.columns: df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')
        df_plot = df_plot.dropna().sort_values(by=param_col)
        if df_plot.empty: logger.warning(f"plot_nyquist_summary: No valid numeric data found. Skipping."); return
    except Exception as e: logger.error(f"Error preparing data for plot_nyquist_summary: {e}", exc_info=False); return

    # Plotting
    actual_error_cols = [col for col in plot_cols if col != param_col]
    num_errs = len(actual_error_cols)
    if num_errs == 0: return

    fig, axes = None, None
    try:
        with plt.style.context(_STYLE_NAME_TO_USE):
            fig, axes = plt.subplots(1, num_errs, figsize=(5 * num_errs, 4.5), squeeze=False)
            axes = axes.flatten()
            x_data = df_plot[param_col]
            for i, err_col in enumerate(actual_error_cols):
                ax = axes[i]
                y_data = df_plot[err_col]
                ax.plot(x_data, y_data, marker='.', linestyle='-', linewidth=BASE_LINEWIDTH*0.8, markersize=BASE_MARKERSIZE, color=COLOR_NYQUIST)

                current_use_log = log_y
                if log_y:
                    if np.any(y_data <= 1e-15):
                        logger.warning(f"\tNon-positive values for '{err_col}'. Plotting Nyquist summary on linear y-scale.")
                        current_use_log = False
                    else:
                        ax.set_yscale('log')

                ax.set_title(f"{err_col} vs N")
                ax.set_xlabel("Number of Samples (N)")
                ax.set_ylabel(err_col)
                ax.grid(True, which='both' if current_use_log else 'major', linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)

            fig.suptitle(title, fontsize=14)
            fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    except Exception as e:
        logger.error(f"Error during plot_nyquist_summary ('{title}'): {e}", exc_info=False)
    finally:
        _save_and_close_plot(fig, save_path)


# --- Fourier Plots ---
def plot_fourier_spectrum(frequencies: np.ndarray, spectrum_mag: np.ndarray, title: str, save_path: Optional[str] = None, label: Optional[str] = None, log_y: bool = True, freq_limit: Optional[float] = None, color: Optional[str] = None) -> None:
    '''Plots a single frequency spectrum magnitude.'''
    # Input validation
    if frequencies is None or spectrum_mag is None: logger.warning(f"plot_fourier_spectrum: Missing data for '{title}'. Skipping."); return
    if frequencies.shape != spectrum_mag.shape: logger.warning(f"plot_fourier_spectrum: Shape mismatch Freq vs Mag for '{title}'. Skipping."); return

    fig, ax = None, None
    plot_color = color if color else COLOR_ORIGINAL
    try:
        with plt.style.context(_STYLE_NAME_TO_USE):
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.plot(frequencies, spectrum_mag, label=label, linewidth=BASE_LINEWIDTH*0.9, color=plot_color)
            ax.set_title(title)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude")

            # Set frequency limits
            min_freq = 0 
            max_freq_data = np.max(frequencies) if frequencies.size > 0 else 1
            actual_freq_limit = freq_limit if freq_limit is not None and freq_limit > min_freq else max_freq_data
            ax.set_xlim(min_freq, actual_freq_limit)

            # Handle log scale carefully
            current_use_log = log_y
            if log_y:
                min_positive_mag = np.min(spectrum_mag[spectrum_mag > 1e-15]) if np.any(spectrum_mag > 1e-15) else None
                if min_positive_mag is not None:
                    ax.set_yscale('log')
                    # Optionally adjust y-limits for log scale for better visibility
                    max_mag = np.max(spectrum_mag)
                    ax.set_ylim(bottom=min_positive_mag * 0.1, top=max_mag * 1.5)
                else:
                    logger.warning(f"\tCannot set log scale for y-axis in '{title}': No magnitude values > 1e-15.")
                    current_use_log = False

            ax.grid(True, which='both' if current_use_log else 'major', linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)
            if label:
                ax.legend(frameon=False)
            fig.tight_layout()
    except Exception as e:
        logger.error(f"Error during plot_fourier_spectrum ('{title}'): {e}", exc_info=False)
    finally:
        _save_and_close_plot(fig, save_path)


def plot_fourier_combined_spectrum(spectra_data: List[Dict[str, Any]], title: str, save_path: Optional[str] = None, log_y: bool = True, freq_limit: Optional[float] = None) -> None:
    '''Plots 2 or 3 frequency spectrum magnitudes on the same axes for comparison.'''
    # Input validation
    if not spectra_data or not isinstance(spectra_data, list) or len(spectra_data) < 2: logger.warning(f"plot_fourier_combined_spectrum: Needs >= 2 spectra for '{title}'. Skipping."); return
    if len(spectra_data) > 3: logger.warning(f"plot_fourier_combined_spectrum: Max 3 spectra supported ('{title}'). Plotting first 3."); spectra_data = spectra_data[:3]

    # Default styles
    default_styles = [
        {'color': COLOR_ORIGINAL, 'linestyle': '-', 'linewidth': BASE_LINEWIDTH, 'alpha': 0.9, 'zorder': 3},
        {'color': COLOR_OPTIMAL_ASDM, 'linestyle': '--', 'linewidth': BASE_LINEWIDTH, 'alpha': 0.85, 'zorder': 2},
        {'color': COLOR_NYQUIST, 'linestyle': ':', 'linewidth': BASE_LINEWIDTH*0.9, 'alpha': 0.8, 'zorder': 1}
    ]

    fig, ax = None, None
    all_magnitudes = []
    max_freq = 0
    try:
        with plt.style.context(_STYLE_NAME_TO_USE):
            fig, ax = plt.subplots(figsize=(10, 5.5))
            for i, spec_info in enumerate(spectra_data):
                freq = spec_info.get('freq')
                mag = spec_info.get('mag')
                label = spec_info.get('label', f'Spectrum {i+1}')

                if freq is None or mag is None or freq.size == 0 or mag.size == 0 or freq.shape != mag.shape:
                    logger.warning(f"\tSkipping spectrum '{label}' in '{title}' due to missing/invalid data.")
                    continue

                # Style application
                style_args = default_styles[i].copy()
                style_args.update({k: v for k, v in spec_info.items() if k in ['color', 'linestyle', 'linewidth', 'alpha', 'zorder']}) # Apply overrides

                ax.plot(freq, mag, label=label, **style_args)
                all_magnitudes.append(mag) 
                max_freq = max(max_freq, np.max(freq)) if freq.size > 0 else max_freq

            ax.set_title(title)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude")

            # Handle log scale based on combined valid data
            current_use_log = log_y
            if log_y:
                combined_mags = np.concatenate([m[m > 1e-15] for m in all_magnitudes if np.any(m > 1e-15)])
                if combined_mags.size > 0:
                    ax.set_yscale('log')
                    min_val = np.min(combined_mags)
                    max_val = np.max(np.concatenate(all_magnitudes))
                    ax.set_ylim(bottom=min_val * 0.1, top=max_val * 1.5)
                else:
                    logger.warning(f"\tCannot set log scale for y-axis in '{title}': No combined magnitude values > 1e-15.")
                    current_use_log = False

            # Set frequency limit
            ax.set_xlim(0, freq_limit if freq_limit is not None and freq_limit > 0 else max_freq)

            ax.grid(True, which='both' if current_use_log else 'major', linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)
            ax.legend(frameon=False)
            fig.tight_layout()

    except Exception as e:
        logger.error(f"Error during plot_fourier_combined_spectrum ('{title}'): {e}", exc_info=False)
    finally:
        _save_and_close_plot(fig, save_path)


def plot_principal_frequencies(principal_data: List[Dict[str, Any]], title: str, save_path: Optional[str] = None) -> None:
    '''Plots principal frequencies and their magnitudes for multiple signals using stem plots.'''
    # --- Basic Input Validation ---
    if not principal_data or not isinstance(principal_data, list):
        logger.warning(f"plot_principal_frequencies: Invalid or empty `principal_data` list for '{title}'. Skipping.")
        return

    # --- Plotting Setup ---
    fig, ax = None, None
    # Use defined colors directly assuming order: Original, Optimal, Nyquist
    plot_colors = [COLOR_ORIGINAL, COLOR_OPTIMAL_ASDM, COLOR_NYQUIST]

    try:
        with plt.style.context(_STYLE_NAME_TO_USE):
            fig, ax = plt.subplots(figsize=(10, 5))

            for i, data_dict in enumerate(principal_data):
                # Limit to 3 datasets for defined colors
                if i >= len(plot_colors):
                    logger.warning(f"\tplot_principal_frequencies: Plotting only first {len(plot_colors)} datasets for '{title}'.")
                    break

                freqs = data_dict.get('freqs')
                mags = data_dict.get('mags')
                label = data_dict.get('label', f'Data {i+1}')

                # Essential validation for data needed for plotting
                if not isinstance(freqs, np.ndarray) or not isinstance(mags, np.ndarray) or \
                   freqs.size == 0 or freqs.shape != mags.shape:
                    logger.warning(f"\tplot_principal_frequencies: Skipping '{label}' in '{title}' due to invalid/empty freqs or mags.")
                    continue

                plot_color = plot_colors[i]

                # Create the stem plot
                try:
                    stem_container = ax.stem(
                        freqs, mags,
                        linefmt='-',      # Solid line style
                        markerfmt='o',    # Circle marker style
                        basefmt=' ',      # No baseline drawn
                        label=label       # Label for the legend
                    )
                    plt.setp(stem_container.markerline, 'markerfacecolor', plot_color)
                    plt.setp(stem_container.markerline, 'markeredgecolor', plot_color) 
                    plt.setp(stem_container.stemlines, 'color', plot_color)

                except Exception as stem_err:
                    logger.error(f"\tError creating stem plot for '{label}' in '{title}': {stem_err}", exc_info=False)
                    continue

            ax.set_title(title)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude")
            ax.legend(frameon=False, loc='best')
            ax.grid(True, linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)
            ax.set_ylim(bottom=0) # Ensure y-axis starts at 0

            fig.tight_layout()

    except Exception as e:
        logger.error(f"Error during plot_principal_frequencies ('{title}'): {e}", exc_info=True)
    finally:
        _save_and_close_plot(fig, save_path)
