# -*- coding: utf-8 -*-
"""
utils.py
Support functions for spot_wave: locating CARMCMC, loading RV files,
checking CWT feasibility, and a wrapper around wavepal.timefreq_analysis.
"""

import os
import sys
import numpy as np
import wavepal as wv
import matplotlib.pyplot as plt


_CARMCMC_LOADED = False


def setup_carmcmc(path=None, verbose=True):
    """
    Adds the path to the carma_pack (CARMCMC) build to sys.path and
    imports the module, just like cell 1 of SPOT_WAVE.ipynb.

    Parameters
    ----------
    path : str, optional
        Path to .../carmcmc/carma_pack/src . If not given, the
        SPOT_WAVE_CARMCMC_PATH environment variable is used if it exists.
    verbose : bool
        If True, print the confirmation message (as in the notebook).

    Returns
    -------
    carmcmc module, or None if it could not be loaded.
    """
    global _CARMCMC_LOADED

    rute_carmcmc = path or os.environ.get("SPOT_WAVE_CARMCMC_PATH")
    if rute_carmcmc and rute_carmcmc not in sys.path:
        sys.path.insert(0, rute_carmcmc)

    try:
        import carmcmc  # noqa: F401
    except ImportError as e:
        if verbose:
            print("[spot_wave] Warning: could not import carmcmc ({0}). "
                  "Set SPOT_WAVE_CARMCMC_PATH or pass `path=` to "
                  "setup_carmcmc().".format(e))
        return None

    _CARMCMC_LOADED = True
    if verbose:
        print("CARMCMC has been successfully loaded with all its C++ libraries!")
    return carmcmc


#####################
"""RV DATA LOADING"""
#####################

def load_rv_file(filename, subtract_instrument_means=True, avoid_time_collisions=True,
                  eps=1.1e-5):
    """
    Loads an RV file:
    detects whether there is a header, detects the number of columns, and
    if there is an instrument column (4th column) subtracts the mean RV
    per instrument.
    
    Expected column layout: time, RV, RV_err[, instrument]

    Arguments:
    filename : str
        Path to the RV .dat file.
    subtract_instrument_means : bool
        If there are several instruments, subtract each one's mean (offset).
    avoid_time_collisions : bool
        If True, shift duplicated/non-increasing timestamps by `eps`
    eps : float
        Time shift used to avoid collisions.
    """
    with open(filename) as f:
        first_line = f.readline().strip()

    try:
        float(first_line.split()[0])
        skip_header = 0
    except (ValueError, IndexError):
        skip_header = 1

    data = np.genfromtxt(filename, dtype=None, encoding=None, skip_header=skip_header)
    if data.ndim == 0:
        data = np.array([data])

    n_cols = len(data[0])
    t = np.array([row[0] for row in data], dtype=float)
    rv = np.array([row[1] for row in data], dtype=float)
    rv_err = np.array([row[2] for row in data], dtype=float) if n_cols >= 3 \
        else np.zeros_like(rv)

    if n_cols >= 4:
        instruments = np.array([row[3] for row in data], dtype=str)
    else:
        instruments = np.array(["single_instrument"] * len(t))

    unique_instruments = np.unique(instruments)
    if subtract_instrument_means and len(unique_instruments) > 1:
        for inst in unique_instruments:
            mask = instruments == inst
            mean_rv = np.mean(rv[mask])
            rv[mask] -= mean_rv

    sort_idx = np.argsort(t)
    t, rv, rv_err, instruments = t[sort_idx], rv[sort_idx], rv_err[sort_idx], instruments[sort_idx]

    if avoid_time_collisions:
        t = t.copy()
        for i in range(1, len(t)):
            while t[i] <= t[i - 1]:
                t[i] += eps

    return t, rv, rv_err, instruments


###########################
"""INITIAL PLOTTING"""
###########################

def rv_plot(t, rv, rv_err=None, instruments=None, xlabel="Time (days)", ylabel="RV (m/s)", show=True):
    """
    RV plot function for quick visualization, supporting multiple instruments.
    """
    plt.figure(figsize=(10, 5))
    
    t = np.array(t)
    rv = np.array(rv)
    if rv_err is not None:
        rv_err = np.array(rv_err)
        
    if instruments is not None:
        instruments = np.array(instruments)
        unique_insts = np.unique(instruments)
        
        for inst in unique_insts:
            mask = (instruments == inst)
            err = rv_err[mask] if rv_err is not None else None
            plt.errorbar(t[mask], rv[mask], yerr=err, fmt='o', capsize=3, label=inst)
            
        plt.legend(loc="best")
        
    else:
        plt.errorbar(t, rv, yerr=rv_err, fmt='o', ecolor='gray', capsize=3)
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if show:
        plt.show()

###########################
"""EXTRACTING K INJECTED"""
###########################

def parse_k_true(filename, header_regex=None):
    """
    Extracts the "true" (injected) semi-amplitude K from the file's first
    line, if the file is synthetic and carries it in the header (format
    "K=...", "K_1: ...", etc.). Returns np.nan if not found. Useful to
    validate K recovery for a single synthetic target.
    """
    import re
    regex = header_regex or re.compile(r"K\w*\s*[=:]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
    with open(filename, "r") as f:
        header = f.readline().strip()
    m = regex.search(header)
    if m:
        return float(m.group(1)), header
    return np.nan, header


#####################
"""CWT FEASIBILITY"""
#####################

def test_cwt_feasibility(wave_obj, w0=8.5, verbose=True):
    """
    Quick sanity check to see if the dataset can handle a CWT with the
    given w0.

    Note: these are theoretical approximations:
    Alarcón Guerri, V., et al. 2026, TFG, Universitat de Barcelona 
    and Merino Tamaral, U., et al. 2026, TFG, Universitat de Barcelona.
    """
    N = wave_obj.t.size
    w0_min = 5.5
    w0_max = 0.327 * N

    time_diffs = np.diff(wave_obj.t)
    cadence = np.mean(time_diffs)
    Tb = wave_obj.t[-1] - wave_obj.t[0]

    P_min = 2.455 * cadence
    P_max = Tb / (1.135 * w0)

    if verbose:
        print("Preliminary CWT Feasibility Analysis\n")
        print(f"Total points (N): {N}")
        if N < 50:
            print("Heads up: Less than 50 points. No work to do here!")
        print(f"Recommended w0 range: [{w0_min}, {w0_max:.2f}]")
        if w0 < w0_min or w0 > w0_max:
            print(f"Warning: Your chosen w0 ({w0}) is outside the safe zone.\n")
        else:
            print(f"Selected w0 ({w0}) is within the optimal range.\n")
        print(f"Avg cadence: {cadence:.2f} days")
        print(f"Baseline (Tb): {Tb:.2f} days\n")
        print("Theoretical Limits")
        print("Disclaimer: These are just mathematical guidelines, not hard walls.\n")
        print(f"P_min (theoretical): ~{P_min:.2f} days")
        print(f"P_max (theoretical): ~{P_max:.2f} days")

    if N < 50:
        return False
    if P_min >= P_max:
        if verbose:
            print("\nRed light: Terrible cadence compared to the total baseline (P_min >= P_max).")
        return False

    if verbose:
        print("\nGreen light: Dataset looks solid to search for periods in that range.")
    return True


#####################
"""WAVEPAL WRAPPER"""
#####################

def wavepal_analyze(t, y, w0, permin=1.0, permax=200.0, deltaj=0.01,
                     percentile=(95., 99.9), t_units="days", mydata_units="m/s",
                     trend_degree=-1, verbose=False):
    """
    Creates and runs a wavepal.Wavepal object for (t, y) with a given w0.
    Arguments:
    t, y : array-like
        Time and RV arrays.
    w0 : float
        Wavelet parameter. See wavepal docs.
    permin, permax : float
        Minimum and maximum periods to analyze (in t_units).
    deltaj : float
        Wavelet scale resolution. See wavepal docs.
    percentile : tuple(float, float)
        Percentiles for the significance contours. See wavepal docs.
    t_units, mydata_units : str
        Units for time and data. See wavepal docs.
    trend_degree : int
        Degree of polynomial to fit the trend. -1 means no trend removal.
    verbose : bool
        If True, print the analysis parameters.
    """
    wave = wv.Wavepal(t, y, "Time", "RV", t_units=t_units, mydata_units=mydata_units)
    wave.check_data()
    wave.choose_trend_degree(trend_degree)
    wave.trend_vectors()
    wave.timefreq_analysis(
        theta=t, w0=float(w0), permin=permin, permax=permax, deltaj=deltaj,
        percentile=np.asarray(percentile),
        shannonnyquistexclusionzone=True,
        computes_amplitude=True, smoothing_coeff=0.0,
    )
    if verbose:
        print(f"[spot_wave] wavepal_analyze: w0={w0}, permin={permin}, permax={permax}")
    return wave
