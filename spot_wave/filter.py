# -*- coding: utf-8 -*-

import numpy as np
from .utils import wavepal_analyze

try:
    from gls import Gls
except ImportError:  # pragma: no cover
    Gls = None


MIN_PERIOD_ANALYSIS_DEFAULT = 1.0
MAX_PERIOD_ANALYSIS_DEFAULT = 200.0


def _require_gls():
    if Gls is None:
        raise ImportError(
            "Could not import `Gls`. Install/add your `gls.py` module "
            "(Zechmeister & Kurster periodogram) to the PYTHONPATH, just "
            "like in analyze_filter_sweep_v3.py."
        )


def make_w0_grid(w0_min, w0_max, w0_step):
    """Shortcut for np.arange(w0_min, w0_max+step, step), same as `_grid()`."""
    return np.arange(w0_min, w0_max + w0_step, w0_step)


def _as_list(x):
    """Normalizes a scalar or an iterable into a plain list."""
    if np.isscalar(x):
        return [x]
    return list(x)

###############################
"""EMPIRICAL S_score FOR GLS"""
###############################

def compute_empirical_gls_score(time_arr, residuals, rv_err, p_rot, p_rot_half,
                                 p_planets, permin=MIN_PERIOD_ANALYSIS_DEFAULT, permax=MAX_PERIOD_ANALYSIS_DEFAULT):
    """
    This is one of the main parts from the code where the S_score is computed, 
    which is used to evaluate the quality of a filtering combination.
    A low S_score means activity has been suppressed well without
    wrecking the planet signal(s), i.e. a better filtering combination.

    S_score = eta_activity / eta_planet, where:
      eta_activity = GLS_power(Prot)/FAP_99 + GLS_power(Prot/2)/FAP_99
      eta_planet   = sum_i GLS_power(P_planet_i)/FAP_99

    Arguments:
    time_arr : np.ndarray
        Time array of the RV observations.
    residuals : np.ndarray
        Residuals of the RV observations.
    rv_err : np.ndarray
        Errors of the RV observations.
    p_rot : float
        Period (days) of the rotation signal.
    p_rot_half : float
        Period (days) of the half-rotation signal.
    p_planets : float or list[float]
        Period (days) of a single planet, or a list of periods if there
        is more than one planet in the system.
    permax : float, optional
        Maximum period to analyze (default is MAX_PERIOD_ANALYSIS_DEFAULT).
    """
    _require_gls()
    gls_real = Gls((time_arr, residuals, rv_err), Pbeg=permin, Pend=permax, verbose=False, fast=True)
    fap_99 = gls_real.powerLevel(0.01)

    f_rot = 1.0 / p_rot
    f_half = 1.0 / p_rot_half
    idx_rot = np.argmin(np.abs(gls_real.f - f_rot))
    idx_half = np.argmin(np.abs(gls_real.f - f_half))
    power_rot = gls_real.power[idx_rot]
    power_half = gls_real.power[idx_half]

    if fap_99 > 0:
        eta_activity = (power_rot / fap_99) + (power_half / fap_99)
    else:
        eta_activity = np.nan

    eta_planet = 0.0
    for p_pl in _as_list(p_planets):
        f_planet = 1.0 / p_pl
        idx_planet = np.argmin(np.abs(gls_real.f - f_planet))
        contribution = gls_real.power[idx_planet] / fap_99 if fap_99 > 0 else np.nan
        eta_planet += contribution
    if eta_planet == 0 or np.isnan(eta_planet):
        eta_planet = 1e-10

    s_score = eta_activity / eta_planet
    return float(s_score), float(eta_activity), float(eta_planet)


####################################################
"""BAND FILTERING (one w0, and one or more bands)"""
####################################################

def _as_band_list(bands):
    """
    Normalizes `bands` into a list of (lo, hi) tuples.
    Accepts: a single (lo, hi) tuple/list, OR a list of such tuples
    [(lo1, hi1), (lo2, hi2), ...].
    """
    bands = list(bands)
    is_single_band = len(bands) == 2 and all(np.isscalar(x) for x in bands)
    if is_single_band:
        return [tuple(bands)]
    return [tuple(b) for b in bands]


def filter_bands_once(t, y, w0, bands, permin=MIN_PERIOD_ANALYSIS_DEFAULT,
                       permax=MAX_PERIOD_ANALYSIS_DEFAULT):
    """
    Runs wavepal ONCE with a given w0 and filters one or more period
    bands simultaneously (all removed together, in a single
    timefreq_band_filtering call).

    Arguments:
    t : np.ndarray
        Time array of the RV observations.
    y : np.ndarray
        RV array of the observations.
    w0 : float
        Wavelet parameter for wavepal.
    bands : tuple(float, float) or list[tuple(float, float)]
        Period band(s) to filter, in days, e.g. (Prot-0.5, Prot+0.5), or [(Prot-0.5, Prot+0.5), (P2-0.5, P2+0.5), ...] to remove several period ranges at once with the same w0.
    permin : float, optional
        Minimum period to analyze (default is MIN_PERIOD_ANALYSIS_DEFAULT).
    permax : float, optional
        Maximum period to analyze (default is MAX_PERIOD_ANALYSIS_DEFAULT).
    """
    band_list = _as_band_list(bands)
    wave = wavepal_analyze(t, y, w0, permin=permin, permax=permax)
    wave.timefreq_band_filtering(band_list)
    if wave.timefreq_band_filtered_signal is None:
        return np.zeros_like(y)
    sig = wave.timefreq_band_filtered_signal[:, :len(band_list)].sum(axis=1)
    sig = sig - np.mean(sig)
    return sig


def filter_band_once(t, y, w0, band, permin=MIN_PERIOD_ANALYSIS_DEFAULT,
                      permax=MAX_PERIOD_ANALYSIS_DEFAULT):
    """Backward-compatible alias for filter_bands_once() (single or multi band)."""
    return filter_bands_once(t, y, w0, band, permin=permin, permax=permax)


##################################################################
"""MODE 1: SIMPLE FILTER (a single w0 range, one or more bands)"""
##################################################################

def single_filter_sweep(t, rv, rv_err, w0_grid, bands, p_rot, p_rot_half,
                         p_planets, permin=MIN_PERIOD_ANALYSIS_DEFAULT,
                         permax=MAX_PERIOD_ANALYSIS_DEFAULT, verbose=True):
    """
    Sweeps a SINGLE w0 range over one or more bands (e.g. the Prot band,
    or several bands at once), for a single target. For each w0, all
    `bands` are filtered together (one wavepal call, see
    `filter_bands_once`), the residual and its S_score are computed, and
    the winning combination (lowest S_score) is returned.

    Arguments:
    t, rv, rv_err : np.ndarray
        RV time series (already loaded, e.g. with utils.load_rv_file).
    w0_grid : array-like
        w0 values to sweep (use make_w0_grid(w0_min, w0_max, w0_step)).
    bands : tuple(float, float) or list[tuple(float, float)]
        Period band(s) to filter, in days, e.g. (Prot-0.5, Prot+0.5), or
        [(Prot-0.5, Prot+0.5), (P2-0.5, P2+0.5), ...] to remove several
        period ranges at once with the same w0.
    p_rot, p_rot_half : float
        Rotation period and its half (days), used for eta_activity.
    p_planets : float or list[float]
        Period(s) (days) of the planet(s) in the system, used for
        eta_planet. With several planets, eta_planet is their sum (see
        `compute_empirical_gls_score`).
    permin, permax : float, optional
        Minimum and maximum periods to analyze (default is MIN_PERIOD_ANALYSIS_DEFAULT and MAX_PERIOD_ANALYSIS_DEFAULT).
    verbose : bool, optional
        If True, prints progress and the S_score for each w0.
    """
    combos = []
    for w0 in w0_grid:
        sig = filter_bands_once(t, rv, w0, bands, permin=permin, permax=permax)
        residuals = rv - sig
        s_score, eta_act, eta_plan = compute_empirical_gls_score(
            t, residuals, rv_err, p_rot, p_rot_half, p_planets, permin=permin, permax=permax
        )
        combos.append({
            "w0": float(w0), "residuals": residuals,
            "S_score": s_score, "eta_activity": eta_act, "eta_planeta": eta_plan,
        })
        if verbose:
            print(f"[single_filter_sweep] w0={w0:.3f} -> S_score={s_score:.4g}")

    best = min(combos, key=lambda c: c["S_score"])
    best["n_combos_evaluated"] = len(combos)
    if verbose:
        print(f"[single_filter_sweep] BEST w0={best['w0']:.3f} -> S_score={best['S_score']:.4g}")
    return best


########################################################################
"""MODE 2: DOUBLE FILTER (two w0 ranges, two bands, order 1 and/or 2)"""
########################################################################

def double_filter_grid(t, rv, rv_err, order, w0_grid_1, w0_grid_2, band_1, band_2,
                        permin=MIN_PERIOD_ANALYSIS_DEFAULT,
                        permax=MAX_PERIOD_ANALYSIS_DEFAULT):
    """
    Version of double filtering: for each w0_1, filter band_1
    (one wavepal call, cached), and for each w0_2, filter band_2 on top of
    the previous residual. Nothing is written to disk.

    order=1 -> band_1 = Prot band,     band_2 = Prot/2 band (e.g. One Active Long "Power_Prot>Power_Prot/2")
    order=2 -> band_1 = Prot/2 band,   band_2 = Prot band (e.g. Two Active Longs or Random "Power_Prot/2>Power_Prot")

    band_1 and band_2 can each be a single (lo, hi) tuple or a list of
    them (see `filter_bands_once`), if you need to remove more than one
    period range at the same w0 step.

    Arguments:
    t, rv, rv_err : np.ndarray
        RV time series (already loaded, e.g. with utils.load_rv_file).
    order : int
        1 or 2, see above.
    w0_grid_1, w0_grid_2 : array-like
        w0 values to sweep for the first and second filter, respectively
        (use make_w0_grid(w0_min, w0_max, w0_step)).
    band_1, band_2 : tuple(float, float) or list[tuple(float, float)]
        Period band(s) to filter for the first and second filter, respectively, in days, e.g. (Prot-0.5, Prot+0.5), or [(Prot-0.5, Prot+0.5), (P2-0.5, P2+0.5), ...] 
        to remove several period ranges at once with the same w0.
    permin, permax : float, optional
        Minimum and maximum periods to analyze (default is MIN_PERIOD_ANALYSIS_DEFAULT and MAX_PERIOD_ANALYSIS_DEFAULT).
    """
    out = []
    for w0_1 in w0_grid_1:
        sig_1 = filter_bands_once(t, rv, w0_1, band_1, permin=permin, permax=permax)
        filtered_1 = rv - sig_1

        for w0_2 in w0_grid_2:
            sig_2 = filter_bands_once(t, filtered_1, w0_2, band_2, permin=permin, permax=permax)
            residuals = filtered_1 - sig_2

            out.append({
                "w0_1": float(w0_1), "w0_2": float(w0_2), "order": order,
                "residuals": residuals,
            })
    return out


def run_double_filter_sweep(t, rv, rv_err, prot_value, prot_half_value, planet_periods,
                             order1=None, order2=None,
                             permin=MIN_PERIOD_ANALYSIS_DEFAULT,
                             permax=MAX_PERIOD_ANALYSIS_DEFAULT, verbose=True):
    """
    DOUBLE filtering sweep, trying order 1 (Prot
    first), order 2 (Prot/2 first), or both, and keeping the overall
    combination with the lowest S_score.

    Arguments:
    t, rv, rv_err : np.ndarray
        RV time series.
    prot_value, prot_half_value : float
        Rotation period and its half, in days.
    planet_periods : float or list[float]
        Period(s) (days) of the planet(s) in the system. With several
        planets, eta_planet in the S_score is their sum (see
        `compute_empirical_gls_score`).
    order1, order2 : dict or None
        Each with the keys:
            w0_grid_1, w0_grid_2 : array-like (via make_w0_grid)
            hw_1, hw_2           : band half-width (days) for each filter
        order1: w0_grid_1/hw_1 filter the Prot band (first),
                w0_grid_2/hw_2 filter the Prot/2 band (second).
        order2: w0_grid_1/hw_1 filter the Prot/2 band (first),
                w0_grid_2/hw_2 filter the Prot band (second).
        Pass None to skip that order. At least one of the two must be
        given.
    permin, permax : float, optional
        Minimum and maximum periods to analyze (default is MIN_PERIOD_ANALYSIS_DEFAULT and MAX_PERIOD_ANALYSIS_DEFAULT).
    verbose : bool, optional
        If True, prints progress and the S_score for each w0 combination.
    """
    if order1 is None and order2 is None:
        raise ValueError("You must provide at least order1 or order2.")

    all_combos = []

    if order1 is not None:
        band_prot = (prot_value - order1["hw_1"], prot_value + order1["hw_1"])
        band_half = (prot_half_value - order1["hw_2"], prot_half_value + order1["hw_2"])
        all_combos.extend(double_filter_grid(
            t, rv, rv_err, order=1,
            w0_grid_1=order1["w0_grid_1"], w0_grid_2=order1["w0_grid_2"],
            band_1=band_prot, band_2=band_half, permin=permin, permax=permax,
        ))

    if order2 is not None:
        band_half = (prot_half_value - order2["hw_1"], prot_half_value + order2["hw_1"])
        band_prot = (prot_value - order2["hw_2"], prot_value + order2["hw_2"])
        all_combos.extend(double_filter_grid(
            t, rv, rv_err, order=2,
            w0_grid_1=order2["w0_grid_1"], w0_grid_2=order2["w0_grid_2"],
            band_1=band_half, band_2=band_prot, permin=permin, permax=permax,
        ))

    best = None
    for combo in all_combos:
        s_score, eta_act, eta_plan = compute_empirical_gls_score(
            t, combo["residuals"], rv_err, prot_value, prot_half_value, planet_periods,
            permax=permax, permin=permin
        )
        combo["S_score"] = s_score
        combo["eta_activity"] = eta_act
        combo["eta_planeta"] = eta_plan
        if verbose:
            print(f"[double_filter_sweep] order={combo['order']} "
                  f"w0_1={combo['w0_1']:.3f} w0_2={combo['w0_2']:.3f} "
                  f"-> S_score={s_score:.4g}")
        if best is None or s_score < best["S_score"]:
            best = combo

    best["n_combos_evaluated"] = len(all_combos)
    if verbose:
        print(f"[double_filter_sweep] BEST order={best['order']} "
              f"w0_1={best['w0_1']:.3f} w0_2={best['w0_2']:.3f} "
              f"-> S_score={best['S_score']:.4g}")
    return best


def save_winner_file(t, residuals, rv_err, output_path):
    """Saves the winning residual to disk (format compatible with CONAN.load_rvs)."""
    import pandas as pd
    pd.DataFrame({"time": t, "residuals": residuals, "rv_err": rv_err}).to_csv(
        output_path, sep=" ", index=False, header=False
    )
    return output_path
