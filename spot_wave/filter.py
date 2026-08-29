# -*- coding: utf-8 -*-
"""
filter.py
Wavelet filtering for a SINGLE target (not the systematic / multi-file
case).

Reimplements, for a single system, the logic of:
  - analyze_filter_sweep_v3.py      -> DOUBLE filtering (two bands, two
                                        w0's), sweeping order1 (Prot
                                        filtered first) and/or order2
                                        (Prot/2 filtered first).
  - analyze_filter_sweep_syst.py    -> post-processing / metrics.

and adds the SIMPLE mode (a single w0 range, a single band), which is
what you used before moving to double filtering.

In both modes, the score that decides the "winner" is the same empirical
S_score (eta_activity / eta_planet) computed via GLS, via
`compute_empirical_gls_score`.
"""

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


# ---------------------------------------------------------------------------
# EMPIRICAL SCORE (GLS)
# ---------------------------------------------------------------------------
def compute_empirical_gls_score(time_arr, residuals, rv_err, p_rot, p_rot_half,
                                 p_planeta, permax=MAX_PERIOD_ANALYSIS_DEFAULT):
    """
    Identical to compute_empirical_gls_score() in analyze_filter_sweep_v3.py.

    S_score = eta_activity / eta_planet, where:
      eta_activity = GLS_power(Prot)/FAP_99 + GLS_power(Prot/2)/FAP_99
      eta_planet   = GLS_power(P_planet)/FAP_99

    A low S_score means activity has been suppressed well without
    wrecking the planet signal, i.e. a better filtering combination.
    """
    _require_gls()
    gls_real = Gls((time_arr, residuals, rv_err), Pbeg=1, Pend=permax, verbose=False, fast=True)
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

    f_planet = 1.0 / p_planeta
    idx_planet = np.argmin(np.abs(gls_real.f - f_planet))
    eta_planet = gls_real.power[idx_planet] / fap_99 if fap_99 > 0 else np.nan
    if eta_planet == 0 or np.isnan(eta_planet):
        eta_planet = 1e-10

    s_score = eta_activity / eta_planet
    return float(s_score), float(eta_activity), float(eta_planet)


# ---------------------------------------------------------------------------
# SINGLE-BAND FILTERING
# ---------------------------------------------------------------------------
def filter_band_once(t, y, w0, band, permin=MIN_PERIOD_ANALYSIS_DEFAULT,
                      permax=MAX_PERIOD_ANALYSIS_DEFAULT):
    """
    Runs wavepal with a given w0 and filters ONE period band. Returns the
    filtered signal (mean-centered), same as `_filter_band_once()` in
    analyze_filter_sweep_v3.py.
    """
    wave = wavepal_analyze(t, y, w0, permin=permin, permax=permax)
    wave.timefreq_band_filtering([band])
    if wave.timefreq_band_filtered_signal is None:
        return np.zeros_like(y)
    sig = wave.timefreq_band_filtered_signal[:, 0]
    sig = sig - np.mean(sig)
    return sig


# ---------------------------------------------------------------------------
# MODE 1: SIMPLE FILTER (a single w0 range, a single band)
# ---------------------------------------------------------------------------
def single_filter_sweep(t, rv, rv_err, w0_grid, band, p_rot, p_rot_half,
                         p_planeta, permin=MIN_PERIOD_ANALYSIS_DEFAULT,
                         permax=MAX_PERIOD_ANALYSIS_DEFAULT, verbose=True):
    """
    Sweeps a SINGLE w0 range over a SINGLE band (e.g. the Prot band), for
    a single target. For each w0, the band is filtered, the residual and
    its S_score are computed, and the winning combination (lowest
    S_score) is returned.

    This is the "single-pass" analog to double filtering: use it when
    filtering a single band (typically the Prot one) is already enough
    to separate activity from the planetary signal.

    Parameters
    ----------
    t, rv, rv_err : np.ndarray
        RV time series (already loaded, e.g. with utils.load_rv_file).
    w0_grid : array-like
        w0 values to sweep (use make_w0_grid(w0_min, w0_max, w0_step)).
    band : tuple(float, float)
        Period band to filter, in days, e.g. (Prot-0.5, Prot+0.5).
    p_rot, p_rot_half, p_planeta : float
        Rotation period, its half, and the planet period (days), used
        for the S_score.

    Returns
    -------
    dict with keys: w0, residuals, S_score, eta_activity, eta_planeta,
    n_combos_evaluated.
    """
    combos = []
    for w0 in w0_grid:
        sig = filter_band_once(t, rv, w0, band, permin=permin, permax=permax)
        residuals = rv - sig
        s_score, eta_act, eta_plan = compute_empirical_gls_score(
            t, residuals, rv_err, p_rot, p_rot_half, p_planeta, permax=permax
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


# ---------------------------------------------------------------------------
# MODE 2: DOUBLE FILTER (two w0 ranges, two bands, order 1 and/or 2)
# ---------------------------------------------------------------------------
def double_filter_grid(t, rv, rv_err, order, w0_grid_1, w0_grid_2, band_1, band_2,
                        permin=MIN_PERIOD_ANALYSIS_DEFAULT,
                        permax=MAX_PERIOD_ANALYSIS_DEFAULT):
    """
    "In-memory" version of double filtering: for each w0_1, filter band_1
    (one wavepal call, cached), and for each w0_2, filter band_2 on top of
    the previous residual. Nothing is written to disk.

    order=1 -> band_1 = Prot band,     band_2 = Prot/2 band
    order=2 -> band_1 = Prot/2 band,   band_2 = Prot band

    Returns a list of dicts (w0_1, w0_2, order, residuals), WITHOUT an
    S_score yet (that is computed separately, see
    `run_double_filter_sweep`). Identical to double_filter_grid_in_memory()
    in analyze_filter_sweep_v3.py.
    """
    out = []
    for w0_1 in w0_grid_1:
        sig_1 = filter_band_once(t, rv, w0_1, band_1, permin=permin, permax=permax)
        filtered_1 = rv - sig_1

        for w0_2 in w0_grid_2:
            sig_2 = filter_band_once(t, filtered_1, w0_2, band_2, permin=permin, permax=permax)
            residuals = filtered_1 - sig_2

            out.append({
                "w0_1": float(w0_1), "w0_2": float(w0_2), "order": order,
                "residuals": residuals,
            })
    return out


def run_double_filter_sweep(t, rv, rv_err, prot_value, prot_half_value, planet_period,
                             order1=None, order2=None,
                             permin=MIN_PERIOD_ANALYSIS_DEFAULT,
                             permax=MAX_PERIOD_ANALYSIS_DEFAULT, verbose=True):
    """
    DOUBLE filtering sweep for a SINGLE target, trying order 1 (Prot
    first), order 2 (Prot/2 first), or both, and keeping the overall
    combination with the lowest S_score. This is the equivalent of
    `process_system()` in analyze_filter_sweep_v3.py, but for a single
    file (no manifest, no multiprocessing across systems).

    Parameters
    ----------
    t, rv, rv_err : np.ndarray
        RV time series.
    prot_value, prot_half_value, planet_period : float
        Rotation period, its half, and the (candidate) planet period, in
        days.
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

    Returns
    -------
    dict with the winning combination: order, w0_1, w0_2, residuals,
    S_score, eta_activity, eta_planeta, n_combos_evaluated.
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
            t, combo["residuals"], rv_err, prot_value, prot_half_value, planet_period,
            permax=permax,
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
