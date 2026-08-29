# -*- coding: utf-8 -*-
"""
conan_fit.py
CONAN wrapper to fit the winning residuals (single or double filter) of a
SINGLE target, and extract their metrics/posteriors.

Gathers, with no logic changes, run_conan_fit / extract_conan_metrics /
extract_k_posterior from analyze_filter_sweep_v3.py and
analyze_filter_sweep_syst.py.
"""

import os
import numpy as np


def run_conan_fit(filtered_data_files, data_path, output_folder, planet_pars,
                   m_star, gamma_prior, n_planets=1, n_live=500, n_cpus_conan=1):
    """
    Launches a CONAN fit on the given residual file(s).

    Parameters
    ----------
    filtered_data_files : list[str]
        File names (relative to `data_path`) with the winning residual.
    data_path : str
        Folder containing those files.
    output_folder : str
        Fit output folder (created if it doesn't exist).
    planet_pars : dict
        kwargs for rv_obj.planet_parameters(**planet_pars), e.g.:
            dict(T_0=[(t0, sigma_t0)], Period=[(P, sigma_P)],
                 Eccentricity=[0], omega=[90],
                 K=[(0.0, K_prior_max/2., K_prior_max)])
    m_star : tuple(float, float)
        (stellar_mass, error) in M_sun.
    gamma_prior : list
        RV offset prior, e.g. [(0, 30)].
    n_planets, n_live, n_cpus_conan : int
        CONAN.fit_setup / sampling parameters.

    Returns
    -------
    str : path to the generated posteriors.dat file.
    """
    import CONAN
    os.makedirs(output_folder, exist_ok=True)

    rv_obj = CONAN.load_rvs(file_list=filtered_data_files, data_filepath=data_path,
                             rv_unit='m/s', nplanet=n_planets, lc_obj=None)
    rv_obj.rescale_data_columns()
    rv_obj.planet_parameters(**planet_pars)
    rv_obj.rv_baseline(gamma=gamma_prior)

    fit_obj = CONAN.fit_setup(M_st=m_star, par_input="Mrho",
                               apply_RVjitter="y", RVjitter_lims=[0, 30])
    fit_obj.sampling(sampler="dynesty", n_cpus=n_cpus_conan, n_live=n_live, verbose=False)

    CONAN.run_fit(lc_obj=None, rv_obj=rv_obj, fit_obj=fit_obj, out_folder=output_folder)
    return os.path.join(output_folder, "posteriors.dat")


def extract_conan_metrics(output_folder, n_data_points, extra_params=2):
    """
    Reads evidence.dat and AIC_BIC.dat from CONAN's output and computes
    logZ, CONAN's "raw" AIC/BIC, and corrected AIC/BIC that add
    `extra_params` extra parameters (e.g. from the wavelet filtering) to
    CONAN's parameter count.
    """
    evidence_file = os.path.join(output_folder, "evidence.dat")
    aic_bic_file = os.path.join(output_folder, "AIC_BIC.dat")

    logz_val = None
    if os.path.exists(evidence_file):
        with open(evidence_file, "r") as f:
            for line in f.readlines():
                if line.startswith("logz:"):
                    logz_val = float(line.split()[1])
                    break

    aic_conan = bic_conan = None
    if os.path.exists(aic_bic_file):
        with open(aic_bic_file, "r") as f:
            last_line = f.readlines()[-1].strip().split()
            if len(last_line) >= 2:
                aic_conan = float(last_line[0])
                bic_conan = float(last_line[1])

    aic_total = bic_total = None
    if aic_conan is not None and bic_conan is not None:
        aic_total = aic_conan + 2 * extra_params
        bic_total = bic_conan + extra_params * np.log(n_data_points)

    return {"logZ": logz_val, "AIC_conan": aic_conan, "BIC_conan": bic_conan,
            "AIC_corrected": aic_total, "BIC_corrected": bic_total}


def extract_k_posterior(output_folder, planet_index=1):
    """
    Reads results_med.dat and flexibly extracts K (or K_<planet_index> if
    there are several planets): median plus 1-sigma and 3-sigma
    intervals.

    Returns
    -------
    tuple (K_med, K_lo_1sigma, K_hi_1sigma, K_lo_3sigma, K_hi_3sigma)
    All np.nan if the row is not found or the file does not exist.
    """
    res_path = os.path.join(output_folder, "results_med.dat")
    if not os.path.exists(res_path):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    targets = {"K", f"K_{planet_index}"}
    with open(res_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()

            if parts[0] not in targets:
                continue
            if len(parts) < 6:
                return np.nan, np.nan, np.nan, np.nan, np.nan

            med = float(parts[1])
            m1, p1 = float(parts[2]), float(parts[3])
            m3, p3 = float(parts[4]), float(parts[5])

            lo1, hi1 = med + m1, med + p1
            lo3, hi3 = med + m3, med + p3
            return float(med), float(lo1), float(hi1), float(lo3), float(hi3)

    return np.nan, np.nan, np.nan, np.nan, np.nan


def evaluate_k_recovery(k_med, k_lo1, k_hi1, k_lo3, k_hi3, k_true):
    """
    Compares the K recovered by CONAN with the injected K (if known, e.g.
    via utils.parse_k_true on synthetic data). Returns a dict ready to
    drop into a results summary.
    """
    if np.isnan(k_true):
        return {"recovered_within_1sigma": None, "recovered_within_3sigma": None,
                "K_residual": np.nan}
    return {
        "recovered_within_1sigma": bool(k_lo1 <= k_true <= k_hi1),
        "recovered_within_3sigma": bool(k_lo3 <= k_true <= k_hi3),
        "K_residual": k_med - k_true,
    }
