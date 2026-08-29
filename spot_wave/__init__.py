# -*- coding: utf-8 -*-
"""
spot_wave
=========
Library that systematizes your SPOT_WAVE.ipynb pipeline: time-frequency
analysis with wavepal/CARMCMC, wavelet filtering (single or double) of
stellar activity in RVs, and orbital fitting with CONAN on the winning
residuals. Currently designed for a SINGLE target per run (for the
systematic multi-target sweep, see the original pipeline scripts
`analyze_filter_sweep_v3.py` / `analyze_filter_sweep_syst.py`, which use
this same library under the hood).

Typical usage
-------------
>>> import spot_wave as sw
>>> sw.setup_carmcmc()  # optional if CARMCMC is already on the PYTHONPATH
>>> t, rv, rv_err, instruments = sw.load_rv_file("my_target.dat")
>>> w0_grid = sw.make_w0_grid(5.5, 20.0, 0.5)
>>> best = sw.single_filter_sweep(
...     t, rv, rv_err, w0_grid, band=(54.5, 55.5),
...     p_rot=55.0, p_rot_half=27.5, p_planeta=19.25,
... )
"""

from .__version__ import __version__

from .utils import (
    setup_carmcmc,
    load_rv_file,
    parse_k_true,
    test_cwt_feasibility,
    wavepal_analyze,
)
from .filter import (
    make_w0_grid,
    compute_empirical_gls_score,
    filter_band_once,
    single_filter_sweep,
    double_filter_grid,
    run_double_filter_sweep,
    save_winner_file,
)
from .conan_fit import (
    run_conan_fit,
    extract_conan_metrics,
    extract_k_posterior,
    evaluate_k_recovery,
)
from . import scalogram

__all__ = [
    "__version__",
    "setup_carmcmc", "load_rv_file", "parse_k_true", "test_cwt_feasibility",
    "wavepal_analyze",
    "make_w0_grid", "compute_empirical_gls_score", "filter_band_once",
    "single_filter_sweep", "double_filter_grid", "run_double_filter_sweep",
    "save_winner_file",
    "run_conan_fit", "extract_conan_metrics", "extract_k_posterior",
    "evaluate_k_recovery",
    "scalogram",
]
