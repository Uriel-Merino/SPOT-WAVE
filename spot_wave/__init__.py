# -*- coding: utf-8 -*-
"""
spot_wave
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
