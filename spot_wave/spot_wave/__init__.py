# -*- coding: utf-8 -*-
"""
spot_wave
=========
Libreria que sistematiza tu pipeline de SPOT_WAVE.ipynb: analisis
tiempo-frecuencia con wavepal/CARMCMC, filtrado wavelet (simple o doble)
de la actividad estelar en RVs, y ajuste orbital con CONAN sobre los
residuos ganadores. Pensada, de momento, para UN UNICO target por
ejecucion (para el barrido sistematico multi-target, ver los scripts
`analyze_filter_sweep_v3.py` / `analyze_filter_sweep_syst.py` del pipeline
original, que usan esta misma libreria por debajo).

Uso tipico
----------
>>> import spot_wave as sw
>>> sw.setup_carmcmc()  # opcional si CARMCMC ya esta en el PYTHONPATH
>>> t, rv, rv_err, instruments = sw.load_rv_file("mi_target.dat")
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
