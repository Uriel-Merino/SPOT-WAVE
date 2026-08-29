# -*- coding: utf-8 -*-
"""
filter.py
Filtrado wavelet para UN UNICO target (no sistematico / no multi-fichero).

Reimplementa, para un solo sistema, la logica de:
  - analyze_filter_sweep_v3.py      -> filtrado DOBLE (dos bandas, dos w0),
                                        barriendo order1 (Prot primero) y/o
                                        order2 (Prot/2 primero).
  - analyze_filter_sweep_syst.py    -> post-procesado / metricas.

y añade el modo SIMPLE (un unico rango de w0, una unica banda), que es el
que usabas antes de pasar al filtrado doble.

En ambos modos, el score que decide el "ganador" es el mismo S_score
empirico (eta_actividad / eta_planeta) via GLS, calculado con
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
            "No se pudo importar `Gls`. Instala/añade al PYTHONPATH tu "
            "modulo `gls.py` (periodograma Zechmeister & Kurster), igual "
            "que en analyze_filter_sweep_v3.py."
        )


def make_w0_grid(w0_min, w0_max, w0_step):
    """Atajo para np.arange(w0_min, w0_max+step, step), igual que `_grid()`."""
    return np.arange(w0_min, w0_max + w0_step, w0_step)


# ---------------------------------------------------------------------------
# SCORE EMPIRICO (GLS)
# ---------------------------------------------------------------------------
def compute_empirical_gls_score(time_arr, residuals, rv_err, p_rot, p_rot_half,
                                 p_planeta, permax=MAX_PERIOD_ANALYSIS_DEFAULT):
    """
    Identico a compute_empirical_gls_score() de analyze_filter_sweep_v3.py.

    S_score = eta_actividad / eta_planeta, donde:
      eta_actividad = potencia_GLS(Prot)/FAP_99 + potencia_GLS(Prot/2)/FAP_99
      eta_planeta   = potencia_GLS(P_planeta)/FAP_99

    Un S_score bajo => se ha suprimido bien la actividad sin cargarse la
    señal del planeta => mejor combinacion de filtrado.
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
# FILTRADO DE UNA BANDA
# ---------------------------------------------------------------------------
def filter_band_once(t, y, w0, band, permin=MIN_PERIOD_ANALYSIS_DEFAULT,
                      permax=MAX_PERIOD_ANALYSIS_DEFAULT):
    """
    Aplica wavepal con un w0 dado y filtra UNA banda de periodos.
    Devuelve la señal filtrada (centrada en 0), igual que
    `_filter_band_once()` de analyze_filter_sweep_v3.py.
    """
    wave = wavepal_analyze(t, y, w0, permin=permin, permax=permax)
    wave.timefreq_band_filtering([band])
    if wave.timefreq_band_filtered_signal is None:
        return np.zeros_like(y)
    sig = wave.timefreq_band_filtered_signal[:, 0]
    sig = sig - np.mean(sig)
    return sig


# ---------------------------------------------------------------------------
# MODO 1: FILTRO SIMPLE (un unico rango de w0, una unica banda)
# ---------------------------------------------------------------------------
def single_filter_sweep(t, rv, rv_err, w0_grid, band, p_rot, p_rot_half,
                         p_planeta, permin=MIN_PERIOD_ANALYSIS_DEFAULT,
                         permax=MAX_PERIOD_ANALYSIS_DEFAULT, verbose=True):
    """
    Barrido de un UNICO rango de w0 sobre UNA UNICA banda (p.ej. la banda de
    Prot), para un solo target. Para cada w0 se filtra la banda, se calcula
    el residuo y su S_score, y se devuelve la combinacion ganadora (S_score
    minimo).

    Es el analogo "de una sola pasada" al filtrado doble: usalo cuando con
    filtrar una unica banda (normalmente la de Prot) ya basta para separar
    la actividad de la señal planetaria.

    Parameters
    ----------
    t, rv, rv_err : np.ndarray
        Serie temporal de RV (ya cargada, p.ej. con utils.load_rv_file).
    w0_grid : array-like
        Valores de w0 a barrer (usa make_w0_grid(w0_min, w0_max, w0_step)).
    band : tuple(float, float)
        Banda de periodos a filtrar, en dias, p.ej. (Prot-0.5, Prot+0.5).
    p_rot, p_rot_half, p_planeta : float
        Periodos (dias) de rotacion, su mitad, y del planeta, usados para
        el S_score.

    Returns
    -------
    dict con las claves: w0, residuals, S_score, eta_activity, eta_planeta,
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
        print(f"[single_filter_sweep] MEJOR w0={best['w0']:.3f} -> S_score={best['S_score']:.4g}")
    return best


# ---------------------------------------------------------------------------
# MODO 2: FILTRO DOBLE (dos rangos de w0, dos bandas, orden 1 y/o 2)
# ---------------------------------------------------------------------------
def double_filter_grid(t, rv, rv_err, order, w0_grid_1, w0_grid_2, band_1, band_2,
                        permin=MIN_PERIOD_ANALYSIS_DEFAULT,
                        permax=MAX_PERIOD_ANALYSIS_DEFAULT):
    """
    Version "en memoria" del filtrado doble: para cada w0_1 filtra la
    banda_1 (una llamada a wavepal, cacheada), y para cada w0_2 filtra la
    banda_2 sobre el residuo anterior. No escribe nada a disco.

    order=1 -> banda_1 = banda Prot,     banda_2 = banda Prot/2
    order=2 -> banda_1 = banda Prot/2,   banda_2 = banda Prot

    Devuelve una lista de dicts (w0_1, w0_2, order, residuals), SIN S_score
    todavia (se calcula aparte, ver `run_double_filter_sweep`).
    Identico a double_filter_grid_in_memory() de analyze_filter_sweep_v3.py.
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
    Barrido de filtrado DOBLE para UN UNICO target, probando el orden 1
    (Prot primero), el orden 2 (Prot/2 primero), o ambos, y quedandose con
    la combinacion global de minimo S_score. Es el equivalente de
    `process_system()` en analyze_filter_sweep_v3.py, pero para un solo
    fichero (sin manifest, sin multiprocessing entre sistemas).

    Parameters
    ----------
    t, rv, rv_err : np.ndarray
        Serie temporal de RV.
    prot_value, prot_half_value, planet_period : float
        Periodo de rotacion, su mitad, y periodo (candidato) del planeta,
        en dias.
    order1, order2 : dict o None
        Cada uno con las claves:
            w0_grid_1, w0_grid_2 : array-like (via make_w0_grid)
            hw_1, hw_2           : semi-anchura de banda (dias) para cada filtro
        order1: w0_grid_1/hw_1 filtran la banda de Prot (primero),
                w0_grid_2/hw_2 filtran la banda de Prot/2 (segundo).
        order2: w0_grid_1/hw_1 filtran la banda de Prot/2 (primero),
                w0_grid_2/hw_2 filtran la banda de Prot (segundo).
        Pasa None para no ejecutar ese orden. Al menos uno de los dos debe
        indicarse.

    Returns
    -------
    dict con la combinacion ganadora: order, w0_1, w0_2, residuals, S_score,
    eta_activity, eta_planeta, n_combos_evaluated.
    """
    if order1 is None and order2 is None:
        raise ValueError("Debes indicar al menos order1 o order2.")

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
        print(f"[double_filter_sweep] MEJOR order={best['order']} "
              f"w0_1={best['w0_1']:.3f} w0_2={best['w0_2']:.3f} "
              f"-> S_score={best['S_score']:.4g}")
    return best


def save_winner_file(t, residuals, rv_err, output_path):
    """Guarda a disco el residuo ganador (formato compatible con CONAN.load_rvs)."""
    import pandas as pd
    pd.DataFrame({"time": t, "residuals": residuals, "rv_err": rv_err}).to_csv(
        output_path, sep=" ", index=False, header=False
    )
    return output_path
