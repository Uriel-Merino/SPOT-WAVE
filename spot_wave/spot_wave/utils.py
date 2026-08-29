# -*- coding: utf-8 -*-
"""
utils.py
Funciones de soporte de spot_wave: localizacion de CARMCMC, carga de ficheros
de RV, chequeo de viabilidad de la CWT y wrapper de wavepal.timefreq_analysis.

Todo lo que aqui se usa proviene de tu notebook SPOT_WAVE.ipynb (celdas 1,
1.1 y 1.2), generalizado para no depender de rutas ni de un fichero concreto.
"""

import os
import sys

import numpy as np
import wavepal as wv

# ---------------------------------------------------------------------------
# CARMCMC
# ---------------------------------------------------------------------------
# wavepal usa CARMCMC (con extensiones C++) para modelar el ruido correlado
# y estimar los niveles de confianza del scalogram. La ruta al build de
# carma_pack es especifica de cada maquina, asi que en vez de dejarla
# hardcodeada (como en el notebook) se resuelve, en este orden:
#   1) argumento explicito `path`
#   2) variable de entorno SPOT_WAVE_CARMCMC_PATH
#   3) no se toca sys.path (asumimos que carmcmc ya es importable)
_CARMCMC_LOADED = False


def setup_carmcmc(path=None, verbose=True):
    """
    Añade al sys.path la ruta al build de carma_pack (CARMCMC) e importa el
    modulo, igual que la celda 1 de SPOT_WAVE.ipynb.

    Parameters
    ----------
    path : str, opcional
        Ruta a .../carmcmc/carma_pack/src . Si no se indica, se usa la
        variable de entorno SPOT_WAVE_CARMCMC_PATH si existe.
    verbose : bool
        Si True, imprime el mensaje de confirmacion (como en el notebook).

    Returns
    -------
    module carmcmc, o None si no se pudo cargar.
    """
    global _CARMCMC_LOADED

    rute_carmcmc = path or os.environ.get("SPOT_WAVE_CARMCMC_PATH")
    if rute_carmcmc and rute_carmcmc not in sys.path:
        sys.path.insert(0, rute_carmcmc)

    try:
        import carmcmc  # noqa: F401
    except ImportError as e:
        if verbose:
            print("[spot_wave] Aviso: no se pudo importar carmcmc ({0}). "
                  "Configura SPOT_WAVE_CARMCMC_PATH o pasa `path=` a "
                  "setup_carmcmc().".format(e))
        return None

    _CARMCMC_LOADED = True
    if verbose:
        print("CARMCMC has been successfully loaded with all its C++ libraries!")
    return carmcmc


# ---------------------------------------------------------------------------
# CARGA DE DATOS DE RV
# ---------------------------------------------------------------------------
def load_rv_file(filename, subtract_instrument_means=True, avoid_time_collisions=True,
                  eps=1.1e-5):
    """
    Carga un fichero de RV con el mismo criterio que la celda 1.1 del
    notebook: detecta si hay cabecera, detecta el nº de columnas, y si hay
    columna de instrumento (4a columna) resta la media de RV por instrumento.

    Formato esperado por columnas: time, RV, RV_err[, instrument]

    Parameters
    ----------
    filename : str
        Ruta al fichero .dat de RVs.
    subtract_instrument_means : bool
        Si hay varios instrumentos, resta la media de cada uno (offset).
    avoid_time_collisions : bool
        Si True, desplaza en `eps` los tiempos duplicados/no crecientes
        (necesario para wavepal, igual que en analyze_filter_sweep_v3.py).
    eps : float
        Desplazamiento temporal usado para evitar colisiones.

    Returns
    -------
    t, rv, rv_err : np.ndarray
    instruments : np.ndarray de str (con "single_instrument" si no habia)
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


def parse_k_true(filename, header_regex=None):
    """
    Extrae la semi-amplitud K "verdadera" (inyectada) de la primera linea
    del fichero, si el fichero es sintetico y la lleva en la cabecera
    (formato "K=...", "K_1: ...", etc.). Devuelve np.nan si no se encuentra.
    Util para validar la recuperacion de K de un unico target sintetico.
    """
    import re
    regex = header_regex or re.compile(r"K\w*\s*[=:]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
    with open(filename, "r") as f:
        header = f.readline().strip()
    m = regex.search(header)
    if m:
        return float(m.group(1)), header
    return np.nan, header


# ---------------------------------------------------------------------------
# VIABILIDAD DE LA CWT
# ---------------------------------------------------------------------------
def test_cwt_feasibility(wave_obj, w0=8.5, verbose=True):
    """
    Chequeo rapido de si el dataset aguanta una CWT con el w0 dado.
    Identico a la funcion de la celda 1.2 del notebook.

    Nota: son aproximaciones teoricas (Torrence & Compo / wavepal). Segun el
    nivel de ruido y la distribucion de huecos en el tiempo, puedes ver
    señales ligeramente fuera de estos limites. Usalo como orientacion.
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
            print("Heads up: Less than 50 points. The scalogram is going to be mostly artifacts.")
        print(f"Recommended w0 range: [{w0_min}, {w0_max:.2f}]")
        if w0 < w0_min or w0 > w0_max:
            print(f"Warning: Your chosen w0 ({w0}) is outside the safe zone. Things might get weird.\n")
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


# ---------------------------------------------------------------------------
# WAVEPAL WRAPPER
# ---------------------------------------------------------------------------
def wavepal_analyze(t, y, w0, permin=1.0, permax=200.0, deltaj=0.01,
                     percentile=(95., 99.9), t_units="days", mydata_units="m/s",
                     trend_degree=-1, verbose=False):
    """
    Crea y ejecuta un objeto wavepal.Wavepal para (t, y) con un w0 dado.
    Equivalente a wavepal_analyze() de analyze_filter_sweep_v3.py, pero
    generalizado (percentiles y unidades configurables).
    """
    wave = wv.Wavepal(t, y, "BJD", "STEP", t_units=t_units, mydata_units=mydata_units)
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
