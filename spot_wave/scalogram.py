# -*- coding: utf-8 -*-
"""
scalogram.py
Funciones de la seccion "2. SCALOGRAM" de SPOT_WAVE.ipynb: analisis
tiempo-frecuencia con un w0 fijo, ploteo del scalogram, y el barrido en w0
("2.1. Loop w0") que genera un PDF por w0 y opcionalmente re-estima el
rango de periodos detectable a partir del texto que imprime wavepal.
"""

import os
import re
import io
import contextlib

import numpy as np
import matplotlib.pyplot as plt

from .utils import wavepal_analyze

DEFAULT_PERCENTILES = (95., 99., 99.9)

DEFAULT_TIME_STRING = [0., 100., 200., 300., 400., 500., 600., 700., 800., 900., 1000.]
DEFAULT_PERIOD_STRING = [5., 10., 12., 15., 20., 25., 30., 35., 40., 45., 50.,
                          55., 60., 65., 70., 75., 80., 85., 90., 95., 100.]


def analyze_and_plot(wave, w0=7.0, permin=1.0, permax=200.0, deltaj=0.01,
                      percentile=DEFAULT_PERCENTILES, time_string=None,
                      period_string=None, dashed_periods=None, figsize=(24, 12),
                      **plot_kwargs):
    """
    Ejecuta wave.timefreq_analysis con un w0 fijo y devuelve la figura del
    scalogram (equivalente a las celdas 2, y a las celdas 10-12 del
    notebook, con w0=7.0 por defecto).
    """
    time_string = time_string or DEFAULT_TIME_STRING
    period_string = period_string or DEFAULT_PERIOD_STRING
    dashed_periods = dashed_periods or period_string

    wave.timefreq_analysis(
        theta=wave.t, w0=float(w0), permin=permin, permax=permax, deltaj=deltaj,
        percentile=np.asarray(percentile), computes_amplitude=True, smoothing_coeff=0.0,
    )
    fig = wave.plot_scalogram_custom(
        color_cl_anal=['indigo', 'black', 'orchid'],
        fontsize_ticks=20, fontsize_axes=20,
        time_string=time_string, period_string=period_string,
        dashed_periods=dashed_periods, linewidth_cl=4, decimals=2,
        linewidth_gscal=2.0, figsize=figsize, **plot_kwargs,
    )
    return fig


_RANGE_PATTERN = re.compile(
    r"Re-estimated period range:\s*from\s+([\d.eE+-]+)\s+to\s+([\d.eE+-]+)"
)


def w0_loop(wave, output_dir, w0_min=5.5, w0_max=196.0, w0_step=2.0,
            permin=1.0, permax=200.0, deltaj=0.05, percentile=DEFAULT_PERCENTILES,
            time_string=None, period_string=None, dashed_periods=None,
            figsize=(24, 12), save_period_ranges=True, verbose=True):
    """
    Barrido en w0: para cada valor guarda un PDF del scalogram y, si
    `save_period_ranges`, captura el "Re-estimated period range" que
    imprime wavepal para dejar constancia del rango de periodos fiable en
    cada w0 (equivalente a la celda 2.1 del notebook).

    Returns
    -------
    list[(w0, period_min, period_max)]  (period_min/max = np.nan si no se
    pudo parsear la salida de wavepal para ese w0).
    """
    os.makedirs(output_dir, exist_ok=True)
    w0_values = np.arange(w0_min, w0_max + w0_step / 2, w0_step)
    ranges_file = os.path.join(output_dir, "period_ranges.txt")

    results = []
    for w0 in w0_values:
        if save_period_ranges:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                fig = analyze_and_plot(
                    wave, w0=w0, permin=permin, permax=permax, deltaj=deltaj,
                    percentile=percentile, time_string=time_string,
                    period_string=period_string, dashed_periods=dashed_periods,
                    figsize=figsize,
                )
            captured_text = buf.getvalue()
            matches = _RANGE_PATTERN.findall(captured_text)
            if matches:
                per_min, per_max = matches[-1]
                per_min, per_max = float(per_min), float(per_max)
            else:
                per_min, per_max = np.nan, np.nan
                if verbose:
                    print(f"[AVISO] No se encontro 'Re-estimated period range' para w0={w0:.2f}")
            results.append((float(w0), per_min, per_max))
        else:
            fig = analyze_and_plot(
                wave, w0=w0, permin=permin, permax=permax, deltaj=deltaj,
                percentile=percentile, time_string=time_string,
                period_string=period_string, dashed_periods=dashed_periods,
                figsize=figsize,
            )

        output_file = os.path.join(output_dir, f"scalogram_w0_{w0:.2f}.pdf")
        fig.savefig(output_file, bbox_inches="tight")
        plt.close(fig)

        if save_period_ranges:
            with open(ranges_file, "w") as f:
                f.write("w0\tperiod_min\tperiod_max\n")
                for w0_i, pmin_i, pmax_i in results:
                    f.write(f"{w0_i:.4f}\t{pmin_i:.6f}\t{pmax_i:.6f}\n")

    if verbose and save_period_ranges:
        print(f"Rangos de periodos guardados en: {ranges_file}")

    return results
