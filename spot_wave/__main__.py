# -*- coding: utf-8 -*-
"""
__main__.py
CLI de spot_wave para correr el pipeline (filtro simple o doble) sobre UN
UNICO target: `python -m spot_wave ...`

Ejemplos
--------
Filtro simple (una banda, un rango de w0):

    python -m spot_wave single \
        --rv-file mi_target.dat --out-dir resultados/ \
        --p-rot 55.0 --p-planeta 19.25 \
        --band-lo 54.5 --band-hi 55.5 \
        --w0-min 5.5 --w0-max 20.0 --w0-step 0.5

Filtro doble, orden 1 (Prot primero) + orden 2 (Prot/2 primero):

    python -m spot_wave double \
        --rv-file mi_target.dat --out-dir resultados/ \
        --p-rot 55.0 --p-planeta 19.25 \
        --w0-1-min 5.5 --w0-1-max 7.0 --w0-1-step 0.5 \
        --w0-2-min 14.5 --w0-2-max 20.0 --w0-2-step 0.25 \
        --hw-prot 0.5 --hw-half 0.5 --orders 1,2
"""

import argparse
import os
import sys

from . import (
    load_rv_file, make_w0_grid, single_filter_sweep, run_double_filter_sweep,
    save_winner_file, setup_carmcmc,
)


def _common_args(p):
    p.add_argument("--rv-file", required=True, help="Fichero .dat de RVs (time rv rv_err [instrument])")
    p.add_argument("--out-dir", required=True, help="Carpeta de salida")
    p.add_argument("--p-rot", type=float, required=True, help="Periodo de rotacion (dias)")
    p.add_argument("--p-planeta", type=float, required=True, help="Periodo candidato del planeta (dias)")
    p.add_argument("--permin", type=float, default=1.0)
    p.add_argument("--permax", type=float, default=200.0)
    p.add_argument("--carmcmc-path", default=None, help="Ruta a carma_pack/src (opcional)")
    p.add_argument("--quiet", action="store_true")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="spot_wave", description="Pipeline de filtrado wavelet de un unico target")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_single = sub.add_parser("single", help="Filtro simple: una banda, un rango de w0")
    _common_args(p_single)
    p_single.add_argument("--band-lo", type=float, required=True)
    p_single.add_argument("--band-hi", type=float, required=True)
    p_single.add_argument("--w0-min", type=float, default=5.5)
    p_single.add_argument("--w0-max", type=float, required=True)
    p_single.add_argument("--w0-step", type=float, default=0.5)

    p_double = sub.add_parser("double", help="Filtro doble: dos bandas (Prot y Prot/2), orden 1 y/o 2")
    _common_args(p_double)
    p_double.add_argument("--w0-1-min", type=float, required=True)
    p_double.add_argument("--w0-1-max", type=float, required=True)
    p_double.add_argument("--w0-1-step", type=float, default=0.5)
    p_double.add_argument("--w0-2-min", type=float, required=True)
    p_double.add_argument("--w0-2-max", type=float, required=True)
    p_double.add_argument("--w0-2-step", type=float, default=0.5)
    p_double.add_argument("--hw-prot", type=float, default=0.5, help="Semi-anchura banda Prot (dias)")
    p_double.add_argument("--hw-half", type=float, default=0.5, help="Semi-anchura banda Prot/2 (dias)")
    p_double.add_argument("--orders", default="1,2", help="Ordenes a probar: '1', '2' o '1,2'")

    args = parser.parse_args(argv)
    verbose = not args.quiet
    os.makedirs(args.out_dir, exist_ok=True)

    setup_carmcmc(path=args.carmcmc_path, verbose=verbose)
    t, rv, rv_err, _instruments = load_rv_file(args.rv_file)
    p_rot_half = args.p_rot / 2.0

    if args.mode == "single":
        w0_grid = make_w0_grid(args.w0_min, args.w0_max, args.w0_step)
        best = single_filter_sweep(
            t, rv, rv_err, w0_grid, band=(args.band_lo, args.band_hi),
            p_rot=args.p_rot, p_rot_half=p_rot_half, p_planeta=args.p_planeta,
            permin=args.permin, permax=args.permax, verbose=verbose,
        )
        out_file = os.path.join(
            args.out_dir, f"single_w0_{best['w0']:.3f}.dat"
        )
        save_winner_file(t, best["residuals"], rv_err, out_file)

    else:  # double
        orders = {int(o) for o in args.orders.split(",")}
        order1 = None
        order2 = None
        if 1 in orders:
            order1 = dict(
                w0_grid_1=make_w0_grid(args.w0_1_min, args.w0_1_max, args.w0_1_step),
                w0_grid_2=make_w0_grid(args.w0_2_min, args.w0_2_max, args.w0_2_step),
                hw_1=args.hw_prot, hw_2=args.hw_half,
            )
        if 2 in orders:
            order2 = dict(
                w0_grid_1=make_w0_grid(args.w0_1_min, args.w0_1_max, args.w0_1_step),
                w0_grid_2=make_w0_grid(args.w0_2_min, args.w0_2_max, args.w0_2_step),
                hw_1=args.hw_half, hw_2=args.hw_prot,
            )
        best = run_double_filter_sweep(
            t, rv, rv_err, prot_value=args.p_rot, prot_half_value=p_rot_half,
            planet_period=args.p_planeta, order1=order1, order2=order2,
            permin=args.permin, permax=args.permax, verbose=verbose,
        )
        out_file = os.path.join(
            args.out_dir,
            f"double_w0_{best['w0_1']:.3f}_{best['w0_2']:.3f}_order{best['order']}.dat",
        )
        save_winner_file(t, best["residuals"], rv_err, out_file)

    print(f"\nGanador -> S_score={best['S_score']:.4g} "
          f"(eta_activity={best['eta_activity']:.4g}, eta_planeta={best['eta_planeta']:.4g})")
    print(f"Residuos guardados en: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
