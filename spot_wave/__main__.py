# -*- coding: utf-8 -*-
"""
__main__.py
spot_wave CLI to run the pipeline (single or double filter) on a SINGLE
target: `python -m spot_wave ...`

Examples
--------
Simple filter (one band, one w0 range):

    python -m spot_wave single \
        --rv-file my_target.dat --out-dir results/ \
        --p-rot 55.0 --p-planeta 19.25 \
        --band-lo 54.5 --band-hi 55.5 \
        --w0-min 5.5 --w0-max 20.0 --w0-step 0.5

Double filter, order 1 (Prot first) + order 2 (Prot/2 first):

    python -m spot_wave double \
        --rv-file my_target.dat --out-dir results/ \
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
    p.add_argument("--rv-file", required=True, help="RV .dat file (time rv rv_err [instrument])")
    p.add_argument("--out-dir", required=True, help="Output folder")
    p.add_argument("--p-rot", type=float, required=True, help="Rotation period (days)")
    p.add_argument("--p-planeta", type=float, required=True, help="Candidate planet period (days)")
    p.add_argument("--permin", type=float, default=1.0)
    p.add_argument("--permax", type=float, default=200.0)
    p.add_argument("--carmcmc-path", default=None, help="Path to carma_pack/src (optional)")
    p.add_argument("--quiet", action="store_true")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="spot_wave", description="Single-target wavelet filtering pipeline")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_single = sub.add_parser("single", help="Simple filter: one band, one w0 range")
    _common_args(p_single)
    p_single.add_argument("--band-lo", type=float, required=True)
    p_single.add_argument("--band-hi", type=float, required=True)
    p_single.add_argument("--w0-min", type=float, default=5.5)
    p_single.add_argument("--w0-max", type=float, required=True)
    p_single.add_argument("--w0-step", type=float, default=0.5)

    p_double = sub.add_parser("double", help="Double filter: two bands (Prot and Prot/2), order 1 and/or 2")
    _common_args(p_double)
    p_double.add_argument("--w0-1-min", type=float, required=True)
    p_double.add_argument("--w0-1-max", type=float, required=True)
    p_double.add_argument("--w0-1-step", type=float, default=0.5)
    p_double.add_argument("--w0-2-min", type=float, required=True)
    p_double.add_argument("--w0-2-max", type=float, required=True)
    p_double.add_argument("--w0-2-step", type=float, default=0.5)
    p_double.add_argument("--hw-prot", type=float, default=0.5, help="Prot band half-width (days)")
    p_double.add_argument("--hw-half", type=float, default=0.5, help="Prot/2 band half-width (days)")
    p_double.add_argument("--orders", default="1,2", help="Orders to try: '1', '2', or '1,2'")

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

    print(f"\nWinner -> S_score={best['S_score']:.4g} "
          f"(eta_activity={best['eta_activity']:.4g}, eta_planeta={best['eta_planeta']:.4g})")
    print(f"Residuals saved to: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
