#!/usr/bin/env python3
"""Laptop-friendly Stone Skipping run: one black hole orbiting inside a soliton.

This follows the recipe of the paper drivers (Zhang et al., "Stone Skipping
Black Holes in Ultralight Dark Matter Solitons"):

1. set PYUL_AXION_MASS_EV *before* importing PyUltraLight2 (read once at import);
2. grid spacing from MeshSpacing, Plummer radius rP = dx/2;
3. circular-orbit speed at radius R from DefaultSolitonOrbit for the soliton
   mass mS, and a barycentric split of positions and velocities between the
   soliton (mS) and the black hole (mP = ratio * mS);
4. GenerateConfig, then evolve with CellCenteredGrid=True.

Defaults are small (64^3, 100 Myr, 10 snapshots; about a minute on a laptop).
The paper production runs used --resol 192/256 --duration 3000 on 16-56 cores;
see examples/pbs and examples/slurm for batch templates.

Examples:
    python examples/run_stone_skipping_local.py
    python examples/run_stone_skipping_local.py --resol 96 --duration 300 --out runs
    python examples/run_stone_skipping_local.py --center-calc --save-options "Energy DF ULDCOM NBody 3WfnRS"
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--axion-mass", type=float, default=1e-21, help="axion mass in eV (PYUL_AXION_MASS_EV)")
    p.add_argument("--resol", type=int, default=64, help="grid points per side")
    p.add_argument("--length", type=float, default=2.5, help="box side in kpc")
    p.add_argument("--duration", type=float, default=100.0, help="simulated time in Myr")
    p.add_argument("--save-number", type=int, default=10, help="number of snapshots after the initial one")
    p.add_argument("--soliton-mass", type=float, default=12.0, help="soliton mass in 1e6 M_sun (M_solar_masses)")
    p.add_argument("--mass-ratio", type=float, default=0.02, help="black-hole mass / soliton mass")
    p.add_argument("--radius", type=float, default=0.14, help="initial BH-soliton separation in kpc")
    p.add_argument("--ns", type=int, default=32, help="RK steps per ULDM step (0, 1 or a multiple of 4)")
    p.add_argument("--save-options", default="Energy DF ULDCOM NBody", help="PyUL save flags")
    p.add_argument("--isop", action="store_true", help="isolated (zero-padded) potential instead of periodic")
    p.add_argument("--center-calc", action="store_true", help="track the ULDM COM and write the *RS resampled outputs")
    p.add_argument("--resolR", type=int, default=None, help="resampled cube resolution (default: --resol)")
    p.add_argument("--length-ratio", type=float, default=1.0, help="resampled cube side / box side")
    p.add_argument("--legacy-resample-box", action="store_true",
                   help="30.40 resample cube side (L-dx)*Length_Ratio instead of L*Length_Ratio")
    p.add_argument("--out", default=os.path.join(HERE, "output"), help="save_path (parent folder of runs)")
    p.add_argument("--name", default="", help="run folder name (default: timestamp@resol)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing run folder of the same name")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # 1. The axion mass must be set before the import.
    os.environ["PYUL_AXION_MASS_EV"] = repr(args.axion_mass)
    sys.path.insert(0, REPO)
    import numpy as np
    import PyUltraLight2 as PyUL

    length, length_units = args.length, 'kpc'
    duration, duration_units = args.duration, 'Myr'
    s_mass_unit, s_position_unit, s_velocity_unit = 'M_solar_masses', 'kpc', 'km/s'

    # 2. Grid spacing and Plummer smoothing rP = dx/2.
    gridspace = PyUL.MeshSpacing(args.resol, length, length_units, silent=False)
    rP = gridspace / 2
    a = PyUL.GenPlummer(rP, length_units)

    # 3. Circular speed at the separation and barycentric split.
    mS = args.soliton_mass
    mP = args.mass_ratio * mS
    R = args.radius
    MIn, vRel0 = PyUL.DefaultSolitonOrbit(args.resol, length, length_units, mS, s_mass_unit,
                                          R, s_position_unit, s_velocity_unit)
    M = mS + mP
    particles = [[mP, [R * mS / M, 0, 0], [0, -vRel0 * mS / M, 0]]]
    solitons = [[mS, [-R * mP / M, 0, 0], [0, vRel0 * mP / M, 0], 0.0]]
    PyUL.printU(f"Enclosed soliton mass {MIn:.4g} x1e6 Msun, relative speed {vRel0:.4g} km/s.", 'Example')

    # 4. Configuration and evolution.
    os.makedirs(args.out, exist_ok=True)
    if args.name and os.path.exists(os.path.join(args.out, args.name)) and not args.overwrite:
        raise SystemExit(f"{os.path.join(args.out, args.name)} exists; use --overwrite or another --name.")
    run_folder = PyUL.GenerateConfig(args.ns, length, length_units, args.resol, duration, duration_units, 1,
                                     args.save_number, args.save_options, args.out, 'npy',
                                     s_mass_unit, s_position_unit, s_velocity_unit, solitons, 0.0,
                                     s_mass_unit, s_position_unit, s_velocity_unit, particles, [],
                                     False, 0, '', a, [0, 0, 0], True, args.name, Overwrite=args.overwrite)
    t0 = time.time()
    PyUL.evolve(args.out, run_folder, DumpInit=True, DumpFinal=True, IsoP=args.isop,
                SelfGravity=True, NBodyInterp=True, NBodyGravity=True,
                CenterCalc=args.center_calc, Length_Ratio=args.length_ratio,
                resolR=args.resolR or args.resol, CellCenteredGrid=True,
                LegacyResampleBox=args.legacy_resample_box, OverwriteOutputs=args.overwrite)
    loc = os.path.join(args.out, run_folder)
    times = np.load(os.path.join(loc, "Outputs", "snapshot_times_Myr.npy"))
    PyUL.printU(f"Done in {time.time() - t0:.1f} s: {len(times)} snapshots (0 - {times[-1]:.4g} Myr) in {loc}",
                'Example')
    return loc


if __name__ == "__main__":
    main()
