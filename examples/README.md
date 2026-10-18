# Examples

| File | Purpose |
|---|---|
| `run_stone_skipping_local.py` | Generic driver for one black hole orbiting in a soliton (the paper recipe), small defaults for a laptop. |
| `pbs/stone_skipping.pbs` | PBS Pro job template that runs the driver. |
| `slurm/stone_skipping.sbatch` | Slurm job template that runs the driver. |

## Local run

From the repository root (any working directory works since 30.50):

```bash
python -m pip install -e .            # or: pip install -r requirements-full.txt
python examples/run_stone_skipping_local.py                      # 64^3, 2.5 kpc, 100 Myr, 10 snapshots
python examples/run_stone_skipping_local.py --resol 96 --duration 300 --out runs --name test96
python examples/run_stone_skipping_local.py --help
```

Outputs go to `examples/output/<run>/` unless `--out` is given. Every run folder has
`config.uldm`, `reproducibility.uldm` (settings, thread count, FFT backend, library versions,
sha256 of the core file), `Outputs/<flag>/<prefix>_#NNN.npy` and the snapshot times
`Outputs/snapshot_times_Myr.npy` (index 0 is the initial state).

The recipe is the one of the paper drivers:

1. `PYUL_AXION_MASS_EV` is set before `import PyUltraLight2` (it is read once, at import);
2. `dx = MeshSpacing(resol, length)`, Plummer radius `rP = dx/2`, `a = GenPlummer(rP)`;
3. `DefaultSolitonOrbit(..., mS, R)` gives the circular speed at separation `R = 0.14 kpc`;
   positions and velocities are split barycentrically between the soliton (`mS`) and the
   black hole (`mP = 0.02 mS` by default);
4. `GenerateConfig(...)`, then `evolve(..., CellCenteredGrid=True)`.

Useful environment variables (all optional):

| Variable | Effect |
|---|---|
| `PYUL_AXION_MASS_EV` | axion mass in eV (default 1e-22; the driver sets `--axion-mass`, default 1e-21) |
| `PYUL_NUM_THREADS` | FFT/numexpr thread count; otherwise `OMP_NUM_THREADS`, `SLURM_CPUS_PER_TASK`, `NCPUS` (PBS), CPU affinity, `cpu_count` |
| `PYUL_FFT_BACKEND` | `pyfftw` (default when installed) or `scipy` |
| `PYUL_PROGRESS` | `bar`, `lines` (default when stdout is not a terminal) or `quiet` |
| `PYUL_INTERACTIVE` | `0` never prompt, `1` always allow prompts |
| `PYUL_DATA_DIR` | folder that contains `Soliton Profile Files/` (for non-editable installs) |
| `PYUL_CACHE_DIR` | where IsoP Green functions and generated profiles are cached |

## Batch jobs

The templates use `set -euo pipefail`, write one log per job under `logs/` with the job id and
a timestamp, never pipe through `tee` (so the job's exit status is the driver's), pin no host, and
export one explicit thread count (`NCPUS` or `SLURM_CPUS_PER_TASK`) to `PYUL_NUM_THREADS`,
`OMP_NUM_THREADS` and `NUMEXPR_MAX_THREADS`. Pass driver options with `RUN_ARGS`:

```bash
qsub   -v PYUL_REPO=$HOME/PyUL_SK,RUN_ARGS="--resol 192 --duration 3000 --name f002_192" examples/pbs/stone_skipping.pbs
mkdir -p logs
sbatch --export=ALL,PYUL_REPO=$HOME/PyUL_SK,RUN_ARGS="--resol 192 --duration 3000" examples/slurm/stone_skipping.sbatch
```

In batch jobs PyUL never calls `input()`: an existing run folder or `Outputs/` folder raises
`FileExistsError` unless `--overwrite` (driver), `Overwrite=True` (`GenerateConfig`) or
`OverwriteOutputs=True` (`evolve`) is given.
