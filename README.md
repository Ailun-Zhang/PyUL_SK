<center><img src="PyULLogo.png" alt="PyUltraLight logo" width="560" align="center"/></center>

# PyUltraLight 2, SK fork (PyUL_SK)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23454001.svg)](https://doi.org/10.5281/zenodo.23454001)

A pseudo-spectral (split-step Fourier) solver for the Schroedinger-Poisson equations of ultralight dark
matter (ULDM), coupled to N-body point masses (black holes), in a non-expanding periodic or isolated box.
This repository is the fork used for the black-hole simulations of *Stone Skipping Black Holes in
Ultralight Dark Matter Solitons* (A. Zhang, Y. Wang, J. L. Zagorac and R. Easther,
[arXiv:2602.11512](https://arxiv.org/abs/2602.11512)).

The current release is **30.50** (`PyUltraLight2/Current_SK.py`, `S_version = 30.50`). It is a
maintenance release of the **30.40** core used for part of the paper runs. With default options it
reproduces 30.40 bit for bit, except the resampled `*RS` outputs, which need one switch (see
[Versions](#versions-and-reproducing-the-paper-core)). Bugs of older versions and their effect on the
paper's figures are documented in [docs/known_issues_and_errata.md](docs/known_issues_and_errata.md).

## Contents

1. [Lineage, purpose and credits](#lineage-purpose-and-credits)
2. [How to cite](#how-to-cite)
3. [Versions and reproducing the paper core](#versions-and-reproducing-the-paper-core)
4. [Installation](#installation)
5. [Quick start on a laptop](#quick-start-on-a-laptop)
6. [Running on clusters (PBS, Slurm)](#running-on-clusters-pbs-slurm)
7. [Environment variables](#environment-variables)
8. [Units and the axion mass](#units-and-the-axion-mass)
9. [Initial conditions: `GenerateConfig` and `config.uldm`](#initial-conditions-generateconfig-and-configuldm)
10. [`evolve()` parameters](#evolve-parameters)
11. [Outputs and save options](#outputs-and-save-options)
12. [Inline eigenmode decomposition (c_nlm)](#inline-eigenmode-decomposition-c_nlm)
13. [`reproducibility.uldm`](#reproducibilityuldm)
14. [Known issues and conventions](#known-issues-and-conventions)
15. [Tests](#tests)
16. [Repository layout and legacy files](#repository-layout-and-legacy-files)
17. [Licence](#licence)

---

## Lineage, purpose and credits

| Stage | Authors | Reference |
|---|---|---|
| **PyUltraLight** (original pseudo-spectral ULDM solver) | F. Edwards, E. Kendall, S. Hotchkiss and R. Easther | JCAP 10 (2018) 027, [arXiv:1807.04037](https://arxiv.org/abs/1807.04037), doi:10.1088/1475-7516/2018/10/027 |
| **PyUltraLight 2 / PyUL_NBody** (N-body particles, energy diagnostics, I/O management, `.uldm` files) | Yourong F. Wang and collaborators (Auckland Cosmology); [Sifyrena.github.io](https://sifyrena.github.io), FWPhys.com | [github.com/Sifyrena/PyUL_NBody](https://github.com/Sifyrena/PyUL_NBody), Zenodo doi:10.5281/zenodo.6336160 |
| **SK fork** (this repository) | Alan Zhang | [github.com/Ailun-Zhang/PyUL_SK](https://github.com/Ailun-Zhang/PyUL_SK) |

The origin of the code and the upstream copyright notice are recorded in [NOTICE](NOTICE).

Modules adapted by the upstream code and acknowledged in its credits banner: the isolated (zero-padded)
potential and the dispersive sponge condition are adapted from J. L. Zagorac et al.; the embedded-soliton
profiles are adapted from N. Guo et al.

What the SK fork adds on top of upstream PyUL 2.30.37:

* cell-centred grids (`CellCenteredGrid=True`), used by the paper runs;
* manual dynamical friction (`ManualDF`, `DF_A`, `df_use_flow`), used by the paper's reduced models;
* a resampled, COM-centred wavefunction cube at every save (`CenterCalc=True`, `3WfnRS`), the input of the
  eigenmode analysis;
* since 30.50: portability (Python 3.9 - 3.14, NumPy 2, SciPy >= 1.17, optional pyFFTW/numba/h5py/IPython),
  safe non-interactive behaviour on clusters, complete provenance records, many bug fixes
  ([CHANGELOG.md](CHANGELOG.md)), and an **inline eigenmode decomposition** `evolve(Cnlm=True)` that writes
  the soliton-eigenmode coefficients `c_nlm(t)` while the simulation runs (with the companion
  [ULDM-Eigenmode-Toolkit](https://github.com/Ailun-Zhang/ULDM-Eigenmode-Toolkit)).

The integrator evolves a rigid N-body ensemble and a mesh-based ULDM field against a non-expanding
background. It produces N-body state vectors, ULDM densities, potentials, phases, momenta, energies and
centres of mass, and (with the toolkit) the eigenmode spectrum `c_nlm(t)` of the ULDM around its centre of mass.

---

## How to cite

Please cite this software with the metadata in [CITATION.cff](CITATION.cff) (GitHub shows a "Cite this
repository" button), **together with** the upstream works it lists under `references`:

* PyUltraLight: Edwards, Kendall, Hotchkiss & Easther, JCAP 10 (2018) 027, doi:10.1088/1475-7516/2018/10/027.
* PyUltraLight 2: Y. Wang and Auckland Cosmology, Zenodo, doi:10.5281/zenodo.6336160.

For work that uses the SK features (black-hole orbits in solitons, manual dynamical friction, the eigenmode
decomposition) please also cite the paper:

```text
A. Zhang, Y. Wang, J. L. Zagorac and R. Easther,
"Stone Skipping Black Holes in Ultralight Dark Matter Solitons",
arXiv:2602.11512 [astro-ph.CO] (2026), https://arxiv.org/abs/2602.11512
```

Each GitHub release of this repository is archived on Zenodo with its own DOI. Release 30.50 (tag
`v30.50`) is [doi:10.5281/zenodo.23454002](https://doi.org/10.5281/zenodo.23454002); the concept DOI
[10.5281/zenodo.23454001](https://doi.org/10.5281/zenodo.23454001) always resolves to the latest
release. CITATION.cff lists the paper under `references`.

---

## Versions and reproducing the paper core

| `S_version` | Status | Where |
|---|---|---|
| **30.50** | This release (October 2026): maintenance release of 30.40 | `PyUltraLight2/Current_SK.py` at the release tag `v30.50` |
| **30.40** | SK release of January 2026; core of the paper runs for Fig. 1 (five of six masses) and the N = 128 and N = 192 resolution runs of Fig. 2 | git commit `ccab161`, `PyUltraLight2/Current_SK.py` (git blob `fbcc686`, sha256 `c112e0221aaefeb536ef1c39a56b81997f274d28731319beb639db08144ffee3`, LF line endings) |
| 30.38 | Pre-release of 30.40 (January 2026); the N = 256 resolution run of Fig. 2, which is also the 30 000 Myr run of Fig. 12 | not in this repository's history (not published) |
| 26.32 | Pre-release SK lineage (local files that kept the upstream version number 2.26.32); all other simulation runs of the paper (the 2 % case of Fig. 1 and the runs of Figs. 4 - 11) | the SK variants are not in this repository's history and are not published (only the upstream 26.32 file is in the history); several different files carry "26.32", so the number does not identify a unique file |
| 30.37 | Upstream PyUL ("November 13 2024, Dynamic Special 2") | `PyUltraLight2/Previous.py` (not imported) |

Every run records the version that produced it in `reproducibility.uldm` (`"Integrator Version"`) and in
`config.uldm` (`"PyUL Version"`). Since 30.50 it also records the sha256 of the core file and the library
versions (see [`reproducibility.uldm`](#reproducibilityuldm)). `S_version` is a float, so 30.50 is stored
and printed as `30.5`. The provenance of the paper runs (which run used which core) is documented in the
paper's data deposit on Zenodo ([doi:10.5281/zenodo.23454343](https://doi.org/10.5281/zenodo.23454343)) and summarised in
[docs/known_issues_and_errata.md](docs/known_issues_and_errata.md#which-pyul-version-made-which-figure).
Figure numbers in this README are the printed figure numbers of the paper (13 figures). The resampled cube
of 30.38 and 30.40 (C1, below) makes the eigenmode amplitudes of Fig. 2 and Fig. 12 up to 2.4 % too high
(frequencies and conclusions unchanged); see the errata page.

### What changed between 30.40 and 30.50

[CHANGELOG.md](CHANGELOG.md) lists every change with an ID and a policy:

* **A** (fixed outright): crashes, wrong results in code paths the paper did not use, unit bugs in helpers,
  portability, unsafe interactive or destructive behaviour, missing metadata.
* **B** (legacy default kept, opt-in switch added).
* **C** (fixed by default, legacy switch kept): only **C1**, the resampled box (below).

With default options and the pyFFTW backend, 30.50 reproduces every array of 30.40 bit for bit
(NBody, DF, ULDCOM, all energy lists, ULDMass, initial and final psi and particle states, periodic and
isolated potential) on the 64^3 regression runs in the reference environment (Python 3.9.25, NumPy 1.26.4,
SciPy 1.13.1, pyFFTW 0.14, 4 FFT threads). With NumPy 2.5 / SciPy 1.18 psi, NBody and DF are still bitwise
identical and the COM and energy lists differ at round-off (<= 6e-15 relative).

### Reproducing 30.40 behaviour exactly

1. Pass `LegacyResampleBox=True` to `evolve`. This restores the 30.38 - 30.40 resampled cube (C1): on a
   cell-centred grid 30.40 resampled onto a side of `-2*gridvec[0]*Length_Ratio = (L - dx)*Length_Ratio`
   instead of `L*Length_Ratio`, a 1/N radial scale error of 1.6 %, 0.78 %, 0.52 % and 0.39 % at
   N = 64, 128, 192 and 256 in the `3WfnRS`, `3DensityRS` and `2DensityRS` outputs. The dynamics are never
   affected, and with `CellCenteredGrid=False` both cubes are identical.
2. Keep the pyFFTW backend (do not set `PYUL_FFT_BACKEND=scipy`; scipy.fft agrees only to about 1e-11).
3. Use the same thread count as the run you compare with (`PYUL_NUM_THREADS`; it is recorded since 30.50).
4. In batch jobs pass `OverwriteOutputs=True` (and `Overwrite=True` to `GenerateConfig`) where 30.40 would
   have deleted existing outputs silently (see [non-interactive safety](#non-interactive-safety)).

```python
PyUL.evolve(save_path, run, CellCenteredGrid=True, LegacyResampleBox=True, OverwriteOutputs=True)
```

To evolve a run folder made by 30.50 with the 30.40 file itself, note that 30.40's `LoadConfig` refuses
any `config.uldm` whose `"PyUL Version"` is newer than its own (`RuntimeError: Configuration file
generated by a newer version of PyUL.`). The format is unchanged, so set that entry to `30.4` in a copy of
the run folder (`cfg["PyUL Version"] = 30.4` with `json`), or generate the configuration with 30.40.
30.50 still loads the configurations written by older versions.

Bug fixes that cannot be switched back (they only affect options the paper runs did not use):
psi no longer gains a factor `i` at every `Momentum`/`AngMomentum`/`2Momentum` save (P12); the
particle-particle force with the Plummer smoothing switched off (`a >= 1e8`) is attractive (P13);
`MassChange` uses each particle's own mass (P14); runs without the `Energy` flag, `ComputeQuad=True`,
`AutoStop2=True` with `NBodyGravity=False` and unsupported `NS` values no longer crash or misbehave
(P8 - P11). To run the 30.40 file itself, check out commit `ccab161`
(`git show ccab161:PyUltraLight2/Current_SK.py`).

---

## Installation

PyUL_SK supports **Python 3.9 - 3.14** on Windows, macOS and Linux. Tested in three environments:
Python 3.9.25 with the cluster versions (NumPy 1.26.4, SciPy 1.13.1, pyFFTW 0.14, numba 0.60, h5py 3.12.1),
Python 3.13 (NumPy 2.5.4, SciPy 1.18.1, pyFFTW 0.15.1, numba 0.68), and Python 3.14 without IPython.

### pip (editable install, recommended)

```bash
git clone https://github.com/Ailun-Zhang/PyUL_SK.git
cd PyUL_SK
python -m pip install -e ".[all]"        # core + pyFFTW, h5py, numba, IPython, psutil
```

The editable install is the supported route: the data folders `Soliton Profile Files/` and
`Green Functions/` stay in the repository next to the package. For a non-editable install set
`PYUL_DATA_DIR` to a folder that contains `Soliton Profile Files/`; new IsoP Green functions are then
cached in `PYUL_CACHE_DIR` or the user cache, never in site-packages. The source distribution (sdist)
is a minimal checkout: package, data files, examples and the test suite. The installed distribution is called
`pyultralight2-sk`; the import name is `PyUltraLight2`.

| Extra | Installs | Needed for |
|---|---|---|
| (core) | numpy, scipy, numexpr, matplotlib | everything |
| `fftw` | pyFFTW | the default FFT backend (used for the paper runs). Without it PyUL uses `scipy.fft` automatically. |
| `hdf5` | h5py | `save_format='hdf5'` and the hdf5 loaders only |
| `numba` | numba | speed of embedded-soliton initialisation and `Momentum`/`AngMomentum` saves only; without it they run as plain Python and a one-line notice is printed at import |
| `notebook` | IPython, Jupyter | the shipped notebooks; without IPython `clear_output` is a no-op |
| `bench` | psutil | machine information in `Benchmark.py` |
| `test` | pytest, pytest-timeout | the test suite |
| `cnlm` | pyshtools, h5py | `evolve(Cnlm=True)`; also needs the ULDM-Eigenmode-Toolkit (not on PyPI, see [below](#requirements)) |
| `all` | pyFFTW, h5py, numba, IPython, psutil | production runs |

Plain requirement files are also provided: `requirements.txt` (core), `requirements-full.txt` (core + all
optional), `requirements-test.txt`.

### conda

```bash
conda env create -f environment.yml      # conda-forge, environment name pyul-sk
conda activate pyul-sk
python -m pip install -e . --no-deps
```

`environment.yml` pins `python>=3.9`; change the pin if you need a specific version. The production
environment of the paper (an HPC cluster running PBS Pro) was Python 3.9 with NumPy 1.26.4, SciPy 1.13.1, numba 0.60,
pyFFTW 0.14 and h5py 3.12.1.

### Optional dependencies and fallbacks

| Missing | Effect |
|---|---|
| pyFFTW | `scipy.fft` backend with the same plan interface; results agree with pyFFTW to about 1e-11 relative, not bitwise. Force it with `PYUL_FFT_BACKEND=scipy`. |
| numba | `initsoliton` (embedded solitons) and `LUQuick` (angular momentum) run as plain Python (slow, same results). |
| h5py | only `save_format='hdf5'` fails, with a clear error. |
| IPython | no `clear_output`; nothing else changes. |
| ffmpeg | only the media helpers of the notebooks need it. |

---

## Quick start on a laptop

### The example driver

```bash
python examples/run_stone_skipping_local.py            # 64^3, 2.5 kpc box, 100 Myr, 10 snapshots
python examples/run_stone_skipping_local.py --resol 96 --duration 300 --out runs --name test96
python examples/run_stone_skipping_local.py --help
```

The driver follows the recipe of the paper drivers: one black hole of mass `0.02 mS` on a circular orbit at
0.14 kpc from a soliton of `mS = 12e6 Msun` (axion mass 1e-21 eV), barycentric initial conditions,
Plummer radius `rP = dx/2`, `CellCenteredGrid=True`. With the defaults a run takes about 5 - 10 s on a
recent workstation (5.2 s measured with 4 FFT threads; grids below 128^3 use at most 4 threads) and writes
about 8 MB (mostly the initial and final wavefunction dumps). Outputs go to `examples/output/<run>/` unless `--out` is given. The paper production runs used
`--resol 128/192/256` and 3000 Myr on 16 - 56 cores (see [examples/README.md](examples/README.md)).

### A minimal Python script

```python
import os
os.environ["PYUL_AXION_MASS_EV"] = "1e-21"     # read once, at import: set it first
import numpy as np
import PyUltraLight2 as PyUL

save_path = "runs"                             # parent folder of the run folders
resol, length = 64, 2.5                        # grid points per side, box side in kpc
mS, mBH, R = 12.0, 0.24, 0.14                  # soliton and BH mass (1e6 Msun), separation (kpc)

dx = PyUL.MeshSpacing(resol, length, 'kpc')
a = PyUL.GenPlummer(dx / 2, 'kpc')             # Plummer smoothing radius rP = dx/2
_, v = PyUL.DefaultSolitonOrbit(resol, length, 'kpc', mS, 'M_solar_masses', R, 'kpc', 'km/s')
M = mS + mBH
solitons = [[mS, [-R * mBH / M, 0, 0], [0, v * mBH / M, 0], 0.0]]   # [mass, [x,y,z], [vx,vy,vz], phase]
particles = [[mBH, [R * mS / M, 0, 0], [0, -v * mS / M, 0]]]        # [mass, [x,y,z], [vx,vy,vz]]

run = PyUL.GenerateConfig(
    NS=32, length=length, length_units='kpc', resol=resol,
    duration=100, duration_units='Myr', step_factor=1, save_number=10,
    save_options='Energy NBody DF ULDCOM', save_path=save_path, save_format='npy',
    s_mass_unit='M_solar_masses', s_position_unit='kpc', s_velocity_unit='km/s', solitons=solitons,
    start_time=0, m_mass_unit='M_solar_masses', m_position_unit='kpc', m_velocity_unit='km/s',
    particles=particles, embeds=[], Uniform=False, Density=0, density_unit='', a=a, UVel=[0, 0, 0],
    NoInteraction=True, Name='quickstart', Overwrite=True)

PyUL.evolve(save_path, run, CellCenteredGrid=True, OverwriteOutputs=True)

# Read the black-hole orbit back.
out = f"{save_path}/{run}/Outputs"
t_myr = np.load(f"{out}/snapshot_times_Myr.npy")
nbody = np.array([np.load(f"{out}/NBody/NTM_#{k:03d}.npy") for k in range(len(t_myr))])
r_kpc = PyUL.convert_back(np.linalg.norm(nbody[:, 0:3], axis=1), 'kpc', 'l')
print(np.c_[t_myr, r_kpc])
```

`GenerateConfig` writes `runs/quickstart/config.uldm`; `evolve` reads it and writes everything else into
that run folder. `Overwrite=True` and `OverwriteOutputs=True` allow the script to be re-run; leave them out
to protect existing results.

---

## Running on clusters (PBS, Slurm)

Templates that run the example driver: [examples/pbs/stone_skipping.pbs](examples/pbs/stone_skipping.pbs)
and [examples/slurm/stone_skipping.sbatch](examples/slurm/stone_skipping.sbatch). They use
`set -euo pipefail`, write one log per job under `logs/` with the job id and a time stamp, do not pipe
through `tee` (the job's exit status is the driver's), pin no host, and export one explicit thread count.

```bash
qsub -v PYUL_REPO=$HOME/PyUL_SK,RUN_ARGS="--resol 192 --duration 3000 --name f002_192" examples/pbs/stone_skipping.pbs
mkdir -p logs
sbatch --export=ALL,PYUL_REPO=$HOME/PyUL_SK,RUN_ARGS="--resol 192 --duration 3000" examples/slurm/stone_skipping.sbatch
```

Edit the resource request, the queue or partition and the environment activation for your site.

**Threads.** PyUL uses one thread count for pyFFTW, numexpr and scipy.fft, resolved in this order:
`PYUL_NUM_THREADS` > `OMP_NUM_THREADS` > `SLURM_CPUS_PER_TASK` > `NCPUS` (PBS) > CPU affinity >
`os.cpu_count()`. Grids with `resol < 128` are capped at 4 threads unless `PYUL_NUM_THREADS` is set. The
count and its source are printed and recorded in `reproducibility.uldm`. Some sites export
`OMP_NUM_THREADS=1` on login nodes; the templates therefore set `PYUL_NUM_THREADS` explicitly from
`NCPUS` / `SLURM_CPUS_PER_TASK`.

**Progress output.** On a terminal or in Jupyter PyUL draws the legacy progress bar. When stdout is not a
terminal (batch logs) it prints one compact ASCII line per snapshot. `PYUL_PROGRESS=bar|lines|quiet`
overrides the choice. Console output never raises `UnicodeEncodeError` (non-encodable characters fall back
to ASCII); log files are UTF-8.

### Non-interactive safety

PyUL never calls `input()` in a non-interactive session (stdin is not a terminal and no Jupyter kernel;
`PYUL_INTERACTIVE=0|1` overrides the detection). Destructive branches need explicit consent:

| Situation | Non-interactive session (batch job, pipe, CI) | Interactive session (unchanged) |
|---|---|---|
| `evolve` with an existing `<run>/Outputs` | `FileExistsError` unless `OverwriteOutputs=True` (`Silent=True` alone no longer deletes) | asks `[Y/n]`; `Silent=True` deletes without asking |
| `GenerateConfig` with an existing run folder | `FileExistsError` unless `Overwrite=True` (`NoInteraction=True` alone no longer deletes) | asks `[Y/n/C]` (`C` reuses the existing config); with `NoInteraction=True` deletes without asking |
| `DSManagement(path, Force=False)` on a non-empty folder | keeps the folder | asks |
| `DSManagement(path, Force=True)` on a non-empty folder | deletes it | deletes it |
| `Runs(save_path)` with several runs | returns the latest run (`latest.uldm`) | asks which run to load |
| `BHRatioTester` does not converge | raises `ValueError` | asks |

End-of-file on stdin counts as "no". With `Cnlm=True` an existing `c_nlm.h5` (or `a_lm.h5`) inside
`<run>/Outputs` follows the `Outputs` row; one outside `Outputs` (absolute `CnlmOutput`/`CnlmSaveAlm`)
needs `OverwriteOutputs=True`.

Every recursive delete of a user path (`DSManagement`, `evolve`'s `Outputs`, `GenerateConfig`'s run
folder, `Runs`' `X<n>` answer) raises `ValueError` instead when the folder resolves to the filesystem
root, the home directory, the current working directory or one of its ancestors, or the repository root
or one of its ancestors (for example `DSManagement('~', Force=True)` or `DSManagement('.', Force=True)`).

---

## Environment variables

All are optional.

| Variable | Effect |
|---|---|
| `PYUL_AXION_MASS_EV` | Axion mass in eV, read **once at import** (default `1e-22`; the paper used `1e-21`). |
| `PYUL_NUM_THREADS` | FFT / numexpr / scipy.fft thread count (highest priority; bypasses the 4-thread cap for `resol < 128`). |
| `OMP_NUM_THREADS`, `SLURM_CPUS_PER_TASK`, `NCPUS` | Fallback thread sources, in this order. |
| `PYUL_FFT_BACKEND` | `pyfftw` (default when importable) or `scipy`. |
| `PYUL_PROGRESS` | `bar`, `lines` (default when stdout is not a terminal) or `quiet`. |
| `PYUL_INTERACTIVE` | `0` never prompt, `1` always allow prompts. |
| `PYUL_DATA_DIR` | Folder that contains `Soliton Profile Files/`. Data are looked up there, then in the working directory (legacy behaviour), then in the repository. |
| `PYUL_CACHE_DIR` | Where IsoP Green functions and generated soliton profiles are cached. |

**Green functions (`IsoP=True`).** `G{resol}.npy` (about `(resol+1)^3 * 8` bytes, 136 MB at 256) is read
from `PYUL_CACHE_DIR`, the data folders or the user cache, and a new one is written atomically to
`PYUL_CACHE_DIR`, else the repository's `Green Functions/` if PyUL runs from a source checkout
(editable install) and that folder is writable, else the user cache
(`%LOCALAPPDATA%\PyUltraLight2\Cache` on Windows, `~/.cache/PyUltraLight2` elsewhere). A non-editable
install never writes into site-packages. Unreadable or
truncated cache files are ignored. `G96.npy` is shipped. Caches generated with different SciPy versions
differ at round-off (about 1e-17 relative).

---

## Units and the axion mass

PyUL works in the code units of PyUltraLight, defined by the axion mass `m`, `H_0 = 67.7 km/s/Mpc` and
`Omega_m0 = 0.31`:

| Quantity | Code unit | at `m = 1e-21 eV` | at `m = 1e-22 eV` |
|---|---|---|---|
| time | `(3 H_0^2 Omega_m0 / 8 pi)^(-1/2)` (independent of `m`) | 75 134.064 Myr | 75 134.064 Myr |
| length | `(8 pi hbar^2 / (3 m^2 H_0^2 Omega_m0))^(1/4)` | 12.133096 kpc | 38.368218 kpc |
| mass | `(3 H_0^2 Omega_m0 / 8 pi)^(1/4) hbar^(3/2) / (m^(3/2) G)` | 70 457.69 Msun | 2 228 067.7 Msun |
| velocity | length / time | 0.158009 km/s | 0.499668 km/s |

A soliton of 12e6 Msun is 170.3149822613627 code masses at 1e-21 eV. Unit strings accepted by `convert`,
`convert_back` and `convert_between` (and in `config.uldm`):

| Kind | Strings |
|---|---|
| length `'l'` | `''` (code), `m`/`SI`, `km`, `pc`, `kpc`, `Mpc`, `ly` |
| mass `'m'` | `''`, `kg`/`SI`, `solar_masses`, `M_solar_masses` (1e6 Msun) |
| time `'t'` | `''`, `s`/`SI`, `yr`, `kyr`, `Myr`, `Gyr` |
| velocity `'v'` | `''`, `m/s`/`SI`, `km/s`, `km/h`, `c` |
| density `'d'` | `''`, `Crit`, `MSol/pc3`, `MMSol/kpc3`, `kg/m3`/`SI` |

The axion mass is read from `PYUL_AXION_MASS_EV` when `PyUltraLight2` is imported and cannot be changed
afterwards in the same process. `config.uldm` does not store it; `reproducibility.uldm` does
(`"Axion Mass"` in kg and `"m22"` = m / 1e-22 eV). The ULDM-Eigenmode-Toolkit reads it from there.

The wavefunction is the macroscopic field in code units (`|psi|^2` is the mass density in code units);
grid coordinates run from `-L/2` to `L/2` in each direction (`gridvec`, node-centred:
`linspace(-L/2, L/2, N, endpoint=False)`; cell-centred: `-L/2 + dx/2 + k dx`).

---

## Initial conditions: `GenerateConfig` and `config.uldm`

```text
GenerateConfig(NS, length, length_units, resol, duration, duration_units, step_factor, save_number,
               save_options, save_path, save_format, s_mass_unit, s_position_unit, s_velocity_unit,
               solitons, start_time, m_mass_unit, m_position_unit, m_velocity_unit, particles, embeds,
               Uniform, Density, density_unit, a, UVel, NoInteraction=False, Name='', Overwrite=False)
```

| Argument | Meaning |
|---|---|
| `NS` | N-body integrator: `0` particles drift without forces, `1` one Euler step, a multiple of 4 = `NS/4` RK4 sub-steps per ULDM step (the paper used 32). Other values raise `ValueError`. |
| `length`, `length_units` | box side and its unit |
| `resol` | grid points per side |
| `duration`, `duration_units` | simulated time and its unit |
| `step_factor` | divides the number of ULDM steps (the step is `h = t / steps`, minimum steps `t / (dx^2/pi)`); 1 for the default step |
| `save_number` | number of snapshots after the initial one (`-1`: every step) |
| `save_options` | space-separated save flags (see [Outputs](#outputs-and-save-options)) or `'Minimum'` |
| `save_path` | parent folder of the run folders (`~` is expanded) |
| `save_format` | `'npy'`, `'npz'` or `'hdf5'` |
| `s_mass_unit`, `s_position_unit`, `s_velocity_unit` | units of `solitons` and `embeds` |
| `solitons` | list of `[mass, [x, y, z], [vx, vy, vz], phase]` (standard solitons, profile `Soliton Profile Files/initial_f.npy`) |
| `start_time` | start time in `duration_units` (enters the soliton phases and the snapshot times) |
| `m_mass_unit`, `m_position_unit`, `m_velocity_unit` | units of `particles` |
| `particles` | list of `[mass, [x, y, z], [vx, vy, vz]]` (mass 0 = observer) |
| `embeds` | list of `[total mass, [x, y, z], [vx, vy, vz], BH-to-total mass ratio, phase]` (soliton with a central black hole; the black hole is added to `particles`) |
| `Uniform`, `Density`, `density_unit`, `UVel` | uniform background field of the given density and velocity (the background velocity uses `s_velocity_unit`) |
| `a` | Plummer smoothing parameter of the particle potential `-m a / sqrt(1 + a^2 r^2)`; build it with `GenPlummer(rP, length_units)`. `a >= 1e8` switches the smoothing off (pure 1/r). |
| `NoInteraction` | `True`: never ask for a run name |
| `Name` | run folder name (default: time stamp `YYYYMMDD_HHMMSS@<resol>`) |
| `Overwrite` | replace an existing run folder of the same name (30.50) |

`GenerateConfig` returns the run folder name. Helpers: `MeshSpacing(resol, length, length_units)` (grid
spacing), `GenPlummer(rP, length_units)` / `RecPlummer(a, length_units)`, `DefaultSolitonOrbit(resol,
length, length_units, s_mass, s_mass_unit, m_radius, m_position_unit, m_velocity_unit)` (enclosed soliton
mass and circular speed at radius `m_radius`), `VizInit2D(...)` (2D preview of the initial grid).

`config.uldm` is a JSON file (UTF-8). `LoadConfig` reads configs of every version, including those written
by 30.40 and older (`"Spacial Resolution"`, list-valued `"Flags"`, `"Field Smoothing"`). A config written by
a newer PyUL is refused. Its keys:

| Key | Content |
|---|---|
| `PyUL Version`, `PyUL Version Descriptor` | `S_version`, `D_version` of the writer |
| `Description`, `Config Generation Time` | "Batch Configuration" when `NoInteraction=True`, else "Single Run Configuration"; time stamp |
| `Save Options` | `Flags` (string), `All Supported Flags`, `Number` (save_number), `Format` |
| `Spatial Resolution`, `Temporal Step Factor`, `RK Steps` | `resol`, `step_factor`, `NS` |
| `Duration` | `Time Duration`, `Start Time`, `Time Units` |
| `Simulation Box` | `Box Length`, `Length Units` |
| `ULDM Solitons` | `Condition` (solitons), `Embedded`, `Mass Units`, `Position Units`, `Velocity Units` |
| `Matter Particles` | `Plummer Radius` (in length units), `Condition` (particles), units |
| `Central Mass` | always 0 |
| `Uniform Field Override` | `Flag`, `Density Unit`, `Density Value`, `Uniform Velocity` |

All `.uldm` files are UTF-8 text; `config.uldm` and `reproducibility.uldm` are JSON.

---

## `evolve()` parameters

```text
evolve(save_path, run_folder, **options)
```

reads `save_path/run_folder/config.uldm` and runs the simulation. The table lists **every** parameter of the
30.50 signature with its default and its effect as implemented. "Inactive" parameters are accepted for
compatibility and have no effect; since 30.50 a `UserWarning` lists them when they are set.

### Paths and I/O

| Parameter | Default | Meaning |
|---|---|---|
| `save_path` | (required) | parent folder of the run folders (`~` is expanded) |
| `run_folder` | (required) | run folder name (as returned by `GenerateConfig`); `''`/`None` does nothing |
| `DumpInit` | `False` | write the initial state to `<run>/Init_psi.npy` and `<run>/Init_TM.npy`, then continue (since 30.40 it no longer stops the run) |
| `DumpFinal` | `False` | write the final state to `<run>/Final_psi.npy` and `<run>/Final_TM.npy` |
| `UseInit` | `False` | start from a stored wavefunction `<InitPath>_psi.npy` (same `resol`); the config's solitons are still built and blended |
| `InitPath` | `''` | path prefix of the stored wavefunction (required with `UseInit`) |
| `InitWeight` | `1` | with `UseInit`: `w` in [0, 1] gives `psi = (1-w) psi_config + w psi_file`; `-1` multiplies `psi_config` by the phase of `psi_file`; other values raise `ValueError` (30.50) |
| `GenerateLog` | `True` | write `<run>/evolve_<timestamp>.log` |
| `Silent` | `False` | clear the console at the start of the loop and print `D_version` and `Message`; in an **interactive** session also delete an existing `Outputs/` without asking |
| `Message` | `''` | text printed with `Silent=True` |
| `OverwriteOutputs` | `False` | explicit consent to delete an existing `<run>/Outputs` (and an existing inline `c_nlm.h5`) (30.50) |
| `Stream` | `False` | append the particle state after every RK4 sub-step to `<run>/NBStream.uldm` (only with `NBodyInterp=True` and `NS` a multiple of 4) |
| `StreamChar` | `[0]` | indices of the 6N state vector written by `Stream` |
| `PrintEK` | `True` | show the change of kinetic energy in the progress line (needs the `Energy` flag) |

### Physics switches

| Parameter | Default | Meaning |
|---|---|---|
| `IsoP` | `False` | isolated (zero-padded, non-periodic) ULDM potential with a cached Green function; `False` = periodic FFT Poisson solve |
| `SelfGravity` | `True` | the ULDM evolves in its own potential plus the particles' potential; `False`: in the particles' potential only (the ULDM potential is still computed and still acts on the particles) |
| `NBodyInterp` | `True` | particles feel the ULDM potential (interpolated gradient); `False`: only mutual gravity (and manual DF) |
| `NBodyGravity` | `True` | the ULDM feels the particles' potential; `False`: particle potential set to zero |
| `CellCenteredGrid` | `False` | grid points at cell centres `-L/2 + dx/2 + k dx` (all paper runs: `True`); `False`: `linspace(-L/2, L/2, N, endpoint=False)` |
| `ExtPhi` | `0` | external potential (scalar or `resol^3` array, code units) added to the ULDM potential at every step; acts on ULDM and particles |
| `ManualDF` | `False` | add a manual dynamical-friction acceleration to every particle (the empirical drag of the paper's reduced models) |
| `DF_A` | `0.0` | dimensionless amplitude `A` of the manual DF: `a_DF = -A (4 pi/3) m_i \|r\|^2 rho(x_i) v_hat` in code units, `\|r\|` measured from the box origin |
| `df_use_flow` | `False` | manual DF along the velocity relative to the local ULDM flow (phase gradient) instead of the particle velocity (falls back to the velocity near the box edge) |
| `MassChange` | `False` | particle masses vary in time: `m_i(t) = m_i * MassFunc(t)` |
| `MassFunc` | `""` | callable of the code time used by `MassChange` |
| `Shift` | `False` | shift the particles' initial positions by `-dx/2` in x, y and z |
| `DR` | `1` | stretch factor of standard solitons: the profile is evaluated at `sqrt(x^2 + y^2 + DR z^2)` (1 = spherical; embedded solitons ignore it) |
| `EdgeClear` | `False` | at every step reset slabs of `resol//16` points at all six faces to the uniform background value `sqrt(Density)` (0 without `Uniform`) |
| `UseDispSponge` | `False` | dispersive sponge: damp psi by `exp(-V(r) h)` with a smooth tanh step `V(r)` (centre `r = 7L/16`, width `L/8`, from about 0 inside to 0.6 outside), adapted from Zagorac et al. |

### Centre of mass, resampled cube and diagnostics

| Parameter | Default | Meaning |
|---|---|---|
| `CenterCalc` | `False` | compute the ULDM centre of mass (`\|psi\|^2`-weighted mean position, non-periodic) at every step (logged to the log file) and write the resampled outputs `3WfnRS`, `3DensityRS`, `2DensityRS` at every save |
| `Length_Ratio` | `0.5` | side of the resampled cube / box side |
| `resolR` | `64` | points per side of the resampled cube |
| `LegacyResampleBox` | `False` | `True`: 30.38 - 30.40 cube side `(L - dx) Length_Ratio` on cell-centred grids (bit-for-bit 30.40); `False`: `L Length_Ratio` (30.50) |
| `CorrectDrift` | `False` | every `CorrectFreq` steps, give the ULDM a Galilean kick that cancels its COM drift velocity measured over two steps (the particles are not kicked); forces `CenterCalc=True` |
| `CorrectFreq` | `6` | interval (steps) of `CorrectDrift` |
| `ComputeQuad` | `False` | write the quadrupole and dipole moments about the current ULDM COM to `Outputs/Quadrupole/QIJ_#NNN` at every save |

### Stopping conditions

| Parameter | Default | Meaning |
|---|---|---|
| `AutoStop` | `False` | with exactly one particle and a uniform background: stop at the next save after the particle's y velocity (relative to the background) changes sign; the stopping time (code units) is written to `<run>/StoppingTime.uldm` (`-1` if it never stops) |
| `AutoStop2` | `False` | stop when the minimum of the total potential falls below `WellThreshold` times the initial particle potential depth (`min(phi_particles)` for `a = 0`, else `-a max(m)`) |
| `WellThreshold` | `100` | threshold factor of `AutoStop2` |
| `AutoStop3` | `False` | **inactive** (not implemented; only recorded in `reproducibility.uldm`) |
| `KEThreshold` | `0.9` | **inactive** (only for the unimplemented `AutoStop3`) |

### Inactive legacy parameters

| Parameter | Default | Meaning |
|---|---|---|
| `Simpson` | `False` | **inactive** (not implemented) |
| `NLM` | `False` | **inactive** (not implemented; use `Cnlm`) |

### Inline eigenmode decomposition (30.50)

| Parameter | Default | Meaning |
|---|---|---|
| `Cnlm` | `False` | decompose the resampled cube into soliton eigenmodes at every save and write `Outputs/c_nlm.h5`; see [below](#inline-eigenmode-decomposition-c_nlm) |
| `CnlmEigenFile` | `None` | eigenfunction HDF5 (`ell/<l>/{r,E,f}`); `None` or `'auto'` builds one with the paper defaults |
| `CnlmLmax` | `10` | maximum degree l |
| `CnlmNShells` | `257` | number of radial shells |
| `CnlmRmax` | `None` | outer shell radius in the box length units (`None`: half the resampled cube side) |
| `CnlmNumPoints` | `300` | N of the N x 2N angular grid |
| `CnlmAngularGrid` | `'legacy'` | `'legacy'` (paper grid) or `'dh'` (Driscoll-Healy) |
| `CnlmSaveAlm` | `False` | also write the a_lm file (`True`: `Outputs/a_lm.h5`, or a file name) |
| `CnlmOutput` | `'c_nlm.h5'` | c_nlm file name relative to `Outputs` (or an absolute path) |
| `CnlmThreads` | `None` | threads for the decomposition (`None`: PyUL's FFT thread count) |
| `CnlmSolitonMass` | `None` | soliton mass (config soliton mass units) for the automatic eigen file |
| `CnlmEigenParams` | `None` | overrides for the automatic eigen file: `n_grid`, `lmax`, `n_eig`, `maxiter`, `tol`, `outer_boundary`, `mass0` |

**Time stepping.** The ULDM step is `h = t / n_steps`. `n_steps` is a multiple of `save_number` and at least
`int(int(t / (dx^2/pi) + 1) / step_factor)` (or `save_number` itself when that is larger); one snapshot is
written every `n_steps / save_number` steps. The kick-drift-kick split-step scheme applies the pending half
kick at every save, so the saved psi is at the snapshot time.

---

## Outputs and save options

A run folder `<save_path>/<run>/` contains `config.uldm`, `reproducibility.uldm`, the log
`evolve_<timestamp>.log`, `LocalDensities.uldm` (ULDM density at the last massive particle at t = 0, code
units), the optional `Init_*`/`Final_*` dumps, and `Outputs/`. `<save_path>/latest.uldm` holds the name of
the last run started there.

Per-save outputs are written to

```text
<save_path>/<run>/Outputs/<Flag>/<PREFIX>_#NNN.<npy|npz|hdf5>
```

with `NNN` = 000 for the initial state and 1 ... `save_number` for the snapshots. In `hdf5` files the
dataset is named after the flag. Separate flags with spaces (no commas). `save_options='Minimum'` means
`Energy 1Density NBody DF Entropy`. All arrays are in code units; `N` = `resol`, `NR` = `resolR`,
`P` = number of particles; planes are `z = N//2` (index), lines are `[N//2, :, N//2]` (along y).

| # | Flag | Folder / prefix | Shape | Content |
|---|---|---|---|---|
| 0 | `3Density` | `3Density/R3D` | (N, N, N) | `rho = \|psi\|^2` |
| 1 | `3Wfn` | `3Wfn/P3D` | (N, N, N) complex | psi |
| 2 | `2Density` | `2Density/R2D` | (N, N) | `rho[:, :, N//2]` |
| 3 | `Energy` | `Energy/` (stays empty) | - | energy lists, written as flat files (below) |
| 4 | `1Density` | `1Density/R1D` | (N,) | `rho[N//2, :, N//2]` |
| 5 | `NBody` | `NBody/NTM` | (6P,) | `[x, y, z, vx, vy, vz]` of every particle |
| 6 | `3Grav` | `3Grav/G3D` | (N, N, N) | ULDM potential `Phi_SP` (incl. `ExtPhi`) |
| 7 | `2Grav` | `2Grav/G2D` | (N, N) | `Phi_SP[:, :, N//2]` |
| 8 | `DF` | `DF/DYF` | (3P,) | ULDM potential gradient at each particle (sign: see [conventions](#known-issues-and-conventions)); `DYF_#000` is zeros |
| 9 | `2Phase` | `2Phase/A2D` | (N, N) | `angle(psi[:, :, N//2])` |
| 10 | `Entropy` | `Entropy/` (stays empty) | - | `-sum(rho log rho)` per save in `Outputs/Entro.npy` |
| 11 | `1Grav` | `1Grav/G1D` | (N,) | `Phi_SP[N//2, :, N//2]` |
| 12 | `3GravF` | `3GravF/F3D` | (N, N, N) | total potential acting on the ULDM (`Phi_SP + Phi_particles`, or `Phi_particles` without `SelfGravity`) |
| 13 | `2GravF` | `2GravF/F2D` | (N, N) | total potential, plane |
| 14 | `1GravF` | `1GravF/F1D` | (N,) | total potential, line |
| 15-17 | `3UMmt`, `2UMmt`, `1UMmt` | `V3D`, `V2D`, `V1D` | - | **reserved, never written** (warned) |
| 18 | `Momentum` | `Momentum/PMT` | (3,) | total ULDM momentum |
| 19 | `AngMomentum` | `AngMomentum/MVR` | (3,) | total ULDM angular momentum about the origin (slow; uses numba when available) |
| 20 | `3DensityRS` | `3DensityRS/R3R` | (NR, NR, NR) | density on the resampled cube (needs `CenterCalc=True`) |
| 21 | `3WfnRS` | `3WfnRS/P3R` | (NR, NR, NR) complex | psi on the resampled cube (needs `CenterCalc=True`; 268 MB per save at NR = 256) |
| 22 | `2EnergyTot` | `2EnergyTot/E2T` | (N, N) | total energy density, plane (needs `Energy`) |
| 23 | `2EnergyKQ` | `2EnergyKQ/E2K` | (N, N) | kinetic + quantum energy density, plane (needs `Energy`) |
| 24 | `2EnergySI` | `2EnergySI/E2G` | (N, N) | self-gravity energy density, plane (needs `Energy`) |
| 25 | `ULDCOM` | `ULDCOM/UCM` | (3,) | ULDM centre of mass (density-weighted mean position, non-periodic) |
| 26 | `2DensityRS` | `2DensityRS/R2R` | (NR, NR) | resampled density, plane `NR//2` (needs `CenterCalc=True`) |
| 27 | `Quadrupole` | `Quadrupole/QIJ` | (3, 4) | written only with `ComputeQuad=True`: lower triangle of `sum rho (3 x_i x_j - delta_ij r^2)` about the ULDM COM, dipole `sum rho x_i` in column 3 (no cell-volume factor) |
| 28 | `2Momentum` | `2Momentum/P2D` | (3, N, N) | momentum density (times cell volume) in the plane `z = N//2` |
| 29-30 | `3GravRS`, `3GravRSF` | `G3R`, `F3R` | - | **reserved, never written** (warned) |

**Resampled cube** (`3WfnRS`, `3DensityRS`, `2DensityRS`, and the inline c_nlm): linear interpolation of
psi (real and imaginary parts, no phase correction, no periodic wrap, 0 outside the grid) onto a
`resolR^3` cube of side `L * Length_Ratio` (30.50 default; `(L - dx) * Length_Ratio` with
`LegacyResampleBox=True` on cell-centred grids), points `linspace(-s/2, s/2, resolR, endpoint=False)`
around the ULDM COM, so the COM sits at index `resolR//2`.

**Momentum timing.** `Momentum`, `AngMomentum` and `2Momentum` are evaluated inside the step, before the
drift, i.e. half a step out of sync with the snapshot.

**Flat files in `Outputs/`** (rewritten at every save):

| File | Content |
|---|---|
| `egylist.npy` | total ULDM energy (with `Energy`) |
| `egpcmlist.npy` | ULDM energy in the particles' potential, `sum Phi_particles rho dV` (with `Energy`) |
| `egpcmMlist.npy` | `sum_i m_i Phi_SP(x_i)` (particle energy in the ULDM potential; with `Energy`) |
| `egpsilist.npy` | ULDM self-gravity energy `0.5 sum Phi_SP rho dV` (with `Energy`) |
| `ekandqlist.npy` | ULDM kinetic + quantum energy (with `Energy`) |
| `masseslist.npy` | ULDM mass (with `Energy`) |
| `ULDMass.npy` | ULDM mass in the grid (always) |
| `Entro.npy` | entropy-like sum (with `Entropy`) |
| `snapshot_times_code.npy`, `snapshot_times_Myr.npy` | time of every save written so far (30.50) |

**Time convention.** Snapshot `k` (k = 0 is the initial state) is at

```text
t_k = start_time + k * duration / save_number          (k = 0 ... save_number)
```

Since 30.50 the times of the snapshots actually written are stored in `Outputs/snapshot_times_code.npy` and
`Outputs/snapshot_times_Myr.npy`, rewritten at every save (complete also after an early AutoStop or a
crash). Some older analysis notebooks (including the upstream notebooks shipped here) used
`t_k = k * duration / (save_number + 1)`, which compresses the time axis by `save_number/(save_number+1)`
(the last of 501 frames is drawn at 2994 Myr instead of 3000 Myr).

**Loading.** `Load_npys(loc, save_options)`, `Load_Data(save_path, ts, save_options, save_number)`,
`IOLoad_npy(loc, Type, k)` and `IOLoad_h5(loc, Type, k)` read these files; the toolkit's
`Eig_ULDM_packages.trajectories.load_run(run_dir)` reads the NBody/ULDCOM/DF frames, the energies and the
time axis without PyUL.

---

## Inline eigenmode decomposition (c_nlm)

With `Cnlm=True`, `evolve` decomposes the ULDM wavefunction into the eigenmodes of the soliton potential
while it runs. At every save, including the initial state, it takes the resampled cube exactly as
`Resample3Box` makes it for `3WfnRS` (same `Length_Ratio`, `resolR`, `LegacyResampleBox` and centre),
computes `a_lm(r)` on 257 shells and projects it onto the eigenfunctions `f_nl(r)`. At the end (also after
`AutoStop`/`AutoStop2`) it writes `Outputs/c_nlm.h5` in exactly the format of the ULDM-Eigenmode-Toolkit's
`c_nlm.h5`, so the toolkit's analysis and plotting code reads it unchanged. The historical offline chain
(copying the toolkit into `Outputs`, r0 segments, per-shell PBS jobs, JSON merge, `compute_c_nlm.ipynb`) is
not needed, and `3WfnRS` does not have to be saved. With `Cnlm=False` (default) nothing is imported and
every output is bit-for-bit as without the feature.

### Requirements

* The [ULDM-Eigenmode-Toolkit](https://github.com/Ailun-Zhang/ULDM-Eigenmode-Toolkit) >= 0.3.0 (package
  `Eig_ULDM_packages`; not on PyPI): `pip install -e <path to ULDM-Eigenmode-Toolkit>`.
* pyshtools and h5py: `pip install -e ".[cnlm]"` in PyUL_SK, or
  `conda install -c conda-forge pyshtools h5py`.
* The run record: the decomposition takes the axion mass and the resampled-box geometry from the run's
  `reproducibility.uldm` (written by 30.50), not from `PYUL_AXION_MASS_EV`.

### Usage

```python
PyUL.evolve(save_path, run,
            CenterCalc=True, Length_Ratio=1, resolR=256,   # the resampled cube that is decomposed
            CellCenteredGrid=True, Cnlm=True,              # paper defaults for every Cnlm* option
            OverwriteOutputs=True)

from Eig_ULDM_packages.c_nlm_integrator import load_cnlm_from_h5
t_myr, c, meta = load_cnlm_from_h5(f"{save_path}/{run}/Outputs/c_nlm.h5")
print(len(t_myr), c[(0, 0, 0)][:3])                       # c[(n, l, m)] is a complex series
```

The defaults are the paper's analysis settings: `CnlmLmax=10`, `CnlmNShells=257` (`linspace(0, r_max, 257)`
with `r[0] = 1e-5` in the box length units), `CnlmRmax=None` (half the resampled-cube side),
`CnlmNumPoints=300`, `CnlmAngularGrid='legacy'`, and an eigenfunction file built automatically
(`CnlmEigenFile=None`) with the toolkit's paper defaults (5120 grid points including r = 0, `n_eig = 25`,
`lmax = CnlmLmax`, `r_max = CnlmRmax`, the run's soliton mass and axion mass), written to
`Outputs/Cnlm/eigs_mass_<mass><units>_Eig_25_lmax_10_nig_5120.h5`. In the cluster validation (128^3 run,
256^3 resampled cube, 5 kpc) the `r`, `f` and `E` datasets of this automatic file were bitwise identical
to those of the eigen file used for the paper (sha256 `2bb42868...`; differences in `f` and `E`: 0.0).
The file hash differs, because toolkit 0.3.0 adds provenance attributes.

`CnlmNShells=257` is a fixed default (the paper's 256^3 cubes have 257 shell radii). The toolkit
pipeline `uldm-cnlm RUN_DIR` defaults to `resolR + 1` shells instead, so on the same `3WfnRS` files the
two agree with default settings only for `resolR = 256`. For another `resolR` use
`uldm-cnlm RUN_DIR --from-inline` (which takes every setting from the inline run) or pass
`--n-shells 257`.

`CenterCalc=True` is recommended: PyUL then resamples at every save anyway and the decomposition reuses that
cube. With `CenterCalc=False` the cube is resampled additionally at each save, centred on the COM of the
current density (the rule of the `ULDCOM` output without `CenterCalc`); at 256^3 that costs about 7 s and
3.4 GB of temporary memory per save.

| Parameter | Default | Meaning and checks |
|---|---|---|
| `Cnlm` | `False` | switch the inline decomposition on |
| `CnlmEigenFile` | `None` | eigen HDF5 (`ell/<l>/{r,E,f}`); `None`/`'auto'` builds one. The automatic build is refused when ambiguous (not exactly one standard soliton, any embedded soliton, or `UseInit=True`) unless `CnlmSolitonMass` is given. An explicit file must exist, be readable, have `lmax >= CnlmLmax` and lie outside `<run>/Outputs` (which `evolve` replaces; copy a previous run's `Outputs/Cnlm/eigs_*.h5` elsewhere first). |
| `CnlmSolitonMass` | `None` | soliton mass (config soliton mass units) for the automatic eigen file |
| `CnlmEigenParams` | `None` | dict overriding the automatic eigen file: `n_grid`, `lmax`, `n_eig`, `maxiter`, `tol`, `outer_boundary` (`'legacy'`/`'rmax'`), `mass0` |
| `CnlmLmax` | `10` | maximum degree; must be <= the eigen file's lmax and <= `CnlmNumPoints/2 - 1` |
| `CnlmNShells` | `257` | number of shells (>= 2) |
| `CnlmRmax` | `None` | outer shell radius in the box length units (`None`: half the resampled cube); must equal the eigen grid's r_max (checked, no silent rescale) |
| `CnlmNumPoints` | `300` | N of the N x 2N angular grid (even) |
| `CnlmAngularGrid` | `'legacy'` | `'legacy'` (endpoint-inclusive grid of the paper) or `'dh'` (Driscoll-Healy; changes l >= 1 amplitudes by 0.1 - 1 %) |
| `CnlmSaveAlm` | `False` | also write the a_lm file in the toolkit's `create_sh_hdf5` layout (`True`: `Outputs/a_lm.h5`, or a file name) |
| `CnlmOutput` | `'c_nlm.h5'` | output name relative to `Outputs` (or an absolute path) |
| `CnlmThreads` | `None` | threads (`None`: PyUL's FFT thread count, capped at 4 for `resol < 128` unless `PYUL_NUM_THREADS` is set) |

Everything is checked before any output is written: dependencies, parameter values, the eigen file or an
unambiguous automatic build, and output paths (an existing `c_nlm.h5` inside `Outputs` is handled
together with the `Outputs` folder, as in [non-interactive safety](#non-interactive-safety); one outside
`Outputs` is refused unless `OverwriteOutputs=True`). Right after `reproducibility.uldm` is written and before the first save, the axion
mass, the resampled-box geometry and the r_max consistency of the eigen grid with the shells are checked. A
failure after the time loop has started (for example a full disk) is reported as a `RuntimeWarning`; the
simulation continues unchanged, without c_nlm.

### Output

`Outputs/c_nlm.h5`, identical in layout to the toolkit's `c_nlm.h5`:

* `/t`: float64, one value per save, in Myr: the real save times `start + k*duration/save_number`, the same
  values as `Outputs/snapshot_times_Myr.npy`.
* `/n/<n>/l/<l>/m_<m>`: complex128 series for n = 0 ... n_eig-1, l = 0 ... lmax, m = -l ... l. These are raw
  c_nlm in sqrt(code mass) units: `sum |c|^2` is the mass in code units; the paper plots
  `|c|^2 / (M (1 + eps^2))`.
* No other groups or datasets. Provenance is stored as root attributes only (ignored by every toolkit
  reader): `source='pyul-inline'`, `pyul_version`, `pyul_core_sha256`, `eigen_file`, `eigen_file_sha256`,
  `l_max`, `n_r`, `num_points`, `angular_grid`, `r_values`, `r_values_code`, `rs_side`, `rs_dx`, `rs_resol`,
  `rs_legacy_resample_box`, `rs_cell_centered_grid`, `axion_mass_ev`, `snapshot_indices`, `times`,
  `time_axis`, `stopped_early` and software versions.

Other files: `Outputs/Cnlm/alm_checkpoints/ALM_#NNN.npz` (one checkpoint per save, in the toolkit pipeline's
format), `Outputs/Cnlm/cnlm_inline.json` (resolved settings, status, timing), `Outputs/Cnlm/pipeline_params.json`
(kernel fingerprint), the automatic eigen file, and the optional `Outputs/a_lm.h5`. Nothing else in
`Outputs` changes.

### Crash recovery

If a run dies (wall time, node failure), assemble what was saved from the checkpoints:

```bash
uldm-cnlm runs/quickstart --from-inline        # -> <run>/Outputs/c_nlm.h5 from Outputs/Cnlm/alm_checkpoints
```

`--from-inline` takes every parameter from `Outputs/Cnlm/cnlm_inline.json` and computes a snapshot from its
`3WfnRS` file only when it has no checkpoint. It reproduces the inline file bit for bit. Final outputs are
never overwritten without `--force`.

### Cost

The cost per save is set by the number of interpolation points (`n_shells x N x 2N` = 46 M at the paper
settings), not by the cube size. Measured on a 16-core workstation:

| Resampled cube | Settings | 1 thread | 8 threads |
|---|---|---|---|
| 48^3 / 128^3 | paper (257 shells, N = 300, lmax 10) | 17.5 / 18.1 s | 3.6 / 4.0 s |
| 256^3 | paper | 21.0 s | 5.6 s |

On the cluster (56 threads) the overhead was 7 s per save at 256^3. For the paper's 501 saves at 256^3 this
adds about 47 min on 8 threads. The final write takes about 1 s; extra peak memory is about 0.25 GB
(128^3) to 0.47 GB (256^3) at the paper settings.

### Equivalence evidence

* Same process: the inline `c_nlm.h5` (and `a_lm.h5`) is **bit-for-bit identical** to the toolkit pipeline
  `uldm-cnlm` run on the saved `3WfnRS` cubes (48^3 and 64^3 runs, `CellCenteredGrid` and
  `LegacyResampleBox` each True and False, `CenterCalc` True and False, `Length_Ratio` 1 and 0.5), and to the
  historical JSON chain `process_file -> create_sh_hdf5 -> load_a_lm -> compute_cnlm`
  (`tests/test_cnlm_inline.py`).
* Another process: <= 3.2e-14 relative per mode against the historical chain. pyshtools' FFT is not
  bit-reproducible across processes; the historical chain differs from itself at the same level.
* Cluster validation (PBS job on the production cluster, Python 3.9.18, SciPy 1.13.1): on real paper
  snapshots (PyUL 26.32 and 30.40 runs) the toolkit kernel reproduces the historical a_lm JSON files to
  <= 4.2e-16 relative and the historical c_nlm files to <= 1.4e-16 of max|c|; in a 128^3 run with the paper
  settings the inline file equals the pipeline file bit for bit and the other 28 output files are
  byte-identical with and without `Cnlm`.

### Limitations

* Only the resampled cube is decomposed, centred on its `|psi|^2`-weighted COM (the paper's convention).
  Resampling is linear and non-periodic: mass outside the resampled cube counts as zero.
* The defaults reproduce the paper's numbers (legacy angular grid, legacy eigen-solver boundary), not the
  most accurate ones; see the toolkit README for `'dh'` and `outer_boundary='rmax'`.
* A run with a non-zero start time has `t[0] = start_time`, whereas the historical notebooks always started
  at 0.
* The toolkit's `units` axion mass is set to the run's value inside the PyUL process (process-wide).
* The dense 60 001-point resampled file of `compute_c_nlm.ipynb` is not produced; make it offline from
  `c_nlm.h5` with the toolkit.

---

## `reproducibility.uldm`

JSON written by `evolve` into the run folder before the first step. 30.40 keys (unchanged):

| Block | Content |
|---|---|
| `Integrator Version` | `S_version` |
| `Axion Mass`, `m22` | axion mass in kg (`eV * 1.78266191e-36`) and in units of 1e-22 eV |
| `Integration Modifiers` | method, `UseDispSponge`, `EdgeClear`, `Shift`, `IsoP` |
| `Built-in I/O Features` | `UseInit` (flag, path, blend), `DumpFinal`, `DumpInit` |
| `Core System Modifiers` | `SelfGravity`, `NBodyInterp`, mutual gravitation (always true), `NBodyGravity` |
| `Manual Dynamical Friction` | `ManualDF`, `DF_A`, `df_use_flow` |
| `Stopping Conditions` | `AutoStop`, `AutoStop3`, `KEThreshold`, `AutoStop2`, `WellThreshold` |
| `Misc. Settings` | draft quality, `CenterCalc`, `DR`, `CorrectDrift` |

Added in 30.50 (appended; old files still load):

| Block | Content |
|---|---|
| `SK Grid and Output Settings` | `CellCenteredGrid`, `ExtPhi` (used, kind, value, shape), `ComputeQuad`, `CorrectFreq`, `MassChange` (enabled, function name), `Resampled Box` (`resolR`, `Length_Ratio`, `LegacyResampleBox`, side and spacing in code units and kpc), simulation grid spacing (code units and kpc) |
| `Run Environment` | `S_version`, `D_version`, core file path, sha256 of the running core file as on disk and LF-normalised, CPU threads and their source, FFT backend, Python / platform / NumPy / SciPy / numexpr / pyFFTW / numba / h5py versions |

Runs made with 26.32, 30.38 or 30.40 have only the first table (26.32 also lacks `Manual Dynamical
Friction` and part of `Misc. Settings`); their grid layout, resampled-box geometry and core file hash are
not recorded.

---

## Known issues and conventions

Bugs fixed in 30.50 that matter for results made with older versions (including the effect of the
30.38/30.40 resampled cube on the paper's Figs. 2 and 12) are documented in
[docs/known_issues_and_errata.md](docs/known_issues_and_errata.md). Conventions that 30.50 keeps:

* **DF (`DYF`) sign.** With `NBodyInterp=True` (the SK `FWNBody`) the `DF` output stores **+grad Phi_SP**
  at the particle, the opposite sign of upstream PyUL and of `FWNBody_NI` (`NBodyInterp=False`), which store
  the acceleration `-grad Phi_SP`. Kept for comparability with existing outputs. `DYF_#000` is always zeros,
  and the gradient is set to 0 when a particle is in the first cell or the last four cells along any axis.
* **Manual DF radius.** `|r|` in the manual dynamical-friction term is measured from the **box origin**, not
  from the soliton centre or the ULDM COM.
* **Resampled cube of 30.38 - 30.40.** On cell-centred grids those versions resampled onto `(L - dx) *
  Length_Ratio` while downstream a_lm scripts assumed `L * Length_Ratio` (radial mislabel of 0.78 %,
  0.52 %, 0.39 % at N = 128, 192, 256). Fixed by default in 30.50 (C1); `LegacyResampleBox=True`
  reproduces it. Effect on the paper's figures:
  [errata](docs/known_issues_and_errata.md#e1-resampled-cube-side-in-3038-and-3040). With the legacy cube
  and `Length_Ratio = 1` the first resampled plane coincides with the first grid plane, so COM round-off
  moves those points in or out of the interpolation domain (index-0 planes only).
* **External potential energy.** `ExtPhi` is added to `Phi_SP`, so `calculate_energies` counts its energy
  with the self-gravity factor 0.5 (an external potential should have factor 1).
* **AutoStop and `DumpFinal`.** After an `AutoStop`/`AutoStop2` break on a non-save step, `Final_psi.npy`
  lacks the last half kick (half a step behind in phase).
* **`DumpInit`** dumps and continues (since 30.40; upstream 30.37 stopped).
* **Soliton profile generator.** The generator of embedded (BH + soliton) profiles (`BHRatioTester`,
  `SolitonProfile`) is not optimised for axion masses other than 1e-22 eV or BH-to-soliton mass ratios
  above 15 % (upstream note, still valid). Generated profiles are cached in
  `Soliton Profile Files/Custom` (or `PYUL_CACHE_DIR`).
* **Standard soliton mass constant.** Standard solitons use `alpha = (M / 3.8827652755822006)^2`; the
  toolkit's bundled `soliton.h5` stores the rounded `mass0 = 3.883` (see the toolkit README).
* **x / y in the generative helpers (upstream note, not re-verified).** The upstream README warned that the
  included generative functions may swap x and y; preview the grid with `PyUL.VizInit2D(...)` before long
  runs. `1Density` is the line along y, and the uniform-background velocity is printed as `(y, x, z)`.
* **Inactive options** (warned since 30.50): save flags `3UMmt`, `2UMmt`, `1UMmt`, `3GravRS`, `3GravRSF`;
  `Quadrupole` without `ComputeQuad=True`; `*RS` flags without `CenterCalc=True`; `Simpson`, `NLM`,
  `AutoStop3`, `KEThreshold`. `2EnergyTot`/`2EnergyKQ`/`2EnergySI` are written only together with `Energy`.
* **Legacy prototypes.** `NLMCompile`, `SphBasic`, `ExpansionCoefficientPrep` and `GalShift` are old
  prototypes that do not work; use `Cnlm=True` or the toolkit.
* **Logging volume.** With `CenterCalc=True` the COM is appended to the log file at every step.

---

## Tests

```bash
python -m pip install -e ".[test]"
python -m pytest
```

The suite (about 1.5 min) covers fresh-process imports (also with IPython, numba, pyFFTW and h5py blocked
and with cp1252/GBK/ASCII stdout), the compatibility shims, thread and path resolution, non-interactive
safety, tiny 32^3 runs (other working directory, `~` paths, no `Energy` flag, `ComputeQuad`, `AutoStop2`,
`MassChange`, momentum saves, IsoP cache), pyFFTW vs scipy.fft, and the inline c_nlm equivalence
(`tests/test_cnlm_inline.py`, skipped without the toolkit or pyshtools). Results on 2026-10-18 without the
baselines: 77 passed, 3 skipped (Python 3.9), 76 passed, 4 skipped (Python 3.13, and Python 3.14 without
IPython); with the baselines (below) nothing is skipped on Python 3.9. The suite also runs from an
extracted sdist (`python -m pytest` in its top folder).

`tests/test_regression.py` compares with 64^3 runs of the untouched 30.40 code when `PYUL_BASELINE_DIR`
points to a folder with `out64_default/` and `out64_isop/` (each a save_path with one run); with
`PYUL_REGRESSION_EXACT=1` every array must be bitwise identical (use the baseline environment), otherwise a
relative tolerance `PYUL_REGRESSION_RTOL` (default 1e-6) applies. The baselines are not shipped.

`Benchmark.py --resol 64 --yes --out <folder>` (or `Bench.sh`) times a short run and prints the machine,
thread count and FFT backend.

---

## Repository layout and legacy files

| Path | Content |
|---|---|
| `PyUltraLight2/Current_SK.py` | the integrator (imported by `import PyUltraLight2`) |
| `PyUltraLight2/_compat.py` | portability layer (optional dependencies, threads, paths, console) |
| `PyUltraLight2/Previous.py` | upstream PyUL 30.37, kept for reference; **not imported** |
| `PyUltraLight.py` | the original PyUltraLight (version 1); **not imported** |
| `examples/` | laptop driver and PBS / Slurm templates |
| `tests/` | pytest suite |
| `Soliton Profile Files/` | `initial_f.npy` (standard soliton profile) and `Custom/` (generated embedded-soliton profiles) |
| `Green Functions/` | `G96.npy` and the cache of generated IsoP Green functions |
| `Reusable/Configurations/` | two old example run configurations (format 21.3; still loadable) |
| `PyUL2_notebook_*.ipynb` | upstream notebooks (`Main`, `Examples`, `Batch`, `Batch_MR`, `SingleBHSoliton`, `3YT`) with stale outputs; they predate the SK features, do not use them, and use the old `k*T/(save_number+1)` time axis |
| `Benchmark.py`, `Bench.sh` | benchmark |
| `AlgorithmMap_APS.jpeg`, `PyULLogo.png`, `Done.mp3` | upstream algorithm map, logo and notebook jingle |
| `docs/known_issues_and_errata.md` | bugs of older versions and their effect on the paper |
| `CHANGELOG.md`, `CITATION.cff`, `.zenodo.json` | change log, citation metadata, Zenodo release metadata |
| `LICENSE`, `NOTICE` | licence; origin of the code and the upstream copyright notice |

---

## Licence

The SK fork is distributed under the MIT licence in [LICENSE](LICENSE). It derives from PyUltraLight
(Edwards, Kendall, Hotchkiss & Easther) and PyUltraLight 2 / PyUL_NBody (Y. F. Wang and collaborators,
[github.com/Sifyrena/PyUL_NBody](https://github.com/Sifyrena/PyUL_NBody)). An earlier version of the
upstream repository carried a BSD-style notice (copyright 2018 erckendall, modifications copyright 2020
Yourong Wang), which upstream removed in 2021; [NOTICE](NOTICE) reproduces it verbatim from this
repository's git history and describes the lineage. Please keep LICENSE, NOTICE and the upstream credits
when you redistribute or modify the code.
