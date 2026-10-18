# Changelog

All notable changes to the PyUltraLight 2 SK fork (`PyUltraLight2/Current_SK.py`) are
documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow the upstream PyUL `S_version` scheme (30.40 = `2.30.40`).

Fix policies:

* **A**: fixed outright (crashes, wrong results in code paths the paper did not use,
  unit bugs in helpers, portability, unsafe interactive/destructive behaviour, metadata gaps).
* **B**: legacy default kept, opt-in switch added.
* **C**: fixed by default, legacy switch kept for bit-for-bit reproduction.

30.50 is a maintenance release of **30.40** (git blob `fbcc686`, file sha256
`c112e0221aaefeb536ef1c39a56b81997f274d28731319beb639db08144ffee3`). 30.40 is the core of part
of the Stone Skipping paper runs (five of the six Fig. 1 masses and the N = 128 and N = 192
resolution runs of Fig. 2); the other paper runs used 30.38 and 26.32 (see the versions table
in the README and the provenance in the data deposit). With default flags and
the pyFFTW backend, 30.50 reproduces 30.40 bit for bit on the 64^3 regression runs (periodic
and isolated potential) in the reference environment (Python 3.9.25, NumPy 1.26.4,
SciPy 1.13.1, pyFFTW 0.14). The only default-output change is the resampled box (C1), which
is reproduced exactly with `LegacyResampleBox=True`.

## [30.50] - 2026-10-18 - SK maintenance release

### Added

- **R1** `pyproject.toml`: `pip install -e .` installs the package `PyUltraLight2` (Python >= 3.9).
  Core dependencies are the modules imported unconditionally (numpy, scipy, numexpr,
  matplotlib); `pyFFTW`, `h5py`, `numba`, `ipython`, `psutil` are extras (`fftw`, `hdf5`,
  `numba`, `notebook`, `bench`, `all`) because each now has a fallback or is only needed by
  one feature. Data folders stay in the repository (editable install or `PYUL_DATA_DIR`). (A)
- **R2** `requirements.txt`, `requirements-full.txt`, `requirements-test.txt`,
  `environment.yml` (conda-forge), valid for Python 3.9 - 3.14. (A)
- **R3** `tests/` (pytest): fresh-process imports (also with IPython, numba, pyFFTW and h5py
  blocked, and with cp1252/GBK/ASCII stdout), compat shims, thread resolution, path
  resolution, non-interactive safety, tiny 32^3 runs (from another working directory, with
  `~` paths, without the Energy flag, ComputeQuad, AutoStop2, MassChange, momentum saves,
  IsoP cache), FFT-backend equivalence and an optional regression against 30.40 baselines
  (`PYUL_BASELINE_DIR`, `PYUL_REGRESSION_EXACT=1`). (A)
- **R4** `examples/run_stone_skipping_local.py` (argparse, laptop defaults, paper recipe),
  `examples/pbs/stone_skipping.pbs`, `examples/slurm/stone_skipping.sbatch`, `examples/README.md`. (A)
- **R6** `CITATION.cff` rewritten for the SK fork (valid CFF 1.2.0); upstream PyUltraLight 2
  (Zenodo 10.5281/zenodo.6336160) and PyUltraLight (JCAP 2018) kept as `references`. (A)
- **R8** this `CHANGELOG.md`.
- `PyUltraLight2/_compat.py`: small helper module for all portability shims below.
- **P25** `reproducibility.uldm` gains two blocks (all existing keys unchanged):
  `"SK Grid and Output Settings"` (CellCenteredGrid, ExtPhi use, ComputeQuad, CorrectFreq,
  MassChange, resampled box resolR / Length_Ratio / legacy flag / side and spacing in code
  units and kpc, grid spacing) and `"Run Environment"` (S_version, D_version, core file path,
  sha256 of the running core file as on disk and LF-normalised, thread count and its source,
  FFT backend, Python/NumPy/SciPy/numexpr/pyFFTW/numba/h5py versions). `config.uldm` and
  `LoadConfig` are unchanged (old and new files load). (A)
- **P26** `Outputs/snapshot_times_code.npy` and `Outputs/snapshot_times_Myr.npy`:
  `t_k = start_time + k*duration/save_number` for the indices actually written, rewritten at
  every save (complete after an early AutoStop or a crash). (A)
- **P31** Optional FFT backend: pyFFTW stays the default when importable;
  `PYUL_FFT_BACKEND=scipy` (or a missing pyFFTW) uses a `scipy.fft` shim with the
  `pyfftw.builders` interface used at every call site. Equivalence on the 64^3 runs:
  <= 1.1e-11 relative for psi, N-body, DF, COM and energies. (A)
- New `evolve` keywords (appended; the defaults keep the 30.40 numbers except the resampled
  outputs on cell-centred grids, C1, which `LegacyResampleBox=True` reproduces bit for bit):
  `LegacyResampleBox=False` (C1), `OverwriteOutputs=False` (P6). New `GenerateConfig` keyword `Overwrite=False` (P6).
  New `Resample3Box` keyword `BoxLength=None` (C1). New `makeDCTGreen(threads=None)`,
  `isolatedPotentialSP(..., green_planes=None)`, `buildGreenPlaneCache(green)` (P4, P32).
- Environment variables: `PYUL_NUM_THREADS`, `PYUL_FFT_BACKEND`, `PYUL_PROGRESS`,
  `PYUL_INTERACTIVE`, `PYUL_DATA_DIR`, `PYUL_CACHE_DIR` (all optional).
- **I1** Inline c_nlm decomposition, `evolve(Cnlm=True)` (default `False`; with it off nothing is
  imported and every output is bit-for-bit as before). At every save, including the initial state,
  the resampled cube is taken exactly as `Resample3Box` makes it for `3WfnRS` (`Length_Ratio`,
  `resolR`, `LegacyResampleBox`; centred on the CenterCalc COM, or, with `CenterCalc=False`, on the
  COM of the current density as for the `ULDCOM` output), without writing 3WfnRS, and decomposed
  with the ULDM-Eigenmode-Toolkit's one-pass kernel and per-snapshot projector
  (`Eig_ULDM_packages.cnlm_inline`, toolkit >= 0.3.0; needs pyshtools and h5py). At the end (also
  after AutoStop/AutoStop2) `Outputs/c_nlm.h5` is written in exactly the toolkit's layout (root `t`
  in Myr = the PyUL save times of `snapshot_times_Myr.npy`, `n/<n>/l/<l>/m_<m>` complex128;
  provenance as root attributes only). Per-save checkpoints in `Outputs/Cnlm/alm_checkpoints`
  (toolkit pipeline format) make a crashed run recoverable with `uldm-cnlm RUN_DIR --from-inline`.
  New keywords (appended): `Cnlm=False`, `CnlmEigenFile=None` (None/'auto': built with the paper
  defaults from the run's single soliton), `CnlmLmax=10`, `CnlmNShells=257`, `CnlmRmax=None` (half
  the resampled box), `CnlmNumPoints=300`, `CnlmAngularGrid='legacy'`, `CnlmSaveAlm=False`,
  `CnlmOutput='c_nlm.h5'`, `CnlmThreads=None` (PyUL's thread count), `CnlmSolitonMass=None`,
  `CnlmEigenParams=None`. All settings are validated before any output is touched (fail fast);
  a failure after the time loop has started is reported as a warning and never stops or changes
  the simulation. Equivalence (tests/test_cnlm_inline.py): bitwise equal to the toolkit pipeline
  run on the saved 3WfnRS cubes in the same process (48^3/64^3, CellCenteredGrid and
  LegacyResampleBox True/False, CenterCalc True/False) and to the historical JSON chain in the same
  process; <= 3.2e-14 relative per mode against the JSON chain in another process (py39: <= 5.9e-15). (A)
- **I2** Internal helpers for I1: `_SnapshotTimes` (the times `_SaveSnapshotTimes` writes, same
  arithmetic), `_CnlmHook`, `_CnlmCube`; `Resample3Box`'s return value is used at the save points.
  `pyproject.toml` extra `cnlm` (pyshtools, h5py). (A)

### Changed

- **C1** (policy C) `Resample3Box` on cell-centred grids: the resampled cube now has side
  `L*Length_Ratio` (spacing `L*Length_Ratio/resolR`), as in the 26.32 lineage and as every
  downstream a_lm script assumes. 30.38-30.40 used `-2*gridvec[0]*Length_Ratio`, which is
  `(L-dx)*Length_Ratio` when `CellCenteredGrid=True` (a 1/resol scale error: 1.6 %, 0.78 %,
  0.52 %, 0.39 % at resol 64, 128, 192, 256). `evolve(LegacyResampleBox=True)` reproduces
  30.40 bit for bit; with `CellCenteredGrid=False` both are identical. `Resample3Box` now
  returns the resampled cube `PsiNew` (hook for the inline c_nlm work). Affects only the
  3WfnRS / 3DensityRS / 2DensityRS outputs (CenterCalc=True), never the dynamics.
- **P1** Imports: `scipy.special.sph_harm` (removed in SciPy 1.17) replaced by a wrapper with
  the legacy call semantics `sph_harm(m, n, azimuth, polar)` (uses `sph_harm_y(n, m, polar,
  azimuth)` when available; equal to round-off, max |diff| 1.0e-14 for l <= 8, SciPy 1.13.1 vs
  1.18.1). IPython is optional (`clear_output` is a no-op without it; IPython 9 removed
  `IPython.core.display.clear_output`). numba is optional: it is only used to jit
  `initsoliton` (embedded solitons) and `LUQuick` (Momentum/AngMomentum saves); without it
  those run as plain Python and a one-line notice is printed at import. h5py is imported on
  first use (`save_format='hdf5'`) with a clear error if missing. pyFFTW is optional (P31).
  The "comment out line #22" message is gone. (A)
- **P2** `scipy.integrate.simps` (removed in SciPy 1.14) replaced by `simpson` with identical
  call semantics in `DefaultSolitonOrbit`, `BHRatioTester`, `SolitonProfile`.
  `DefaultSolitonOrbit` for the paper inputs is bitwise identical to 30.40 on SciPy 1.13.1
  and also on SciPy 1.18.1. (A)
- **P3** Console output never raises `UnicodeEncodeError`: `D_version` and the credits banner
  are plain ASCII; `printU`, the progress bar, `DSManagement` and the profile generator fall
  back to ASCII/replacement characters when the stream cannot encode them (cp1252, GBK,
  ASCII). Log files are written as UTF-8. (A)
- **P4** Thread count: `PYUL_NUM_THREADS` > `OMP_NUM_THREADS` > `SLURM_CPUS_PER_TASK` >
  `NCPUS` (PBS) > `len(os.sched_getaffinity(0))` > `os.cpu_count()`; grids with resol < 128
  are still capped at 4 (except an explicit `PYUL_NUM_THREADS`). Used for pyFFTW, numexpr and
  scipy.fft (also the IsoP Green-function DCT), printed and recorded. (A)
- **P5** Resource paths are independent of the working directory: data
  (`Soliton Profile Files/initial_f.npy`, custom profiles, HFK) are looked up in
  `PYUL_DATA_DIR`, then the working directory (legacy), then the repository root. IsoP Green
  functions are read from `PYUL_CACHE_DIR`, the data folders or the user cache, and new ones
  are written atomically (temporary file + `os.replace`, `makedirs(exist_ok=True)`) to
  `PYUL_CACHE_DIR`, else the existing `Green Functions/` folder of a source checkout if writable
  (never created, so a non-editable install does not write into site-packages; R9-04), else the
  user cache. Unreadable/truncated cache files are ignored. (A)
- **P6** Non-interactive safety (behaviour change). PyUL never calls `input()` when the session
  is not interactive (interactive = stdin is a TTY or a Jupyter kernel;
  `PYUL_INTERACTIVE=0|1` overrides) and never raises `EOFError`. Destructive branches need
  explicit consent in non-interactive sessions:
  `evolve` with an existing `Outputs/` raises `FileExistsError` unless `OverwriteOutputs=True`
  (`Silent=True` alone used to delete it); `GenerateConfig` with an existing run folder raises
  `FileExistsError` unless `Overwrite=True` (`NoInteraction=True` used to delete it, and
  `NoInteraction=False` called `input()`); `DSManagement(Force=False)` keeps a non-empty
  folder; `Runs` returns the latest run; `BHRatioTester` raises its "failed to converge"
  `ValueError` instead of prompting. Interactive behaviour is unchanged (EOF answers count as
  "no"). (A)
- **P7** Progress output: on a non-TTY stdout (PBS/Slurm logs) one compact ASCII line per
  snapshot instead of ~6 `\r` bar updates per step; `PYUL_PROGRESS=bar|lines|quiet`
  overrides. Computation unchanged. (A)
- **P12** `pEval`/`LpEval` no longer multiply the caller's Fourier-space wavefunction by `i`
  in place. With Momentum/AngMomentum/2Momentum saves, psi used to gain a global factor `i`
  at every momentum save (corrupting saved phases); now psi is identical with and without
  momentum saves. Momentum values are unchanged except index 0 of `Momentum`/`AngMomentum`
  when `2Momentum` is also enabled: 30.40 computed those from -F (an extra factor `i`, because
  `pEval` had already multiplied the shared array in place), and they are now correct. (A)
- **P13** Particle-particle force with Plummer `a == 0` (set when `a >= 1e8`) is now
  attractive `-1/r^3` like the Plummer branch (it was repulsive) in `FWNBody`, `FWNBody_NI`
  and `FWNBodySimple`. Runs with `a < 1e8` (all paper runs) are unaffected. (A)
- **P21** `Momentum`/`AngMomentum` saves honour `save_format` in the loop (index 0 already
  did); `2Momentum` no longer switches `Momentum`/`AngMomentum` off, and the momentum index
  advances once per momentum save point. (A)
- **P22** `UseInit=True` with `InitWeight` outside [0, 1] and not -1 raises `ValueError` (the
  loaded wavefunction was silently discarded). (A)
- **P23** Accepted-but-inactive options (save flags 3UMmt/2UMmt/1UMmt/3GravRS/3GravRSF,
  Quadrupole without ComputeQuad, *RS flags without CenterCalc, and the kwargs Simpson, NLM,
  AutoStop3, KEThreshold) now trigger one `UserWarning` listing them. (A)
- **P32** IsoP: the even-extended Green planes are built once per run instead of every step
  (bitwise identical; ~0.5 GB fewer allocations per step at resol 256). (A)
- `S_version = 30.50`; `D_version = 'October 2026, SK maintenance release 30.50 by Alan Zhang'`.
- **R5** `Bench.sh` is a bash script (it had a python shebang); `Benchmark.py` takes
  `--resol/--yes/--out/--axion-mass`, never prompts when not interactive, no longer tries to
  `pip install` packages (including the standard-library `multiprocessing`), and the
  momentum/COM-correcting soliton uses its intended mass `MF` (it used the last random mass).
- **R7** `.gitattributes`: `* text=auto eol=lf` plus binary patterns.
- **R10** `.gitignore`: Python/test caches, run-time Green-function and profile caches,
  example outputs; `tests/` is explicitly not ignored (the old `/Tests` rule matched it on
  case-insensitive file systems).

### Fixed

- **P8** Runs without the `Energy` save flag no longer crash with `NameError: egpcmMlist` at
  the first save (inherited from upstream). (A)
- **P9** `ComputeQuad=True` creates `Outputs/Quadrupole` itself (FileNotFoundError before) and
  uses the current COM at each save (it was stale without CenterCalc). (A)
- **P10** `AutoStop2=True` with `NBodyGravity=False` no longer raises `NameError: phiRef`. (A)
- **P11** Unsupported `NS` values (not 0, 1 or a multiple of 4) raise a clear `ValueError` at
  the start of `evolve` and in `NBodyAdvance(_NI)` (they returned None). `NS == 0` returns a
  3N-long DF log like every other NS (it was 6N). (A)
- **P14** `MassChange`: every particle uses its own mass (a stale loop variable gave all
  particles the last particle's mass), converted with `m_mass_unit`, also when
  `NBodyGravity=False`. (A)
- **P15** `convert_back(value, 'c', 'v')` is the inverse of `convert` (was
  `value*time_unit/length_unit/c`). (A)
- **P16** `NBodyEnergy`: uses the `length_units` argument (it passed the literal string and
  raised NameError for any Plummer `a != 0`) and 3D separations (it used x, y only). (A)
- **P17** `InterpolateCurve3` builds the z spline from z (it used y). (A)
- **P18** `DSManagement(Force=True)` works on non-empty folders (UnboundLocalError) and on
  missing folders (FileNotFoundError), and creates nested paths. (A)
- **P19** `Runs(Automatic=False)` with a blank answer loads the latest run (AttributeError
  on `int.startswith`). (A)
- **P20** `~` in `save_path` is expanded before any path is built (`evolve`, `GenerateConfig`,
  `DSManagement`, `Runs`). (A)

### Documented (behaviour unchanged)

- **D1** The SK `FWNBody` writes **+grad(Phi_SP)** to the DF (DYF) output, the opposite sign
  of upstream PyUL and of `FWNBody_NI` (which store the acceleration -grad Phi). `DYF_#000`
  is always zeros. Kept for comparability with existing outputs (policy B).
- **D2** After an AutoStop/AutoStop2 break on a non-save step, `DumpFinal` writes psi without
  the final half-kick (half a step behind in phase).
- **D3** `ExtPhi` is added to Phi_SP, so `calculate_energies` counts its energy with the
  self-gravity factor 0.5 (an external potential should have factor 1).
- **D4** Manual dynamical friction uses |r| measured from the box origin, not from the
  soliton centre or the ULDM COM.
- **D5** `DumpInit=True` no longer stops the run (since 30.40, unlike upstream 30.37).
- **D6** With the legacy resample cube and `Length_Ratio=1` on a cell-centred grid, the first
  resampled plane coincides with the first grid plane, so round-off in the COM moves those
  points in or out of the interpolation domain (value vs fill 0); differences between
  libraries/backends in 3WfnRS are confined to index-0 planes. With the new default cube
  (Length_Ratio=1, |COM| < dx/2) the first plane lies outside the grid and is always 0.
- **D7** 30.40's `LoadConfig` refuses a `config.uldm` whose `"PyUL Version"` is newer than its
  own, so it cannot load configurations written by 30.50 (`RuntimeError: Configuration file
  generated by a newer version of PyUL.`), although the format is unchanged. Workaround for a
  cross-check with the 30.40 file: set `"PyUL Version"` to `30.4` in a copy of the run folder, or
  generate the configuration with 30.40. 30.50 loads older configurations (README).

### Review fixes (R9, 2026-10-18)

Physics, default outputs and the 30.40 regression are unchanged by all of these.

- **R9-01** Recursive deletes of user paths refuse dangerous targets: `DSManagement` (`Force=True`
  or the interactive "Y"), `evolve`'s `<run>/Outputs`, `GenerateConfig`'s run folder (and its
  `Outputs` for the "C" answer) and `Runs`' `X<n>` answer raise `ValueError` instead of deleting
  when the folder resolves to the filesystem root, the home directory, the current working
  directory or one of its ancestors, or the repository root (`REPO_ROOT`) or one of its
  ancestors (`_compat.check_safe_rmtree`). Before, `DSManagement('~', Force=True)` or
  `DSManagement('.', Force=True)` deleted the home folder or the working directory. (A)
- **R9-02** `evolve(Cnlm=True)` with an existing `Outputs/c_nlm.h5` (or `a_lm.h5`) inside
  `<run>/Outputs` no longer raises `FileExistsError` before evolve's own decision on the existing
  `Outputs` folder, so an interactive session is asked `[Y/n]` again and interactive
  `Silent=True` replaces `Outputs` as without `Cnlm`. Non-interactive sessions still need
  `OverwriteOutputs=True` (the `Outputs` rule), and `CnlmOutput`/`CnlmSaveAlm` paths outside
  `Outputs` still need `OverwriteOutputs=True`. (A)
- **R9-03** `evolve(Cnlm=True, CnlmEigenFile=<file inside <run>/Outputs>)` (for example the
  automatic `Outputs/Cnlm/eigs_*.h5` of the previous run, or `uldm-cnlm --build-eigen`'s default
  `Outputs/cnlm_pipeline/`) is refused with a `ValueError` before `Outputs` is deleted; before, the
  run deleted the eigen file with `Outputs` and then died with `FileNotFoundError`. The check is
  in the toolkit's `CnlmInlineSettings.validate` (`outputs_replaced`, default True, which is what
  `evolve` does); the PyUL call site is unchanged and works with toolkits with and without it.
  Test: `tests/test_cnlm_inline.py::test_eigen_file_inside_outputs_refused_before_delete`. (A)
- **R9-04** IsoP Green-function cache: the repository's `Green Functions/` is used only when
  PyUL runs from a source checkout (`pyproject.toml` and the shipped `Green Functions/` next to the
  package) and is never created. A non-editable install (where `REPO_ROOT` is site-packages)
  now writes new `G{resol}.npy` files to `PYUL_CACHE_DIR` or the user cache; before, it created
  `site-packages/Green Functions/` (up to 136 MB per file, left behind by `pip uninstall`). (A)
- **R9-05** `pyproject.toml`: SPDX licence metadata (`license = "MIT"`,
  `license-files = ["LICENSE"]`, metadata 2.4 `License-Expression`) instead of the table form
  that setuptools deprecates (deadline 2027-02-18); `requires = ["setuptools>=77"]`, the first
  release with SPDX support, which is available for Python 3.9. Verified: `uv build` of sdist and wheel with
  setuptools 82.0.1 (Python 3.9) and 84.0.0 (Python 3.14) without warnings about the licence. (A)
- **R9-06** `MANIFEST.in`: the sdist is a minimal source checkout (package, `tests/` including
  `conftest.py`, `examples/`, `Soliton Profile Files/initial_f.npy`, `Green Functions/G96.npy`,
  `Reusable/Configurations/`, docs and requirement files), so the shipped test suite runs from the
  extracted sdist. The wheel still contains only the package. (A)
- **R9-07** Documentation corrections: P12 (momentum index 0 with `2Momentum`), the
  "defaults keep the 30.40 numbers" wording of the new `evolve` keywords (C1 exception) in this
  file and in the `evolve` docstring, the paper-run versions (30.40 is the core of part of the
  runs; 30.38 and 26.32 for the others), D7, and in the README the automatic eigen file (its
  `r`/`f`/`E` datasets, not its file hash, equal the paper's eigen file), the different shell
  defaults of the inline recorder (`CnlmNShells=257`) and the toolkit pipeline (`resolR + 1`),
  and the production cluster (described, not named).

## [30.40] - 2026-01-29 - SK release

Core of part of the Stone Skipping paper runs (five of the six Fig. 1 masses and the N = 128 and
N = 192 resolution runs of Fig. 2); the other paper runs used 30.38 and 26.32. See the versions
table in the README and the provenance in the data deposit.
