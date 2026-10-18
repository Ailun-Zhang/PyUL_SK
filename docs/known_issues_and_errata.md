# Known issues and errata

This page lists the bugs fixed in PyUL_SK 30.50 that can matter to anyone who used an older version
(the SK pre-releases 26.32 and 30.38, the SK release 30.40, or upstream PyUL), says which outputs they
touch, and states their effect on the published results of

> A. Zhang, Y. Wang, J. L. Zagorac and R. Easther, *Stone Skipping Black Holes in Ultralight Dark Matter
> Solitons*, [arXiv:2602.11512](https://arxiv.org/abs/2602.11512) (2026).

The paper itself is not changed; this page is where the effects are documented. Figure numbers are the
printed figure numbers of the paper (13 figures; Figs. 6 and 7 are two separate single-mode figures). Each
figure is also named by its content.

The full list of changes, with IDs (C1, P1 - P32, D1 - D7, R9-xx), is in [CHANGELOG.md](../CHANGELOG.md).
Conventions that are unchanged are summarised at the end of this page and in the
[README](../README.md#known-issues-and-conventions).

## Contents

1. [Summary](#summary)
2. [E1: resampled-cube side in 30.38 and 30.40](#e1-resampled-cube-side-in-3038-and-3040)
3. [Other fixed bugs that matter for older versions](#other-fixed-bugs-that-matter-for-older-versions)
4. [Conventions (behaviour unchanged)](#conventions-behaviour-unchanged)
5. [Which PyUL version made which figure](#which-pyul-version-made-which-figure)

## Summary

| Issue | Versions | Outputs affected | Paper |
|---|---|---|---|
| **E1** resampled cube is `(L - dx) * Length_Ratio` instead of `L * Length_Ratio` on cell-centred grids (C1) | 30.38, 30.40 | `3WfnRS`, `3DensityRS`, `2DensityRS` and everything computed from them (`a_lm`, `c_nlm`, `P_dip`) | Fig. 2 (`P_dip` of all three resolutions) and Fig. 12: amplitudes up to 2.4 % too high; frequencies and conclusions unchanged |
| psi gains a factor `i` at every momentum save (P12) | all before 30.50 | saved phases of psi and of anything computed from them | none (no paper run saved momenta) |
| pair force repulsive when the Plummer smoothing is off, `a >= 1e8` (P13) | all before 30.50 | dynamics of runs with two or more particles | none (all paper runs used `a < 1e8`) |
| `MassChange` gives every particle the last particle's mass (P14) | SK versions before 30.50 | dynamics with `MassChange=True` | none (`MassChange=False`) |
| `ComputeQuad` uses a stale centre of mass without `CenterCalc` (P9) | SK versions before 30.50 | quadrupole output | none (`ComputeQuad=False`) |
| crashes: no `Energy` flag (P8), `AutoStop2` without `NBodyGravity` (P10), unsupported `NS` (P11), `NBodyEnergy` with `a != 0` (P16) | all before 30.50 | none (the run or call stops) | none |
| post-processing helpers: `convert_back(v, 'c', 'v')` inverted (P15), `NBodyEnergy` uses x and y only (P16), `InterpolateCurve3` builds z from y (P17) | all before 30.50 | values returned by those helpers | none (not used by the paper's drivers or figure scripts) |
| `UseInit=True` with `InitWeight` outside [0, 1] silently discards the loaded wavefunction (P22) | all before 30.50 | initial state | none (`InitWeight = 1`) |
| `Silent=True` / `NoInteraction=True` delete existing outputs without asking (P6, R9-01) | all before 30.50 | your files | not a numerical issue |

## E1: resampled-cube side in 30.38 and 30.40

### What

With `CenterCalc=True`, `Resample3Box` interpolates the wavefunction at every save onto a cube of
`resolR^3` points centred on the centre of mass (`3WfnRS`; also `3DensityRS` and `2DensityRS`). Every
downstream eigenmode analysis (the historical a_lm scripts and the
[ULDM-Eigenmode-Toolkit](https://github.com/Ailun-Zhang/ULDM-Eigenmode-Toolkit)) assumes that this cube has
side `L * Length_Ratio`, that is a spacing of `L * Length_Ratio / resolR`.

30.38 and 30.40 computed the side as `-2 * gridvec[0] * Length_Ratio`. On a cell-centred grid
(`CellCenteredGrid=True`, used by the paper runs) `gridvec[0] = -(L - dx)/2`, so the cube was
`(L - dx) * Length_Ratio`: shorter by one simulation cell. An analysis that labels the points with the
assumed spacing therefore places every point at a radius that is too large by the factor `N/(N - 1)`,
where `N` is the simulation resolution:

| N | 64 | 128 | 192 | 256 |
|---|---|---|---|---|
| cube shorter by `dx/L = 1/N` | 1.56 % | 0.78 % | 0.52 % | 0.39 % |
| radii too large by `1/(N - 1)` | 1.59 % | 0.79 % | 0.52 % | 0.39 % |

The soliton then looks slightly wider than it is, so its projection onto the eigenmodes is biased: the
ground-state coefficient `|c_000|^2/M` comes out above 1 at t = 0, and part of the ground state leaks into
the radial modes (`n >= 1`, `l = 0`).

### Which versions and outputs

* **Affected:** SK 30.38 and 30.40 with `CellCenteredGrid=True` and `CenterCalc=True`, in the outputs
  `3WfnRS`, `3DensityRS` and `2DensityRS`, and in all quantities computed from them (`a_lm`, `c_nlm`, the
  dipole power `P_dip`).
* **Not affected:** the dynamics and every other output (`NBody`, `DF`, `ULDCOM`, energies, `ULDMass`,
  `3Wfn`, `Final_psi`, ...). With `CellCenteredGrid=False` both formulas give the same cube. The 26.32
  lineage used `L * Length_Ratio` and is correct. Upstream PyUL has no cell-centred grid.
* **30.50** uses `L * Length_Ratio` by default (CHANGELOG **C1**). `evolve(..., LegacyResampleBox=True)`
  reproduces the 30.38 - 30.40 cube bit for bit. The inline decomposition `evolve(Cnlm=True)` uses the same
  cube as `3WfnRS`, so it follows the same switch. Since 30.50 the side and spacing of the resampled cube
  are written to `reproducibility.uldm` (`"SK Grid and Output Settings"`, `"Resampled Box"`).
* **How to recognise an affected `c_nlm` file:** for a single soliton the t = 0 value of `|c_000|^2/M` is
  0.994 - 1.000 in the paper's 26.32 runs, and 1.0191, 1.0132 and 1.0096 in the 30.40 run at 128^3, the
  30.40 run at 192^3 and the 30.38 run at 256^3. These three values are what the `1/(N - 1)` error
  predicts. (30.38 does not record `CellCenteredGrid` in `reproducibility.uldm`; for the 256^3 run the
  t = 0 value is the evidence that it is affected.)

### Quantified effect on the paper runs

The three affected runs were decomposed again from their saved `3WfnRS` cubes with the true spacing
`(L - dx) * Length_Ratio / resolR` and otherwise identical settings (toolkit a_lm kernel, `l_max = 10`,
`num_points = 300`, legacy angular grid), at save indices 0, 25 and 50:

| Run (printed figure) | PyUL | change of `P_dip` | change of `\|c_000\|^2/M` | `\|c_000\|^2/M` at t = 0, historical -> true |
|---|---|---|---|---|
| 128^3 (Fig. 2) | 30.40 | -1.6 % to -2.4 % | -2.3 % to -2.4 % | 1.019 -> 0.995 |
| 192^3 (Fig. 2) | 30.40 | -0.9 % to -1.6 % | -1.6 % | 1.013 -> 0.997 |
| 256^3 (Fig. 2, and the 30 000 Myr run of Fig. 12) | 30.38 | -0.6 % to -1.2 % | -1.2 % | 1.010 -> 0.998 |

(Change = true spacing relative to the historical values.) Over the three runs, `P_dip` is lower by
0.6 - 2.4 % and `|c_000|^2/M` by 1.2 - 2.4 % with the true spacing, most at 128^3. The corrected t = 0
values agree with those of the 26.32 runs.

**Fig. 2** (resolution comparison at 128^3, 192^3 and 256^3). The `P_dip` curves of all three resolutions
are 0.6 - 2.4 % too high, most at 128^3. The orbital radius `r_sys` comes from the N-body output and is not
affected. The spread between resolutions shrinks slightly (the 128^3 to 256^3 difference in `P_dip` at
1500 Myr goes from about 13.8 % to about 12.7 %); the conclusion drawn from the figure is unchanged.

**Fig. 12** (spectral analysis of `c_{1,1,-1}(t)` of the 30 000 Myr, 256^3 run). The error is a fixed
rescaling of the radius, the same at every time, so it changes amplitudes but not frequencies: the
frequencies in Fig. 12 are unaffected. The amplitude `|c_{1,1,-1}|^2` changes by -0.06 % and -1.0 % at
save indices 25 and 50 (and by +3.4 % at t = 0, where the mode is about a thousand times weaker than
later). The qualitative conclusions are unchanged.

**Not affected:** Figs. 5, 6 and 7 (the `|c_nlm(t)|^2` panels, the dipole `|c_{1,1,-1}|^2` and the radial
`|c_{1,0,0}|^2` of the single-BH run) come from 26.32 runs, whose resampled cube is correct. The orbit
figures (Figs. 1, 4, 8, 9, 10 and 11) use the N-body output, which never depends on the resampled cube.
Figs. 3 and 13 are not simulation outputs.

**Radial modes in the deposited `c_nlm` files of the three runs.** No printed figure shows them, but they
are the most sensitive to E1, because the leaked ground state adds to them. `|c_100|^2/M` at t = 0 drops
from 7.5e-5, 2.9e-5 and 1.6e-5 (historical; 128^3, 192^3, 256^3) to 2.5e-6, 4.9e-7 and 2.9e-7 with the
true spacing; at save indices 25 and 50 it rises by 22 - 32 % (128^3), 13 - 20 % (192^3) and 10 - 15 %
(256^3). Recompute these coefficients with the true spacing before using them.

### How to correct or reproduce

* New runs with 30.50: nothing to do (`LegacyResampleBox=False` is the default).
* To reproduce a 30.38/30.40 run exactly: `evolve(..., LegacyResampleBox=True)`.
* To analyse an existing 30.38/30.40 `3WfnRS` cube from a cell-centred run correctly, use the true spacing
  `dx_RS = (L - L/N) * Length_Ratio / resolR` (in the length unit the analysis expects) instead of
  `L * Length_Ratio / resolR`, for example as the `dx` argument of the toolkit's `compute_alm_cube`.

## Other fixed bugs that matter for older versions

None of these affects the paper: according to the run records (`reproducibility.uldm`, save options) and
the run drivers, the paper runs did not use the options involved. "Inherited" means the bug is also in
upstream PyUL 30.37.

* **psi multiplied by `i` at every momentum save (P12; inherited).** `pEval` and `LpEval` multiplied the
  caller's Fourier-space wavefunction by `i` in place. With the `Momentum`, `AngMomentum` or `2Momentum` save
  flags, psi therefore gained a global factor `i` at every momentum save. Densities, potentials and the
  dynamics do not depend on a global phase, but every saved phase (`3Wfn`, `2Wfn`, `Final_psi`, the
  resampled `3WfnRS`) jumps by pi/2 at each momentum save, and so do the phases of any `c_nlm(t)` computed
  from them (which would corrupt a spectral analysis such as that of Fig. 12). In addition, index 0 of the
  `Momentum`/`AngMomentum` outputs had the wrong sign when `2Momentum` was also on. *Not used by the paper
  runs* (no paper run saved momenta).
* **Repulsive pair force without smoothing (P13; inherited).** `evolve` sets the Plummer parameter `a = 0`
  when `a >= 1e8` ("smoothing turned off"), and in that branch the particle-particle force had the wrong
  sign (repulsive) in `FWNBody`, `FWNBody_NI` and `FWNBodySimple`. Runs with two or more particles and
  `a >= 1e8` are wrong. *Not used by the paper runs* (all used `a < 1e8`, including the binaries).
* **`MassChange` masses (P14).** With `MassChange=True` every particle got the mass of the last particle (a
  stale loop variable), and the mass unit conversion was skipped with `NBodyGravity=False`. *Not used by the
  paper runs.*
* **`ComputeQuad` (P9).** `ComputeQuad=True` crashed (missing `Outputs/Quadrupole`) unless the `Quadrupole`
  save flag created the folder, and without `CenterCalc` it used a stale centre of mass. *Not used by the
  paper runs.*
* **`NameError` without the `Energy` save flag (P8; inherited).** Every run without `Energy` crashed at the
  first save (`egpcmMlist`). A crash, so no completed run is affected.
* **Other crashes (P10, P11).** `AutoStop2=True` with `NBodyGravity=False` raised `NameError: phiRef`;
  `NS` values other than 0, 1 or a multiple of 4 returned `None`; `NS = 0` wrote a 6N-long DF log instead
  of 3N. The paper runs used `AutoStop2=False` and `NS = 32`.
* **Helpers (P15, P16, P17).** `convert_back(v, 'c', 'v')` computed `v * time_unit / length_unit / c`
  instead of the inverse of `convert`; `NBodyEnergy` raised `NameError` for any Plummer `a != 0` and used
  only the x and y components of the separations (inherited); `InterpolateCurve3` built the z spline from
  y. *Not used by the paper's drivers or figure scripts.*
* **`InitWeight` (P22).** `UseInit=True` with `InitWeight` outside [0, 1] (and not -1) silently discarded
  the loaded wavefunction. The paper runs used `InitWeight = 1`.
* **Deleted outputs (P6, R9-01).** `evolve(Silent=True)` deleted an existing `Outputs/` folder and
  `GenerateConfig(NoInteraction=True)` an existing run folder without asking, and several `input()` prompts
  failed in batch jobs. 30.50 never deletes in a non-interactive session without `OverwriteOutputs=True` /
  `Overwrite=True`, and refuses dangerous targets (home folder, working directory, repository root).
* **Imports on current libraries (P1, P2).** Versions before 30.50 do not import with SciPy >= 1.17
  (`sph_harm`) or IPython 9, and `DefaultSolitonOrbit` fails with SciPy >= 1.14 (`simps`). 30.50 runs on
  Python 3.9 - 3.14 and reproduces 30.40 bit for bit with the paper's software stack.

## Conventions (behaviour unchanged)

These are kept so that new outputs stay comparable with existing ones. Details are in the
[README](../README.md#known-issues-and-conventions) and in CHANGELOG D1 - D7.

* **DF sign (D1).** With `NBodyInterp=True` (the SK `FWNBody`) the `DF` output stores `+grad Phi_SP` at the
  particle, the opposite sign of upstream PyUL and of `FWNBody_NI`. `DYF_#000` is always zeros.
* **Manual dynamical friction (D4).** `|r|` in the manual DF term is measured from the box origin, not from
  the soliton centre or the ULDM centre of mass.
* **`ExtPhi` energy (D3)** is counted with the self-gravity factor 0.5.
* **`DumpFinal` after an AutoStop break (D2)** is half a step behind in phase.
* **Snapshot times.** Save `k` (k = 0 ... `save_number`) is at `t_k = start_time + k * duration /
  save_number`; since 30.50 these times are written to `Outputs/snapshot_times_code.npy` and
  `Outputs/snapshot_times_Myr.npy`. Some older analysis notebooks (including the upstream notebooks in this
  repository) used `k * duration / (save_number + 1)`, which compresses the time axis by the factor
  `save_number / (save_number + 1)` (0.2 % for 500 saves: the last point is drawn at 2994 Myr instead of
  3000 Myr). Printed Figs. 4, 8 and 9, and the digitised source curves of Figs. 10 and 11, were drawn with
  that time axis; Figs. 1, 2, 5, 6, 7 and 12 use the save times above. The toolkit's
  `trajectories.load_run(legacy_time_axis=True)` reproduces the old axis.
* **Eigenmode-analysis conventions** (toolkit; the same for every `c_nlm` in the paper; frequencies are
  unaffected): the legacy angular grid includes both endpoints (amplitude errors of 0.1 - 0.4 % for
  `l >= 1`, plus some leakage between `(l, m)`), and the legacy eigen solver puts the outer wall at
  `r_max + h`. Both are documented, with opt-in alternatives, in the
  [toolkit README](https://github.com/Ailun-Zhang/ULDM-Eigenmode-Toolkit#conventions-and-switches).

## Which PyUL version made which figure

From the `"Integrator Version"` recorded in each run's `reproducibility.uldm`:

| PyUL | Printed figures | E1 |
|---|---|---|
| 30.40 (git commit `ccab161`) | Fig. 1 (five of the six mass fractions; orbits only); Fig. 2 (128^3 and 192^3 runs) | affects the `P_dip` of Fig. 2 |
| 30.38 (pre-release, not published) | Fig. 2 (256^3 run), which is also the 30 000 Myr run of Fig. 12 | affects both |
| 26.32 (pre-release SK lineage, not published) | Fig. 1 (2 % case) and the runs of Figs. 4 - 11 | correct cube |

Figs. 3 and 13 are not simulation outputs. The 26.32 and 30.38 core files are not published. 30.50
reproduces 30.40 bit for bit with default options (plus `LegacyResampleBox=True` for the resampled
outputs); see [Reproducing 30.40 behaviour exactly](../README.md#reproducing-3040-behaviour-exactly).
