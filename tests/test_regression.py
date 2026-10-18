"""Regression against the 64^3 baselines produced by the untouched 30.40 code.

Enabled when PYUL_BASELINE_DIR points to a folder containing out64_default/ and
out64_isop/ (each a save_path with latest.uldm and one run folder), e.g. the
maintainers' tests/baseline_smoke folder.  The baselines were produced on
Python 3.9 / NumPy 1.26.4 / SciPy 1.13.1 / pyFFTW 0.14 with 4 FFT threads.

With PYUL_REGRESSION_EXACT=1 every array must be bitwise identical (use this
in the baseline environment); otherwise a relative tolerance is applied
(PYUL_REGRESSION_RTOL, default 1e-6 of the array maximum) because other
library versions change the round-off.
"""
import glob
import os

import numpy as np
import pytest

from conftest import tiny_run

BASE = os.environ.get("PYUL_BASELINE_DIR")
EXACT = os.environ.get("PYUL_REGRESSION_EXACT", "") == "1"
RTOL = float(os.environ.get("PYUL_REGRESSION_RTOL", "1e-6"))

pytestmark = [pytest.mark.skipif(not BASE, reason="PYUL_BASELINE_DIR not set"), pytest.mark.timeout(1800)]


def _rundir(save_path):
    with open(os.path.join(save_path, "latest.uldm")) as fh:
        return os.path.join(save_path, fh.read().strip())


def _compare(base_run, new_run, skip=()):
    files = sorted(os.path.relpath(p, base_run) for p in glob.glob(os.path.join(base_run, "**", "*.npy"), recursive=True))
    assert files
    bad = []
    for rel in files:
        if any(s in rel for s in skip):
            continue
        a = np.load(os.path.join(base_run, rel))
        b = np.load(os.path.join(new_run, rel))
        assert a.shape == b.shape and a.dtype == b.dtype, rel
        if EXACT:
            if a.tobytes() != b.tobytes():
                bad.append(rel)
        else:
            scale = np.max(np.abs(a)) if a.size else 0
            if "3WfnRS" in rel:
                # With the legacy cube (side (L-dx)*Length_Ratio, Length_Ratio=1) the first
                # resampled plane lies exactly on the first grid plane, so a round-off change
                # of the COM moves those points in or out of the interpolation domain
                # (value vs fill 0).  Compare the interior only.
                a, b = a[1:, 1:, 1:], b[1:, 1:, 1:]
            if a.size and np.max(np.abs(a - b)) > RTOL * scale:
                bad.append((rel, float(np.max(np.abs(a - b)) / scale)))
    assert not bad, bad


@pytest.mark.parametrize("mode", ["default", "isop"])
def test_baseline_64_legacy_box(tmp_path, mode):
    kw = dict(IsoP=(mode == "isop"), LegacyResampleBox=True)
    r = tiny_run(tmp_path / "out", evolve_kw=kw, resol=64, duration=300, saves=10, name="r64")
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-3000:]
    _compare(_rundir(os.path.join(BASE, "out64_" + mode)), os.path.join(str(tmp_path / "out"), "r64"))


def test_baseline_64_default_box_only_geometry_differs(tmp_path):
    r = tiny_run(tmp_path / "out", resol=64, duration=300, saves=10, name="r64")
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-3000:]
    new_run = os.path.join(str(tmp_path / "out"), "r64")
    base_run = _rundir(os.path.join(BASE, "out64_default"))
    _compare(base_run, new_run, skip=("3WfnRS",))
    # the new default resamples the same psi on a cube of side L (not L - dx)
    import PyUltraLight2 as PyUL
    L = PyUL.convert(5.0, 'kpc', 'l')
    dx = L / 64
    g = -L / 2 + dx / 2 + np.arange(64) * dx
    for k, tag in ((0, "Init"), (10, "Final")):
        psi = np.load(os.path.join(base_run, f"{tag}_psi.npy"))
        com = np.load(os.path.join(base_run, "Outputs", "ULDCOM", f"UCM_#{k:03d}.npy"))
        expect = PyUL.Resample3Box(psi, com, g, "", k, "npy", 1, 64, BoxLength=L)
        got = np.load(os.path.join(new_run, "Outputs", "3WfnRS", f"P3R_#{k:03d}.npy"))
        if EXACT:
            assert got.tobytes() == expect.tobytes()
        else:
            assert np.max(np.abs(got - expect)) <= RTOL * np.max(np.abs(expect))
