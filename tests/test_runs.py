"""Tiny end-to-end runs (32^3, a few snapshots) in fresh processes."""
import glob
import json
import os

import numpy as np
import pytest

from conftest import tiny_run

pytestmark = pytest.mark.timeout(900)


def _ok(r):
    assert r.returncode == 0, (r.stdout.decode("utf-8", "replace")[-3000:] + "\n---\n"
                               + r.stderr.decode("utf-8", "replace")[-3000:])
    return r.stdout.decode("utf-8", "replace")


def _rundir(save_path, name="tiny"):
    return os.path.join(str(save_path), name)


def test_tiny_run_from_other_cwd(tmp_path):
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    out = _ok(tiny_run(tmp_path / "out", cwd=str(cwd)))
    rd = _rundir(tmp_path / "out")
    assert "snapshot 005" in out                                # P7 line progress
    for flag, prefix in (("NBody", "NTM"), ("DF", "DYF"), ("ULDCOM", "UCM"), ("3WfnRS", "P3R")):
        files = sorted(glob.glob(os.path.join(rd, "Outputs", flag, prefix + "_#*.npy")))
        assert len(files) == 6, flag
    # P26 snapshot times
    tc = np.load(os.path.join(rd, "Outputs", "snapshot_times_code.npy"))
    tm = np.load(os.path.join(rd, "Outputs", "snapshot_times_Myr.npy"))
    assert tc.shape == (6,) and tc[0] == 0
    np.testing.assert_allclose(tm, np.linspace(0, 30, 6), rtol=1e-12, atol=1e-12)
    # P25 reproducibility metadata (old keys kept, new keys added)
    rep = json.load(open(os.path.join(rd, "reproducibility.uldm")))
    for k in ("Integrator Version", "Axion Mass", "m22", "Integration Modifiers", "Misc. Settings"):
        assert k in rep
    sk = rep["SK Grid and Output Settings"]
    assert sk["Cell Centered Grid"] is True
    rb = sk["Resampled Box"]
    assert rb["resolR"] == 32 and rb["Legacy Resample Box"] is False
    assert rb["Side (kpc)"] == pytest.approx(5.0, rel=1e-12)
    assert rb["Spacing (kpc)"] == pytest.approx(5.0 / 32, rel=1e-12)
    env = rep["Run Environment"]
    assert env["S_version"] == 30.5 and len(env["Core SHA256"]) == 64
    assert env["CPU Threads"] >= 1 and env["FFT Backend"] in ("pyfftw", "scipy")
    assert "NumPy" in env["Library Versions"]
    # nothing was written into the working directory
    assert os.listdir(str(cwd)) == []


def test_tilde_save_path(tmp_path):                    # P20
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    _ok(tiny_run("~/out", cwd=str(cwd), saves=2, duration=10,
                 env={"HOME": str(home), "USERPROFILE": str(home)}))
    assert os.path.isfile(os.path.join(str(home), "out", "tiny", "Outputs", "NBody", "NTM_#002.npy"))
    assert os.listdir(str(cwd)) == []                 # no literal '~' folder


def test_existing_outputs_noninteractive(tmp_path):
    _ok(tiny_run(tmp_path / "out"))
    rd = _rundir(tmp_path / "out")
    # GenerateConfig reuses the name 'tiny': the run folder exists -> refuses without Overwrite
    r = tiny_run(tmp_path / "out")
    assert r.returncode != 0
    assert b"FileExistsError" in r.stderr and b"Overwrite=True" in r.stderr
    assert os.path.isdir(os.path.join(rd, "Outputs", "NBody"))


def test_evolve_outputs_guard(tmp_path, PyUL):
    _ok(tiny_run(tmp_path / "out", saves=2, duration=10))
    rd = _rundir(tmp_path / "out")
    keep = os.path.join(rd, "Outputs", "keep.txt")
    open(keep, "w").write("x")
    with pytest.raises(FileExistsError, match="OverwriteOutputs"):
        PyUL.evolve(str(tmp_path / "out"), "tiny", Silent=True)    # Silent no longer deletes in batch
    assert os.path.exists(keep)
    PyUL.evolve(str(tmp_path / "out"), "tiny", OverwriteOutputs=True, CellCenteredGrid=True)
    assert not os.path.exists(keep)
    assert os.path.exists(os.path.join(rd, "Outputs", "snapshot_times_code.npy"))


def test_run_without_energy_flag(tmp_path):            # P8
    _ok(tiny_run(tmp_path / "out", save_options="NBody DF"))
    assert len(glob.glob(os.path.join(_rundir(tmp_path / "out"), "Outputs", "NBody", "*.npy"))) == 6


def test_compute_quad_without_flag(tmp_path):          # P9
    _ok(tiny_run(tmp_path / "out", save_options="Energy NBody", evolve_kw=dict(ComputeQuad=True, CenterCalc=False)))
    assert len(glob.glob(os.path.join(_rundir(tmp_path / "out"), "Outputs", "Quadrupole", "QIJ_*.npy"))) == 6


def test_autostop2_without_nbody_gravity(tmp_path):    # P10
    _ok(tiny_run(tmp_path / "out", evolve_kw=dict(AutoStop2=True, NBodyGravity=False)))


def test_invalid_ns_rejected(tmp_path):               # P11
    r = tiny_run(tmp_path / "out", ns=6)
    assert r.returncode != 0 and b"NS (RK Steps) = 6" in r.stderr


def test_invalid_initweight_rejected(tmp_path):       # P22
    r = tiny_run(tmp_path / "out", evolve_kw=dict(UseInit=True, InitPath="x", InitWeight=2))
    assert r.returncode != 0 and b"InitWeight" in r.stderr


def test_momentum_saves_do_not_change_psi(tmp_path):  # P12 / P21
    _ok(tiny_run(tmp_path / "a", save_options="Energy NBody", name="ref"))
    _ok(tiny_run(tmp_path / "b", save_options="Energy NBody Momentum AngMomentum 2Momentum", name="mom"))
    pa = np.load(os.path.join(str(tmp_path / "a"), "ref", "Final_psi.npy"))
    pb = np.load(os.path.join(str(tmp_path / "b"), "mom", "Final_psi.npy"))
    assert pa.tobytes() == pb.tobytes()
    out = os.path.join(str(tmp_path / "b"), "mom", "Outputs")
    for flag, prefix in (("Momentum", "PMT"), ("AngMomentum", "MVR"), ("2Momentum", "P2D")):
        assert len(glob.glob(os.path.join(out, flag, prefix + "_#*.npy"))) == 6, flag


def test_masschange_constant_function_matches(tmp_path):   # P14
    _ok(tiny_run(tmp_path / "a", save_options="Energy NBody", name="ref"))
    _ok(tiny_run(tmp_path / "b", save_options="Energy NBody", name="mc", env={"TINY_MASSFUNC_ONE": "1"}))
    a = np.load(os.path.join(str(tmp_path / "a"), "ref", "Outputs", "NBody", "NTM_#005.npy"))
    b = np.load(os.path.join(str(tmp_path / "b"), "mc", "Outputs", "NBody", "NTM_#005.npy"))
    assert a.tobytes() == b.tobytes()


def test_isop_green_cache_and_rerun(tmp_path):         # P5 / P32
    cache = tmp_path / "cache"
    env = {"PYUL_CACHE_DIR": str(cache)}
    _ok(tiny_run(tmp_path / "a", name="iso1", evolve_kw=dict(IsoP=True), env=env, saves=2, duration=10))
    g = cache / "Green Functions" / "G32.npy"
    assert g.exists()
    out = _ok(tiny_run(tmp_path / "b", name="iso2", evolve_kw=dict(IsoP=True), env=env, saves=2, duration=10))
    assert "Using pre-computed Green function" in out
    a = np.load(os.path.join(str(tmp_path / "a"), "iso1", "Final_psi.npy"))
    b = np.load(os.path.join(str(tmp_path / "b"), "iso2", "Final_psi.npy"))
    assert a.tobytes() == b.tobytes()


def test_fft_backend_equivalence(tmp_path):            # P31
    pytest.importorskip("pyfftw")
    _ok(tiny_run(tmp_path / "a", name="fftw", env={"PYUL_FFT_BACKEND": "pyfftw"}))
    _ok(tiny_run(tmp_path / "b", name="scipy", env={"PYUL_FFT_BACKEND": "scipy"}))
    for rel in ("Final_psi.npy", os.path.join("Outputs", "NBody", "NTM_#005.npy"),
                os.path.join("Outputs", "egylist.npy"), os.path.join("Outputs", "3WfnRS", "P3R_#005.npy")):
        a = np.load(os.path.join(str(tmp_path / "a"), "fftw", rel))
        b = np.load(os.path.join(str(tmp_path / "b"), "scipy", rel))
        assert np.max(np.abs(a - b)) <= 1e-10 * np.max(np.abs(a)), rel


@pytest.mark.parametrize("encoding", ["cp1252", "gbk"])
def test_run_with_narrow_stdout_bar_mode(tmp_path, encoding):   # P3 / P7
    _ok(tiny_run(tmp_path / "out", saves=2, duration=10,
                 env={"PYTHONIOENCODING": encoding, "PYUL_PROGRESS": "bar"}))
