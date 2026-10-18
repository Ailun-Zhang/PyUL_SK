"""Unit tests for the Policy A bug fixes (P11-P23, C1) that do not need a full run."""
import json
import os
import warnings

import numpy as np
import pytest


def test_convert_back_c_roundtrip(PyUL):          # P15
    for v in (1e-3, 0.1, 0.5):
        code = PyUL.convert(v, 'c', 'v')
        assert PyUL.convert_back(code, 'c', 'v') == pytest.approx(v, rel=1e-14)


def _two_body_state(sep=0.5):
    # particle 0 at -sep/2, particle 1 at +sep/2 on the x axis, at rest
    return np.array([-sep / 2, 0, 0, 0, 0, 0, sep / 2, 0, 0, 0, 0, 0], dtype=float)


@pytest.mark.parametrize("which", ["FWNBody", "FWNBody_NI"])
def test_pair_force_point_mass_attractive(PyUL, which):   # P13
    resol, L = 16, 4.0
    PyUL._GRID_CONTEXT.reset()
    phi = np.zeros((resol, resol, resol))
    masses = np.array([1.0, 2.0])
    f = getattr(PyUL, which)
    d0, _ = f(0, _two_body_state(), masses, phi, 0, L, resol)
    dP, _ = f(0, _two_body_state(), masses, phi, 1e6, L, resol)   # Plummer, a -> infinity
    # particle 0 is pulled towards +x (towards particle 1), particle 1 towards -x
    assert d0[3] > 0 and d0[9] < 0
    assert d0[3] == pytest.approx(2.0 / 0.5 ** 2, rel=1e-12)
    np.testing.assert_allclose(d0[3:6], dP[3:6], rtol=1e-6)
    np.testing.assert_allclose(d0[9:12], dP[9:12], rtol=1e-6)


def test_pair_force_simple_attractive(PyUL):      # P13 (FWNBodySimple)
    d0 = PyUL.FWNBodySimple(_two_body_state(), np.array([1.0, 2.0]), a=0)
    dP = PyUL.FWNBodySimple(_two_body_state(), np.array([1.0, 2.0]), a=1e6)
    assert d0[3] > 0 and d0[9] < 0
    np.testing.assert_allclose(d0, dP, rtol=1e-6)


def test_nbody_advance_validates_ns(PyUL):        # P11
    phi = np.zeros((16, 16, 16))
    st = _two_body_state()
    for NS in (2, 3, 6, 10):
        with pytest.raises(ValueError, match="NS"):
            PyUL.NBodyAdvance(st.copy(), 1e-3, np.array([1.0, 2.0]), phi, 0, 4.0, 16, NS)
        with pytest.raises(ValueError, match="NS"):
            PyUL.NBodyAdvance_NI(st.copy(), 1e-3, np.array([1.0, 2.0]), phi, 0, 4.0, 16, NS)
        assert not PyUL._valid_NS(NS)
    for NS in (0, 1, 4, 32):
        assert PyUL._valid_NS(NS)
    _, g = PyUL.NBodyAdvance(st.copy(), 1e-3, np.array([1.0, 2.0]), phi, 0, 4.0, 16, 0)
    assert g.shape == (6,)                        # 3 per particle (was 6 per particle)


def _psi_setup(n=8, L=2.0):
    x = np.linspace(-L / 2, L / 2, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    psi = np.exp(-(X ** 2 + Y ** 2 + Z ** 2)) * np.exp(1j * (0.7 * X - 0.3 * Y))
    k = 2 * np.pi * np.fft.fftfreq(n, L / n)
    Kx, Ky, Kz = np.meshgrid(k, k, k, sparse=True, indexing='ij')
    return x, psi, Kx, Ky, Kz


@pytest.mark.parametrize("fname", ["pEval", "LpEval"])
def test_momentum_does_not_modify_funct(PyUL, fname):     # P12
    x, psi, Kx, Ky, Kz = _psi_setup()
    funct = np.fft.fftn(psi)
    ref = funct.copy()
    getattr(PyUL, fname)(psi, np.abs(psi) ** 2, funct, 8, x, Kx, Ky, Kz, np.fft.ifftn)
    assert np.array_equal(funct, ref)


def test_pEval_values_unchanged(PyUL):          # P12: same numbers as the in-place version
    x, psi, Kx, Ky, Kz = _psi_setup()
    funct = np.fft.fftn(psi)
    out = PyUL.pEval(psi, None, funct, 8, x, Kx, Ky, Kz, np.fft.ifftn)
    f2 = funct.copy()
    f2 *= 1j
    DPx = np.fft.ifftn(Kx * f2)
    A = np.abs(psi)
    DAx = np.gradient(A, x[1] - x[0])[0]
    ref = np.imag(A * DAx - np.conj(psi) * DPx) * (x[1] - x[0]) ** 3
    np.testing.assert_allclose(out[0], ref, rtol=1e-12, atol=1e-15)


def test_nbody_energy_plummer_and_z(PyUL):        # P16
    G = 6.67e-11
    m = [1e30, 2e30]
    # separation purely along z: 1e19 m
    data = [np.array([0, 0, 0, 0, 0, 0, 0, 0, 1e19, 0, 0, 0], dtype=float)]
    n, KS, PS = PyUL.NBodyEnergy(m, data, 1, a=0, length_units='kpc')
    assert PS[0] == pytest.approx(-G * m[0] * m[1] / 1e19, rel=1e-12)
    a = PyUL.GenPlummer(0.01, 'kpc')
    n, KS, PSa = PyUL.NBodyEnergy(m, data, 1, a=a, length_units='kpc')
    assert np.isfinite(PSa[0]) and PSa[0] < 0


def test_interpolate_curve3_uses_z(PyUL):         # P17
    t = np.arange(6.0)
    fx, fy, fz = PyUL.InterpolateCurve3(t, 2 * t, 3 * t, ResolGain=2)
    np.testing.assert_allclose(fz, 3 * fx, rtol=1e-12, atol=1e-12)


def _resample_case(PyUL, cell_centred, **kw):
    L, n, nR = 4.0, 16, 8
    dx = L / n
    if cell_centred:
        g = -L / 2 + dx / 2 + np.arange(n) * dx
    else:
        g = np.linspace(-L / 2, L / 2, n, endpoint=False)
    X = np.meshgrid(g, g, g, indexing='ij')[0]
    psi = X + 0j                                   # linear in x: interpolation is exact
    return PyUL.Resample3Box(psi, np.zeros(3), g, '', 0, 'npy', 0.5, nR, **kw), L, nR, g


def test_resample3box_geometry_and_return(PyUL):  # C1
    new, L, nR, g = _resample_case(PyUL, True, BoxLength=4.0)
    old, *_ = _resample_case(PyUL, True)
    assert new is not None and new.shape == (nR,) * 3
    side_new = L * 0.5
    side_old = -2 * g[0] * 0.5                      # (L - dx) * Length_Ratio
    assert side_old == pytest.approx((L - L / 16) * 0.5)
    exp_new = np.linspace(-side_new / 2, side_new / 2, nR, endpoint=False)
    exp_old = np.linspace(-side_old / 2, side_old / 2, nR, endpoint=False)
    np.testing.assert_allclose(new[:, 0, 0].real, exp_new, atol=1e-12)
    np.testing.assert_allclose(old[:, 0, 0].real, exp_old, atol=1e-12)
    # node-centred grid: both definitions are bitwise identical
    a, *_ = _resample_case(PyUL, False, BoxLength=4.0)
    b, *_ = _resample_case(PyUL, False)
    assert a.tobytes() == b.tobytes()


def test_inactive_options_warning(PyUL):          # P23
    so = PyUL.SaveOptionsDigest('Energy 3GravRS 3WfnRS Quadrupole')
    with pytest.warns(UserWarning, match="no effect") as rec:
        notes = PyUL._WarnInactiveOptions(so, True, True, True, 0.5, False, False)
    assert len(rec) == 1
    assert len(notes) == 7
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert PyUL._WarnInactiveOptions(PyUL.SaveOptionsDigest('Energy NBody'), False, False, False,
                                         0.9, False, False) == []


# ---------------------------------------------------------------- non-interactive safety (P6, P18, P19)

def test_dsmanagement_force_and_nested(PyUL, tmp_path):
    p = tmp_path / "a" / "b" / "c"
    PyUL.DSManagement(str(p), Force=True)          # missing + nested: was FileNotFoundError
    assert p.is_dir()
    (p / "f.txt").write_text("x")
    PyUL.DSManagement(str(p), Force=False)         # non-interactive: keeps the data, no EOFError
    assert (p / "f.txt").exists()
    PyUL.DSManagement(str(p), Force=True)          # non-empty + Force: was UnboundLocalError
    assert p.is_dir() and not (p / "f.txt").exists()


def test_dsmanagement_force_refuses_cwd_and_ancestors(PyUL, tmp_path, monkeypatch):   # R9-01
    save = tmp_path / "save"
    work = save / "run" / "sub"
    work.mkdir(parents=True)
    (save / "keep.txt").write_text("x")
    monkeypatch.chdir(work)
    for p in (str(save), str(save / "run"), ".", ".."):
        with pytest.raises(ValueError, match="refusing to delete"):
            PyUL.DSManagement(p, Force=True)
    assert (save / "keep.txt").exists() and work.is_dir()
    home = tmp_path / "home"
    (home / "data").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    with pytest.raises(ValueError, match="home directory"):
        PyUL.DSManagement("~", Force=True)
    assert (home / "data").is_dir()
    monkeypatch.chdir(tmp_path)
    PyUL.DSManagement(str(save), Force=True)       # a dedicated save folder is still cleaned
    assert save.is_dir() and not (save / "keep.txt").exists()


def test_evolve_outputs_guard(PyUL, tmp_path, monkeypatch):     # R9-01: no rmtree of Outputs containing the cwd
    args = _config_args(tmp_path, "run")
    assert PyUL.GenerateConfig(*args, True, "run") == "run"
    inner = tmp_path / "run" / "Outputs" / "here"
    inner.mkdir(parents=True)
    monkeypatch.chdir(inner)
    with pytest.raises(ValueError, match="refusing to delete"):
        PyUL.evolve(str(tmp_path), "run", OverwriteOutputs=True)
    assert inner.is_dir()


def _config_args(save_path, name):
    return (32, 5.0, 'kpc', 32, 1, 'Myr', 1, 2, 'Energy NBody', str(save_path), 'npy', 'M_solar_masses',
            'kpc', 'km/s', [[12, [0, 0, 0], [0, 0, 0], 0]], 0., 'M_solar_masses', 'kpc', 'km/s', [], [],
            False, 0, '', 100.0, [0, 0, 0])


@pytest.mark.parametrize("no_interaction", [False, True])
def test_generateconfig_collision_noninteractive(PyUL, tmp_path, no_interaction):
    args = _config_args(tmp_path, "run")
    assert PyUL.GenerateConfig(*args, no_interaction, "run") == "run"
    marker = tmp_path / "run" / "keep.txt"
    marker.write_text("data")
    with pytest.raises(FileExistsError, match="Overwrite=True"):
        PyUL.GenerateConfig(*args, no_interaction, "run")
    assert marker.exists()
    assert PyUL.GenerateConfig(*args, no_interaction, "run", Overwrite=True) == "run"
    assert not marker.exists()
    cfg = json.load(open(tmp_path / "run" / "config.uldm"))
    assert cfg["PyUL Version"] == PyUL.S_version


def test_loadconfig_reads_old_and_new(PyUL, tmp_path):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("QDrag1_QW", "QDrag1_Regular"):         # shipped configs, PyUL 21.3
        out = PyUL.LoadConfig(os.path.join(repo, "Reusable", "Configurations", name))
        assert len(out) == 25
    args = _config_args(tmp_path, "new")
    PyUL.GenerateConfig(*args, True, "new")
    out = PyUL.LoadConfig(str(tmp_path / "new"))
    assert out[0] == 32 and out[3] == 32
    base = os.environ.get("PYUL_BASELINE_DIR")
    if base:                                             # a 30.40 config
        sp = os.path.join(base, "out64_default")
        run = open(os.path.join(sp, "latest.uldm")).read().strip()
        assert PyUL.LoadConfig(os.path.join(sp, run))[3] == 64


def test_runs_noninteractive(PyUL, tmp_path):
    for n in ("r1", "r2"):
        (tmp_path / n).mkdir()
    (tmp_path / "latest.uldm").write_text("r2")
    assert PyUL.Runs(str(tmp_path), Automatic=True) == "r2"
    assert PyUL.Runs(str(tmp_path), Automatic=False) == "r2"
    assert (tmp_path / "r1").is_dir()


@pytest.mark.timeout(300)
def test_bhratiotester_noninteractive(PyUL):
    with pytest.raises(ValueError, match="converge"):
        PyUL.BHRatioTester(0.3, 1, 1e-12, 0, 1, 0)
