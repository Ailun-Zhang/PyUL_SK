"""Compatibility shims: sph_harm, simpson (P1/P2), threads (P4), paths (P5), console (P3)."""
import io
import os

import numpy as np
import pytest

from PyUltraLight2 import _compat


# ---------------------------------------------------------------- sph_harm (legacy call semantics)

def _grid():
    th = np.linspace(0, 2 * np.pi, 13)      # azimuth
    ph = np.linspace(0, np.pi, 11)          # polar
    return np.meshgrid(th, ph, indexing="ij")


def test_sph_harm_closed_forms():
    TH, PH = _grid()
    Y = _compat.sph_harm
    assert np.allclose(Y(0, 0, TH, PH), 0.5 / np.sqrt(np.pi), atol=1e-14)
    assert np.allclose(Y(0, 1, TH, PH), np.sqrt(3 / (4 * np.pi)) * np.cos(PH), atol=1e-14)
    assert np.allclose(Y(1, 1, TH, PH), -np.sqrt(3 / (8 * np.pi)) * np.sin(PH) * np.exp(1j * TH), atol=1e-14)
    assert np.allclose(Y(-1, 1, TH, PH), np.sqrt(3 / (8 * np.pi)) * np.sin(PH) * np.exp(-1j * TH), atol=1e-14)
    assert np.allclose(Y(2, 2, TH, PH), 0.25 * np.sqrt(15 / (2 * np.pi)) * np.sin(PH) ** 2 * np.exp(2j * TH),
                       atol=1e-14)


def test_sph_harm_matches_legacy_scipy_when_available():
    import scipy.special as ss
    if not hasattr(ss, "sph_harm"):
        pytest.skip("scipy.special.sph_harm not present (SciPy >= 1.17)")
    import warnings
    TH, PH = _grid()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for n in range(5):
            for m in range(-n, n + 1):
                np.testing.assert_allclose(_compat.sph_harm(m, n, TH, PH), ss.sph_harm(m, n, TH, PH),
                                           rtol=0, atol=1e-13)


def test_module_level_SPH_is_compat(PyUL):
    assert PyUL.SPH is _compat.sph_harm


# ---------------------------------------------------------------- simpson

def test_simpson_matches_scipy():
    import scipy.integrate as si
    x = np.linspace(0, 1, 101)
    y = np.sin(3 * x)
    assert _compat.simpson(y, x) == si.simpson(y, x=x)
    x = np.linspace(0, 1, 100)   # even number of samples
    assert _compat.simpson(np.cos(x), x) == si.simpson(np.cos(x), x=x)


def test_default_soliton_orbit_reference(PyUL):
    # Paper inputs; the value is identical for the untouched 30.40 code on SciPy 1.13.1
    # (scipy.integrate.simps) and for 30.50 on SciPy 1.13.1 and 1.18.1.
    if PyUL.axion_E != 1e-21:
        pytest.skip("reference value is for PYUL_AXION_MASS_EV=1e-21")
    M, v = PyUL.DefaultSolitonOrbit(256, 5.0, 'kpc', 12, 'M_solar_masses', 0.14, 'kpc', 'km/s')
    assert v == pytest.approx(6.397271277497396, rel=1e-13)
    M2, v2 = PyUL.DefaultSolitonOrbit(192, 5.0, 'kpc', 12, 'M_solar_masses', 0.14, 'kpc', 'km/s')
    assert (M2, v2) == (M, v)


# ---------------------------------------------------------------- threads

THREAD_VARS = ("PYUL_NUM_THREADS", "OMP_NUM_THREADS", "SLURM_CPUS_PER_TASK", "NCPUS")


@pytest.fixture
def clean_thread_env(monkeypatch):
    for v in THREAD_VARS:
        monkeypatch.delenv(v, raising=False)
    return monkeypatch


def test_thread_priority_order(clean_thread_env):
    mp = clean_thread_env
    mp.setenv("NCPUS", "5")
    assert _compat.resolve_num_threads(256) == (5, "NCPUS")
    mp.setenv("SLURM_CPUS_PER_TASK", "6")
    assert _compat.resolve_num_threads(256) == (6, "SLURM_CPUS_PER_TASK")
    mp.setenv("OMP_NUM_THREADS", "7,2")
    assert _compat.resolve_num_threads(256) == (7, "OMP_NUM_THREADS")
    mp.setenv("PYUL_NUM_THREADS", "9")
    assert _compat.resolve_num_threads(256) == (9, "PYUL_NUM_THREADS")


def test_thread_small_grid_cap(clean_thread_env):
    mp = clean_thread_env
    mp.setenv("OMP_NUM_THREADS", "16")
    n, src = _compat.resolve_num_threads(64)
    assert n == 4 and "capped" in src
    assert _compat.resolve_num_threads(128)[0] == 16
    mp.setenv("OMP_NUM_THREADS", "2")
    assert _compat.resolve_num_threads(64)[0] == 2
    mp.setenv("PYUL_NUM_THREADS", "8")       # explicit request is not capped
    assert _compat.resolve_num_threads(64) == (8, "PYUL_NUM_THREADS")


def test_thread_invalid_values_ignored(clean_thread_env):
    mp = clean_thread_env
    mp.setenv("PYUL_NUM_THREADS", "zero")
    mp.setenv("OMP_NUM_THREADS", "0")
    mp.setenv("NCPUS", "3")
    assert _compat.resolve_num_threads(256) == (3, "NCPUS")


def test_thread_fallback(clean_thread_env):
    n, src = _compat.resolve_num_threads(None)
    assert n >= 1 and src in ("sched_getaffinity", "cpu_count")
    if hasattr(os, "sched_getaffinity"):
        assert n == len(os.sched_getaffinity(0))


# ---------------------------------------------------------------- paths

def test_resource_resolution_order(tmp_path, monkeypatch):
    rel = os.path.join("Soliton Profile Files", "initial_f.npy")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYUL_DATA_DIR", raising=False)
    # repository fallback
    assert os.path.samefile(_compat.find_resource(rel), os.path.join(_compat.REPO_ROOT, rel))
    # CWD (legacy) wins over the repository
    (tmp_path / "Soliton Profile Files").mkdir()
    np.save(tmp_path / "Soliton Profile Files" / "initial_f.npy", np.zeros(3))
    assert os.path.samefile(_compat.find_resource(rel), str(tmp_path / rel))
    # PYUL_DATA_DIR wins over CWD
    d = tmp_path / "data"
    (d / "Soliton Profile Files").mkdir(parents=True)
    np.save(d / "Soliton Profile Files" / "initial_f.npy", np.ones(3))
    monkeypatch.setenv("PYUL_DATA_DIR", str(d))
    assert os.path.samefile(_compat.find_resource(rel), str(d / rel))


def test_missing_resource_message(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="PYUL_DATA_DIR"):
        _compat.require_resource("no such file.npy")


def test_green_cache_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("PYUL_CACHE_DIR", str(tmp_path / "c"))
    p = _compat.green_function_write_path(12)
    assert p == os.path.join(str(tmp_path / "c"), "Green Functions", "G12.npy")
    assert _compat.green_function_read_candidates(12)[0] == p
    _compat.atomic_save_npy(p, np.arange(5.0))
    assert np.array_equal(np.load(p), np.arange(5.0))
    leftovers = [f for f in os.listdir(os.path.dirname(p)) if f.endswith(".tmp")]
    assert leftovers == []


def _redirect_user_cache(monkeypatch, root):
    monkeypatch.setenv("LOCALAPPDATA", str(root))
    monkeypatch.setenv("XDG_CACHE_HOME", str(root))
    return _compat.user_cache_dir()


def test_green_cache_not_in_site_packages(tmp_path, monkeypatch):      # R9-04 (PORT-1)
    # A non-editable install: REPO_ROOT is site-packages (no pyproject.toml, no Green Functions).
    site = tmp_path / "site-packages"
    site.mkdir()
    monkeypatch.setattr(_compat, "REPO_ROOT", str(site))
    monkeypatch.delenv("PYUL_CACHE_DIR", raising=False)
    user = _redirect_user_cache(monkeypatch, tmp_path / "usercache")
    assert _compat.repo_checkout_dir() is None
    p = _compat.green_function_write_path(12)
    assert p == os.path.join(user, "Green Functions", "G12.npy")
    assert not (site / "Green Functions").exists()          # never created in site-packages
    # PYUL_CACHE_DIR still wins
    monkeypatch.setenv("PYUL_CACHE_DIR", str(tmp_path / "c"))
    assert _compat.green_function_write_path(12) == os.path.join(str(tmp_path / "c"), "Green Functions", "G12.npy")


def test_green_cache_in_checkout(tmp_path, monkeypatch):               # R9-04 (PORT-1)
    repo = tmp_path / "PyUL_SK"
    (repo / "Green Functions").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\n")
    monkeypatch.setattr(_compat, "REPO_ROOT", str(repo))
    monkeypatch.delenv("PYUL_CACHE_DIR", raising=False)
    _redirect_user_cache(monkeypatch, tmp_path / "usercache")
    assert _compat.repo_checkout_dir() == str(repo)
    assert _compat.green_function_write_path(12) == os.path.join(str(repo), "Green Functions", "G12.npy")
    # the real repository is recognised as a checkout (tests run from a checkout or an sdist)
    monkeypatch.undo()
    assert _compat.repo_checkout_dir() == _compat.REPO_ROOT


def test_check_safe_rmtree(tmp_path, monkeypatch):                     # R9-01 (CORE-1)
    work = tmp_path / "a" / "b"
    work.mkdir(parents=True)
    monkeypatch.chdir(work)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    for bad in (os.path.abspath(os.sep), str(home), "~", ".", "..", str(tmp_path), str(work),
                _compat.REPO_ROOT, os.path.dirname(_compat.REPO_ROOT)):
        with pytest.raises(ValueError, match="refusing to delete"):
            _compat.check_safe_rmtree(bad)
    ok = work / "runs"
    assert _compat.check_safe_rmtree(str(ok)) == os.path.realpath(str(ok))
    assert _compat.check_safe_rmtree(str(home / "runs")) == os.path.realpath(str(home / "runs"))


# ---------------------------------------------------------------- console / interactivity

class _Narrow(io.StringIO):
    encoding = "cp1252"

    def write(self, s):
        s.encode(self.encoding)       # raises like a real cp1252 stream
        return super().write(s)


def test_safe_print_narrow_stream():
    buf = _Narrow()
    _compat.safe_print("a‑b●", file=buf)
    assert buf.getvalue().startswith("a?b?")
    assert _compat.progress_glyphs(buf)["full"] == "#"


def test_is_interactive_env_override(monkeypatch):
    monkeypatch.setenv("PYUL_INTERACTIVE", "1")
    assert _compat.is_interactive() is True
    monkeypatch.setenv("PYUL_INTERACTIVE", "0")
    assert _compat.is_interactive() is False


def test_progress_mode_env(monkeypatch):
    for m in ("bar", "lines", "quiet"):
        monkeypatch.setenv("PYUL_PROGRESS", m)
        assert _compat.resolve_progress_mode() == m
    monkeypatch.delenv("PYUL_PROGRESS")
    assert _compat.resolve_progress_mode(io.StringIO()) == "lines"


def test_fft_backend_selection(monkeypatch):
    monkeypatch.setenv("PYUL_FFT_BACKEND", "scipy")
    assert _compat.resolve_fft_backend().name == "scipy"
    monkeypatch.setenv("PYUL_FFT_BACKEND", "nonsense")
    with pytest.raises(ValueError):
        _compat.resolve_fft_backend()


def test_scipy_plans_match_numpy():
    be = _compat.resolve_fft_backend("scipy")
    rng = np.random.default_rng(0)
    a = rng.standard_normal((8, 8, 8)) + 1j * rng.standard_normal((8, 8, 8))
    f = be.builders.fftn(a, axes=(0, 1, 2), threads=2)
    assert np.allclose(f(a), np.fft.fftn(a))
    fi = be.builders.ifftn(a, axes=(0, 1, 2), threads=2)
    assert np.allclose(fi(f(a)), a)
    r = rng.standard_normal((8, 8, 8))
    rf = be.builders.rfftn(r, axes=(0, 1, 2))
    k = rf(r)
    irf = be.builders.irfftn(k, axes=(0, 1, 2))
    assert np.allclose(irf(k), r)
    fx = be.builders.fftn(np.zeros((16, 8, 8), complex), axes=(0,))
    assert np.allclose(fx(r.repeat(2, axis=0)), np.fft.fft(r.repeat(2, axis=0), axis=0))
