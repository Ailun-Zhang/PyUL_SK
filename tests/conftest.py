"""Shared pytest set-up for PyUL_SK.

The axion mass is read once at import, so it is fixed here (1e-21 eV, the
paper value) before PyUltraLight2 is imported.  All tests run in
non-interactive mode with line progress output, and write caches and outputs
under pytest's tmp_path (use ``--basetemp`` to choose the folder).
"""
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("PYUL_AXION_MASS_EV", "1e-21")
os.environ["PYUL_INTERACTIVE"] = "0"
os.environ.setdefault("PYUL_PROGRESS", "lines")
os.environ.setdefault("MPLBACKEND", "Agg")

if REPO not in sys.path:
    sys.path.insert(0, REPO)


@pytest.fixture(scope="session", autouse=True)
def cache_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("pyul_cache")
    os.environ["PYUL_CACHE_DIR"] = str(d)
    return d


@pytest.fixture(scope="session")
def PyUL(cache_dir):
    """The core module (public and private names)."""
    from PyUltraLight2 import Current_SK
    return Current_SK


def run_python(code, env=None, cwd=None, timeout=600):
    """Run ``code`` in a fresh interpreter (same executable) and return the
    CompletedProcess.  The repository is put on PYTHONPATH."""
    full_env = dict(os.environ)
    full_env["PYTHONPATH"] = REPO + os.pathsep + full_env.get("PYTHONPATH", "")
    full_env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    if env:
        full_env.update(env)
    return subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, env=full_env,
                          cwd=cwd, timeout=timeout, stdin=subprocess.DEVNULL)


TINY_DRIVER = r'''
import os, sys, json
import numpy as np
import PyUltraLight2 as PyUL
save_path = sys.argv[1]
resol = int(os.environ.get("TINY_RESOL", "32"))
length, length_units = 5.0, 'kpc'
duration, duration_units = float(os.environ.get("TINY_DURATION", "30")), 'Myr'
NS = int(os.environ.get("TINY_NS", "32"))
save_number = int(os.environ.get("TINY_SAVES", "5"))
Save_Options = os.environ.get("TINY_SAVE_OPTIONS", 'Energy DF ULDCOM NBody 3WfnRS')
mS = 12; mP = 0.02 * mS; Radius = 0.14
gridspace = PyUL.MeshSpacing(resol, length, length_units, silent=True)
a = PyUL.GenPlummer(gridspace / 2, length_units)
MIn, vRel0 = PyUL.DefaultSolitonOrbit(resol, length, length_units, mS, 'M_solar_masses', Radius, 'kpc', 'km/s')
M = mS + mP
particles = [[mP, [Radius*mS/M, 0, 0], [0, -vRel0*mS/M, 0]]]
solitons = [[mS, [-Radius*mP/M, 0, 0], [0, vRel0*mP/M, 0], 0]]
PyUL.DSManagement(save_path, Force=False)
run_folder = PyUL.GenerateConfig(NS, length, length_units, resol, duration, duration_units, 1,
    save_number, Save_Options, save_path, 'npy', 'M_solar_masses', 'kpc', 'km/s', solitons,
    0., 'M_solar_masses', 'kpc', 'km/s', particles, [], False, 0, '', a, [0,0,0], False,
    os.environ.get("TINY_NAME", "tiny"))
kw = json.loads(os.environ.get("TINY_EVOLVE_KW", "{}"))
base = dict(DumpInit=True, DumpFinal=True, CenterCalc=True, Length_Ratio=1, resolR=resol, CellCenteredGrid=True)
base.update(kw)
if os.environ.get("TINY_MASSFUNC_ONE"):
    def one(t):
        return 1.0
    base.update(MassChange=True, MassFunc=one)
PyUL.evolve(save_path, run_folder, **base)
print("RUN_FOLDER", run_folder)
'''


def tiny_run(save_path, cwd=None, evolve_kw=None, env=None, **settings):
    """Run a tiny (default 32^3, 5 snapshots) simulation in a fresh process."""
    import json
    e = {"TINY_EVOLVE_KW": json.dumps(evolve_kw or {})}
    for k, v in settings.items():
        e["TINY_" + k.upper()] = str(v)
    if env:
        e.update(env)
    code = TINY_DRIVER.replace("sys.argv[1]", repr(str(save_path)))
    return run_python(code, env=e, cwd=cwd)
