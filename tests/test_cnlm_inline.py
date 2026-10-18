"""Inline c_nlm decomposition, evolve(Cnlm=True) (30.50).

Equivalence requirements (WP3):

(a) In the same process right after evolve, the toolkit pipeline ``uldm-cnlm`` run on the saved
    3WfnRS cubes gives a c_nlm file (and a_lm file) that is bitwise identical to the inline one,
    including the time axis; ``uldm-cnlm --from-inline`` assembles the same file from the inline
    checkpoints alone.
(b) The historical JSON chain (functions3.process_file -> create_sh_hdf5 -> load_a_lm ->
    compute_cnlm) agrees within pyshtools' cross-process noise (<= 1e-13 relative per mode).
(c) CellCenteredGrid True/False, LegacyResampleBox True/False, CenterCalc True/False
    (CenterCalc=False: the reference cube is rebuilt from the saved full wavefunction 3Wfn
    with the same COM rule).
(d) With Cnlm off nothing changes (test_regression.py, test_runs.py).

The tests are skipped when the ULDM-Eigenmode-Toolkit (Eig_ULDM_packages.cnlm_inline) or
pyshtools cannot be imported.
"""
import json
import os

import numpy as np
import pytest

from conftest import run_python

cnlm_inline = pytest.importorskip("Eig_ULDM_packages.cnlm_inline")
pytest.importorskip("pyshtools")
pytest.importorskip("h5py")

pytestmark = pytest.mark.timeout(1800)

LEGACY_BLOCK = r'''
# (b) historical JSON chain for the first snapshots
if cfg.get("legacy_snaps"):
    idx = list(range(cfg["legacy_snaps"]))
    geom = inl["rs_geometry"]
    work = os.path.join(run, "legacy")
    os.makedirs(os.path.join(work, "3WfnRS"))
    for k in idx:
        shutil.copy(os.path.join(out, "3WfnRS", f"P3R_#{k:03d}.npy"), os.path.join(work, "3WfnRS"))
    cwd = os.getcwd()
    os.chdir(work)
    try:
        r_max = inl["kernel"]["r_max"]
        n_r = cfg["n_shells"]
        r0 = np.linspace(0, r_max, n_r, endpoint=True); r0[0] = 1e-5
        np.save("r_0_values.npy", r0)
        for k in idx:
            for r in np.load("r_0_values.npy"):
                f3.process_file(k, r, cfg["l_max"], geom["dx"], 1, "3WfnRS/P3R_#", cfg["num_points"])
        create_sh_hdf5(output_path="PyUL_psi_alm.h5", start_index=idx[0], end_index=idx[-1], n_r=n_r,
                       l_max=cfg["l_max"], num_points=cfg["num_points"], r_values=r0, file_prefix="3WfnRS/P3R_#",
                       n_workers=1)
        units.set_axion_mass(1e-21)
        r_f, f_dict = load_eigenfunctions(inl["eigen"]["file"])
        rc = np.linspace(0.0, units.convert_between(r_max, 'kpc', '', 'l'), n_r, endpoint=True)
        rc[0] = units.convert_between(1e-5, 'kpc', '', 'l')
        r_al = (rc / rc[-1]) * float(r_f[-1])
        alm = load_a_lm("PyUL_psi_alm.h5", r_al=r_al, final_time=1.0)
        t_ret, c_leg = compute_cnlm(r_f=r_f, f_dict=f_dict, r_al=alm['r'], t_al=alm['t'], a_lm=alm['a_lm'],
                                    r_interp_kind='cubic', time_interp=None, t_out=None, grid_choice='eigen',
                                    r_manual=None, num_workers=1, progress=False)
    finally:
        os.chdir(cwd)
    rel, exact = [], 0
    for key, series in c_leg.items():
        ref = c_inl[key][idx]
        scale = float(np.max(np.abs(ref)))
        d = float(np.max(np.abs(np.asarray(series) - ref)))
        exact += int(d == 0.0)
        rel.append(d / scale if scale > 0 else d)
    rep["legacy"] = {"n_modes": len(rel), "max_rel_per_mode": max(rel), "n_bitwise_modes": exact,
                     "keys_equal": sorted(c_leg) == sorted(c_inl)}
'''

DRIVER = r'''
import os, sys, json, shutil
import numpy as np
import numexpr as ne
import h5py
import PyUltraLight2 as PyUL
from PyUltraLight2 import Current_SK as CS
from Eig_ULDM_packages import pipeline, functions3 as f3, units
from Eig_ULDM_packages.alm_utils import create_sh_hdf5
from Eig_ULDM_packages.c_nlm_integrator import load_eigenfunctions, load_a_lm, compute_cnlm, load_cnlm_from_h5

cfg = json.loads(os.environ["CNLM_CFG"])
save_path = cfg["save_path"]
resol = cfg["resol"]
length, lu = 5.0, 'kpc'
mS = 12; mP = 0.02 * mS; Radius = 0.14
gs = PyUL.MeshSpacing(resol, length, lu, silent=True)
a = PyUL.GenPlummer(gs / 2, lu)
MIn, v = PyUL.DefaultSolitonOrbit(resol, length, lu, mS, 'M_solar_masses', Radius, 'kpc', 'km/s')
M = mS + mP
particles = [[mP, [Radius * mS / M, 0, 0], [0, -v * mS / M, 0]]]
solitons = [[mS, [-Radius * mP / M, 0, 0], [0, v * mP / M, 0], 0]]
flags = 'Energy NBody ULDCOM ' + ('3WfnRS' if cfg["center_calc"] else '3Wfn')
rf = PyUL.GenerateConfig(32, length, lu, resol, cfg["duration"], 'Myr', 1, cfg["saves"], flags, save_path, 'npy',
                         'M_solar_masses', 'kpc', 'km/s', solitons, 0., 'M_solar_masses', 'kpc', 'km/s', particles, [],
                         False, 0, '', a, [0, 0, 0], False, 'cnlm')
kern = dict(CnlmNShells=cfg["n_shells"], CnlmNumPoints=cfg["num_points"], CnlmLmax=cfg["l_max"])
kw = dict(CenterCalc=cfg["center_calc"], Length_Ratio=cfg["lr"], resolR=cfg["resolR"], CellCenteredGrid=cfg["ccg"],
          LegacyResampleBox=cfg["legacy_box"], Cnlm=True, CnlmEigenParams=dict(n_grid=cfg["n_grid"], n_eig=cfg["n_eig"]),
          CnlmSaveAlm=True, CnlmThreads=cfg["threads"], **kern)
kw.update(cfg.get("extra_kw", {}))
PyUL.evolve(save_path, rf, **kw)
run = os.path.join(save_path, rf)
out = os.path.join(run, "Outputs")
inl = json.load(open(os.path.join(out, "Cnlm", "cnlm_inline.json")))
rep = {"inline": inl, "n_saved": int(np.load(os.path.join(out, "snapshot_times_Myr.npy")).size)}

def datasets(path):
    d = {}
    with h5py.File(path, "r") as f:
        f.visititems(lambda n, o: d.__setitem__(n, o[()]) if isinstance(o, h5py.Dataset) else None)
    return d

def bitwise(p1, p2):
    d1, d2 = datasets(p1), datasets(p2)
    bad = sorted(set(d1) ^ set(d2))
    for k in set(d1) & set(d2):
        a1, a2 = np.asarray(d1[k]), np.asarray(d2[k])
        if a1.shape != a2.shape or a1.dtype != a2.dtype or a1.tobytes() != a2.tobytes():
            bad.append(k)
    return {"n_datasets": len(d1), "not_bitwise": sorted(bad)[:10], "n_bad": len(bad)}

cnlm_inline_file = os.path.join(out, "c_nlm.h5")
alm_inline_file = os.path.join(out, "a_lm.h5")
with h5py.File(cnlm_inline_file, "r") as f:
    def scalar(v):
        if isinstance(v, bytes):
            return v.decode()
        return v.item() if hasattr(v, "item") and np.ndim(v) == 0 else v
    rep["attrs"] = {k: scalar(f.attrs[k])
                    for k in ("source", "pyul_version", "pyul_core_sha256", "eigen_file_sha256", "rs_dx",
                              "rs_side", "rs_resol", "l_max", "num_points", "n_r", "angular_grid", "stopped_early")}
    rep["root_keys"] = sorted(f.keys())
t_inl, c_inl, _ = load_cnlm_from_h5(cnlm_inline_file)
rep["t_equals_snapshot_times"] = bool(np.array_equal(t_inl, np.load(os.path.join(out, "snapshot_times_Myr.npy"))))
rep["n_series"] = len(c_inl)

# (a0) --from-inline from the checkpoints alone (no 3WfnRS file used or needed)
res = pipeline.run_pipeline(pipeline.build_parser().parse_args(
    [run, "--from-inline", "--cnlm-out", os.path.join(run, "from_inline_cnlm.h5"),
     "--alm-out", os.path.join(run, "from_inline_alm.h5")]))
rep["from_inline_cnlm"] = bitwise(cnlm_inline_file, res["cnlm_file"])
rep["from_inline_alm"] = bitwise(alm_inline_file, res["alm_file"])

# reference cubes for CenterCalc=False: rebuild 3WfnRS from the saved full wavefunction
if not cfg["center_calc"]:
    lengthC = CS.convert(length, lu, 'l')
    dx = lengthC / float(resol)
    if cfg["ccg"]:
        gridvec = -lengthC / 2.0 + dx / 2.0 + np.arange(resol) * dx
    else:
        gridvec = np.linspace(-lengthC / 2.0, lengthC / 2.0, resol, endpoint=False)
    xa, ya, za = np.meshgrid(gridvec, gridvec, gridvec, sparse=True, indexing='ij')
    os.makedirs(os.path.join(out, "3WfnRS"))
    for k in range(rep["n_saved"]):
        psi = np.load(os.path.join(out, "3Wfn", f"P3D_#{k:03d}.npy"))
        rho = ne.evaluate("abs(abs(psi)**2)").real
        COM = CS.Find3BoxCOM(rho, xa, ya, za)
        CS.Resample3Box(psi, COM, gridvec, run, k, 'npy', cfg["lr"], cfg["resolR"], Save_Psi=True,
                        BoxLength=None if cfg["legacy_box"] else lengthC)

# (a) toolkit pipeline on the 3WfnRS files, same process
pdir = os.path.join(run, "pipe")
res = pipeline.run_pipeline(pipeline.build_parser().parse_args(
    [run, "--eigen-file", inl["eigen"]["file"], "--n-shells", str(cfg["n_shells"]), "--num-points",
     str(cfg["num_points"]), "--l-max", str(cfg["l_max"]), "--workers", "2", "--out-dir", pdir, "--alm-out", "alm.h5", "--skip-missing"]))
rep["pipeline_cnlm"] = bitwise(cnlm_inline_file, res["cnlm_file"])
rep["pipeline_alm"] = bitwise(alm_inline_file, res["alm_file"])

''' + LEGACY_BLOCK + '''print("CNLM_REPORT " + json.dumps(rep, default=str))
'''

LEGACY_ONLY = r'''
import os, sys, json, shutil
import numpy as np
from Eig_ULDM_packages import functions3 as f3, units
from Eig_ULDM_packages.alm_utils import create_sh_hdf5
from Eig_ULDM_packages.c_nlm_integrator import load_eigenfunctions, load_a_lm, compute_cnlm, load_cnlm_from_h5
cfg = json.loads(os.environ["CNLM_CFG"])
run = os.path.join(cfg["save_path"], "cnlm")
out = os.path.join(run, "Outputs")
inl = json.load(open(os.path.join(out, "Cnlm", "cnlm_inline.json")))
t_inl, c_inl, _ = load_cnlm_from_h5(os.path.join(out, "c_nlm.h5"))
shutil.rmtree(os.path.join(run, "legacy"))
rep = {}
''' + LEGACY_BLOCK + '''
print("CNLM_REPORT " + json.dumps(rep, default=str))
'''

BASE = dict(duration=20, saves=3, n_shells=17, num_points=32, l_max=4, n_grid=512, n_eig=5, threads=2,
            lr=1, legacy_snaps=0)

CASES = {
    "48_ccg_newbox_cc":     dict(resol=48, resolR=48, ccg=True, legacy_box=False, center_calc=True, threads=1,
                                 legacy_snaps=2),
    "48_ccg_legacybox_cc":  dict(resol=48, resolR=48, ccg=True, legacy_box=True, center_calc=True, threads=3),
    "48_node_newbox_cc":    dict(resol=48, resolR=48, ccg=False, legacy_box=False, center_calc=True),
    "48_node_legacybox_nocc": dict(resol=48, resolR=48, ccg=False, legacy_box=True, center_calc=False,
                                   legacy_snaps=2),
    "64_ccg_newbox_cc_lr05": dict(resol=64, resolR=32, lr=0.5, ccg=True, legacy_box=False, center_calc=True,
                                  legacy_snaps=2),
    "64_ccg_legacybox_nocc": dict(resol=64, resolR=64, ccg=True, legacy_box=True, center_calc=False),
}


def _run_driver(code, cfg, name):
    r = run_python(code, env={"CNLM_CFG": json.dumps(cfg), "PYUL_PROGRESS": "quiet"}, timeout=1500)
    out = r.stdout.decode("utf-8", "replace")
    assert r.returncode == 0, out[-3000:] + "\n---\n" + r.stderr.decode("utf-8", "replace")[-4000:]
    line = [ln for ln in out.splitlines() if ln.startswith("CNLM_REPORT ")][-1]
    rep = json.loads(line[len("CNLM_REPORT "):])
    report_dir = os.environ.get("PYUL_CNLM_REPORT_DIR")   # optional: keep the numbers for a report
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
        with open(os.path.join(report_dir, name + ".json"), "w", encoding="utf-8") as fh:
            json.dump({"cfg": cfg, "report": rep}, fh, indent=1, default=str)
    return rep


def _run_case(tmp_path, cfg, name):
    cfg = dict(BASE, **cfg, save_path=str(tmp_path / "out"))
    return cfg, _run_driver(DRIVER, cfg, name)


@pytest.mark.parametrize("case", sorted(CASES))
def test_inline_equals_pipeline_and_legacy(tmp_path, case):
    cfg, rep = _run_case(tmp_path, CASES[case], case)
    n_t = cfg["saves"] + 1
    # file layout: exactly the toolkit c_nlm layout, provenance as root attributes only
    assert rep["root_keys"] == ["n", "t"]
    assert rep["n_series"] == cfg["n_eig"] * (cfg["l_max"] + 1) ** 2
    assert rep["n_saved"] == n_t and rep["t_equals_snapshot_times"]
    at = rep["attrs"]
    assert at["source"] == "pyul-inline" and at["pyul_version"] == 30.5 and len(at["pyul_core_sha256"]) == 64
    assert len(at["eigen_file_sha256"]) == 64 and at["n_r"] == cfg["n_shells"] and at["l_max"] == cfg["l_max"]
    assert at["angular_grid"] == "legacy" and not at["stopped_early"]
    # RS geometry: the cube's true spacing
    L = 5.0
    side = (L - L / cfg["resol"]) * cfg["lr"] if (cfg["legacy_box"] and cfg["ccg"]) else L * cfg["lr"]
    assert at["rs_resol"] == cfg["resolR"]
    assert at["rs_side"] == pytest.approx(side, rel=1e-12)
    assert at["rs_dx"] == pytest.approx(side / cfg["resolR"], rel=1e-12)
    # (a) bitwise: toolkit pipeline on the 3WfnRS files, and --from-inline on the checkpoints
    for key in ("pipeline_cnlm", "pipeline_alm", "from_inline_cnlm", "from_inline_alm"):
        assert rep[key]["n_bad"] == 0, (key, rep[key])
    assert rep["pipeline_cnlm"]["n_datasets"] == 1 + rep["n_series"]
    assert rep["pipeline_alm"]["n_datasets"] == 1 + 2 * n_t
    # (b) historical JSON chain: same process (bitwise in practice) and a separate process
    #     (pyshtools' FFT is only reproducible within one process: <= 1e-13 relative per mode)
    if cfg["legacy_snaps"]:
        leg = rep["legacy"]
        assert leg["keys_equal"] and leg["n_modes"] == rep["n_series"]
        assert leg["max_rel_per_mode"] <= 1e-13, leg
        leg2 = _run_driver(LEGACY_ONLY, cfg, case + "_legacy_other_process")["legacy"]
        assert leg2["keys_equal"] and leg2["n_modes"] == rep["n_series"]
        assert leg2["max_rel_per_mode"] <= 1e-13, leg2


def test_inline_autostop_writes_partial_file(tmp_path):
    # AutoStop2 with a tiny threshold halts the run after the first step(s).
    cfg, rep = _run_case(tmp_path, dict(resol=32, resolR=32, ccg=True, legacy_box=False, center_calc=True,
                                        saves=4, extra_kw=dict(AutoStop2=True, WellThreshold=1e-9)), "autostop")
    assert 1 <= rep["n_saved"] < cfg["saves"] + 1
    assert rep["t_equals_snapshot_times"] and rep["attrs"]["stopped_early"]
    for key in ("pipeline_cnlm", "from_inline_cnlm"):
        assert rep[key]["n_bad"] == 0, (key, rep[key])
    assert rep["inline"]["status"].startswith("finished (stopped early)")


# --------------------------------------------------------------------------- fail fast
def _config(PyUL, tmp_path, solitons=None, name="v"):
    resol, length, lu = 32, 5.0, 'kpc'
    gs = PyUL.MeshSpacing(resol, length, lu, silent=True)
    a = PyUL.GenPlummer(gs / 2, lu)
    if solitons is None:
        solitons = [[12, [0, 0, 0], [0, 0, 0], 0]]
    particles = [[0.24, [0.1, 0, 0], [0, 0, 0]]]
    save_path = str(tmp_path / "out")
    rf = PyUL.GenerateConfig(32, length, lu, resol, 10, 'Myr', 1, 2, 'Energy NBody', save_path, 'npy',
                             'M_solar_masses', 'kpc', 'km/s', solitons, 0., 'M_solar_masses', 'kpc', 'km/s',
                             particles, [], False, 0, '', a, [0, 0, 0], False, name)
    return save_path, rf


def _no_outputs(save_path, rf):
    return not os.path.exists(os.path.join(save_path, rf, "Outputs"))


def test_validation_fails_before_any_output(PyUL, tmp_path):
    save_path, rf = _config(PyUL, tmp_path, solitons=[[12, [0, 0, 0], [0, 0, 0], 0], [6, [1, 0, 0], [0, 0, 0], 0]])
    with pytest.raises(ValueError, match="CnlmSolitonMass"):
        PyUL.evolve(save_path, rf, Cnlm=True)
    assert _no_outputs(save_path, rf)
    save2, rf2 = _config(PyUL, tmp_path / "ui", name="ui")
    with pytest.raises(ValueError, match="UseInit"):
        PyUL.evolve(save2, rf2, Cnlm=True, UseInit=True, InitPath="whatever")
    assert _no_outputs(save2, rf2)
    for kw, exc, pat in ((dict(CnlmNumPoints=31), ValueError, "even"),
                         (dict(CnlmLmax=20, CnlmNumPoints=32), ValueError, "CnlmLmax"),
                         (dict(CnlmNShells=1), ValueError, "CnlmNShells"),
                         (dict(CnlmAngularGrid="gauss"), ValueError, "CnlmAngularGrid"),
                         (dict(CnlmThreads=0), ValueError, "CnlmThreads"),
                         (dict(CnlmEigenParams=dict(bogus=1)), ValueError, "CnlmEigenParams"),
                         (dict(CnlmEigenFile=str(tmp_path / "missing.h5")), FileNotFoundError, "CnlmEigenFile")):
        with pytest.raises(exc, match=pat):
            PyUL.evolve(save_path, rf, Cnlm=True, CnlmSolitonMass=12, **kw)
        assert _no_outputs(save_path, rf), kw
    existing = tmp_path / "exists.h5"
    existing.write_bytes(b"x")
    with pytest.raises(FileExistsError, match="OverwriteOutputs"):
        PyUL.evolve(save_path, rf, Cnlm=True, CnlmSolitonMass=12, CnlmOutput=str(existing))
    assert _no_outputs(save_path, rf)


def test_eigen_lmax_and_rmax_checked_before_loop(PyUL, tmp_path):
    from Eig_ULDM_packages import units
    from Eig_ULDM_packages.pipeline import build_eigen_file
    units.set_axion_mass(1e-21)
    eig = str(tmp_path / "eig_l2.h5")
    build_eigen_file(eig, soliton_mass=12.0, mass_units="M_solar_masses", rmax=2.5, length_units="kpc",
                     n_grid=128, lmax=2, n_eig=2)
    save_path, rf = _config(PyUL, tmp_path)
    with pytest.raises(ValueError, match="lmax=2"):
        PyUL.evolve(save_path, rf, Cnlm=True, CnlmEigenFile=eig, CnlmLmax=3, CnlmNumPoints=16)
    assert _no_outputs(save_path, rf)
    # r_max mismatch (eigen grid to 2.5 kpc, shells to 1.25 kpc) is found in setup, before the first save
    with pytest.raises(ValueError, match="r_max"):
        PyUL.evolve(save_path, rf, Cnlm=True, CnlmEigenFile=eig, CnlmLmax=2, CnlmNumPoints=16, CnlmNShells=9,
                    CenterCalc=True, Length_Ratio=0.5, resolR=16, OverwriteOutputs=True)
    out = os.path.join(save_path, rf, "Outputs")
    assert not os.path.exists(os.path.join(out, "NBody", "NTM_#000.npy"))
    assert not os.path.exists(os.path.join(out, "c_nlm.h5"))


def test_cnlm_off_does_not_import_toolkit(tmp_path):
    code = ("import sys, PyUltraLight2\n"
            "print('TOOLKIT_LOADED', any(m.startswith('Eig_ULDM_packages') for m in sys.modules),"
            " 'pyshtools' in sys.modules)\n")
    r = run_python(code)
    assert r.returncode == 0 and b"TOOLKIT_LOADED False False" in r.stdout


# --------------------------------------------------------------------------- nothing else changes
_TINY_KW = dict(Cnlm=True, CnlmNShells=9, CnlmNumPoints=16, CnlmLmax=2, CnlmEigenParams=dict(n_grid=256, n_eig=3))
_INLINE_ONLY = ("Cnlm", "c_nlm.h5", "a_lm.h5")


def _ok(r):
    assert r.returncode == 0, (r.stdout.decode("utf-8", "replace")[-3000:] + "\n---\n"
                               + r.stderr.decode("utf-8", "replace")[-3000:])
    return r


def _files(root):
    out = {}
    for d, _, names in os.walk(root):
        for n in names:
            rel = os.path.relpath(os.path.join(d, n), root).replace("\\", "/")
            if rel.startswith("Outputs/") and rel.split("/")[1] in _INLINE_ONLY:
                continue
            if n.endswith(".log"):
                continue
            data = open(os.path.join(d, n), "rb").read()
            if rel == "config.uldm":            # written by GenerateConfig with a time stamp
                cfg = json.loads(data)
                cfg.pop("Config Generation Time", None)
                data = json.dumps(cfg, sort_keys=True).encode()
            out[rel] = data
    return out


def test_cnlm_on_leaves_every_other_output_unchanged(tmp_path):
    from conftest import tiny_run
    _ok(tiny_run(tmp_path / "off", saves=3, duration=10))
    _ok(tiny_run(tmp_path / "on", saves=3, duration=10, evolve_kw=_TINY_KW))
    off, on = _files(str(tmp_path / "off" / "tiny")), _files(str(tmp_path / "on" / "tiny"))
    assert sorted(off) == sorted(on)
    differ = [k for k in off if off[k] != on[k]]
    assert differ == [], differ
    assert os.path.isfile(str(tmp_path / "on" / "tiny" / "Outputs" / "c_nlm.h5"))


def test_failure_during_the_run_is_a_warning(tmp_path):
    from conftest import TINY_DRIVER
    prefix = ("from Eig_ULDM_packages import cnlm_inline as _ci\n"
              "_orig = _ci.InlineCnlmRecorder.record\n"
              "def _bad(self, index, psi, t):\n"
              "    if index == 2:\n"
              "        raise OSError('simulated full disk')\n"
              "    return _orig(self, index, psi, t)\n"
              "_ci.InlineCnlmRecorder.record = _bad\n")
    from conftest import tiny_run
    _ok(tiny_run(tmp_path / "off", saves=3, duration=10))
    code = prefix + TINY_DRIVER.replace("sys.argv[1]", repr(str(tmp_path / "on")))
    r = _ok(run_python(code, env={"TINY_EVOLVE_KW": json.dumps(_TINY_KW), "TINY_SAVES": "3",
                                  "TINY_DURATION": "10"}))
    err = r.stderr.decode("utf-8", "replace") + r.stdout.decode("utf-8", "replace")
    assert "inline c_nlm failed at save 2" in err and "simulated full disk" in err
    run_on = str(tmp_path / "on" / "tiny")
    assert not os.path.exists(os.path.join(run_on, "Outputs", "c_nlm.h5"))
    ck = sorted(os.listdir(os.path.join(run_on, "Outputs", "Cnlm", "alm_checkpoints")))
    assert ck == ["ALM_#000.npz", "ALM_#001.npz"]
    off, on = _files(str(tmp_path / "off" / "tiny")), _files(run_on)
    assert sorted(off) == sorted(on) and all(off[k] == on[k] for k in off)


def test_existing_cnlm_inside_outputs_follows_outputs_rule(PyUL, tmp_path, monkeypatch, capsys):  # R9-02
    # An existing Outputs/c_nlm.h5 is replaced together with Outputs (interactive Silent=True,
    # or the [Y/n] answer), not refused before evolve's own Outputs decision (CNLM-6).
    save_path, rf = _config(PyUL, tmp_path)
    kw = dict(_TINY_KW, CenterCalc=True, Length_Ratio=1, resolR=16, CellCenteredGrid=True)
    cnlm = os.path.join(save_path, rf, "Outputs", "c_nlm.h5")
    PyUL.evolve(save_path, rf, **kw)                                 # first run (non-interactive)
    assert os.path.isfile(cnlm)
    with pytest.raises(FileExistsError, match="Outputs already exists"):
        PyUL.evolve(save_path, rf, **kw)                             # non-interactive: Outputs rule
    monkeypatch.setenv("PYUL_INTERACTIVE", "1")
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    assert PyUL.evolve(save_path, rf, **kw) is None                  # answer 'n': nothing touched
    assert os.path.isfile(cnlm)
    os.utime(cnlm, (1, 1))
    capsys.readouterr()
    PyUL.evolve(save_path, rf, Silent=True, **kw)                    # interactive Silent=True: replaced
    assert "Pre-existing Output files removed." in capsys.readouterr().out
    assert os.path.isfile(cnlm) and os.path.getmtime(cnlm) > 1
    outside = tmp_path / "elsewhere.h5"
    outside.write_bytes(b"x")
    with pytest.raises(FileExistsError, match="OverwriteOutputs"):  # outside Outputs: still refused
        PyUL.evolve(save_path, rf, Silent=True, CnlmOutput=str(outside), **kw)
    assert outside.read_bytes() == b"x" and os.path.isfile(cnlm)


def test_eigen_file_inside_outputs_refused_before_delete(PyUL, tmp_path):            # R9-03 (CNLM-1)
    # Rerunning with the automatic eigen file of the previous run (Outputs/Cnlm/eigs_*.h5) and
    # OverwriteOutputs=True must stop before Outputs is deleted (toolkit check, outputs_replaced).
    import inspect
    if "outputs_replaced" not in inspect.signature(cnlm_inline.CnlmInlineSettings.validate).parameters:
        pytest.skip("toolkit without the CnlmEigenFile-inside-Outputs check")
    save_path, rf = _config(PyUL, tmp_path)
    kw = dict(_TINY_KW, CenterCalc=True, Length_Ratio=1, resolR=16, CellCenteredGrid=True)
    PyUL.evolve(save_path, rf, **kw)
    out = os.path.join(save_path, rf, "Outputs")
    eig = [f for f in os.listdir(os.path.join(out, "Cnlm")) if f.startswith("eigs_") and f.endswith(".h5")]
    assert len(eig) == 1
    eig = os.path.join(out, "Cnlm", eig[0])
    before = sorted(os.listdir(out))
    with pytest.raises(ValueError, match="inside"):
        PyUL.evolve(save_path, rf, CnlmEigenFile=eig, OverwriteOutputs=True, **kw)
    assert os.path.isfile(eig) and sorted(os.listdir(out)) == before
    assert os.path.isfile(os.path.join(out, "c_nlm.h5"))
