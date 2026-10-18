"""Import in a fresh process: portability (P1), console encodings (P3), version."""
import sys

import pytest

from conftest import run_python


def test_import_fresh_process():
    r = run_python("import PyUltraLight2 as P; print('VERSION', P.S_version); print('DV', P.D_version)")
    out = r.stdout.decode("utf-8", "replace")
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    assert "VERSION 30.5" in out
    assert "comment out line" not in out          # P1: old h5py message removed
    assert "Importing h5py" not in out


def test_d_version_is_ascii():
    r = run_python("import PyUltraLight2 as P; assert all(ord(c) < 128 for c in P.D_version)")
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")


def test_optional_modules_are_lazy():
    # h5py must not be imported at import time; IPython is optional.
    code = ("import sys; import PyUltraLight2 as P; "
            "print('H5PY_LOADED', 'h5py' in sys.modules); print('IPY_LOADED', 'IPython' in sys.modules); "
            "print('CLEAR', callable(P.clear_output))")
    r = run_python(code)
    out = r.stdout.decode()
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    assert "H5PY_LOADED False" in out
    assert "IPY_LOADED False" in out
    assert "CLEAR True" in out


def test_import_without_ipython_numba_pyfftw_h5py():
    # Simulate a minimal install by blocking the optional modules.
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('IPython', 'numba', 'pyfftw', 'h5py'):\n"
        "            raise ImportError('blocked for test: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "import PyUltraLight2 as P\n"
        "print('NUMBA', P.numba, 'PYFFTW', P.pyfftw)\n"
        "print('BACKEND', P._compat.resolve_fft_backend().name)\n"
        "P.clear_output()\n")
    r = run_python(code)
    out = r.stdout.decode("utf-8", "replace")
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    assert "NUMBA None PYFFTW None" in out
    assert "BACKEND scipy" in out
    assert "numba is not available" in out


@pytest.mark.parametrize("encoding", ["cp1252", "gbk", "ascii"])
def test_import_and_progress_bar_with_narrow_stdout(encoding):
    # P3: redirected stdout with a narrow encoding must not crash, also for the bar glyphs.
    code = ("import PyUltraLight2 as P\n"
            "P._PROGRESS_MODE = 'bar'\n"
            "P.prog_bar(100, 5, 0.1, 'FT', 'x')\n"
            "P.prog_bar(100, 50, 0.1, 'IO', 'x', OutNumber=3)\n"
            "P.printU('path with unicode \\u2011\\u00a0\\u25cf', 'Test')\n"
            "P.PyULCredits(True, True, [1])\n")
    r = run_python(code, env={"PYTHONIOENCODING": encoding})
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")


def test_banner_ascii_only():
    r = run_python("import PyUltraLight2", env={"PYTHONIOENCODING": "ascii"})
    assert r.returncode == 0, r.stderr.decode("ascii", "replace")
    assert all(b < 128 for b in r.stdout)
