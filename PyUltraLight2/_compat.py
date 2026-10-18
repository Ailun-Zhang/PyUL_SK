"""Portability and runtime helpers for the PyUltraLight SK fork (added in 30.50).

Everything in this module is infrastructure: it never changes the physics or
the numbers produced by ``evolve``.  It provides

* import-compatibility shims (``scipy.special.sph_harm``,
  ``scipy.integrate.simps``, IPython ``clear_output``, optional numba),
* an optional FFT backend layer (pyFFTW by default, ``scipy.fft`` fallback),
* thread-count resolution that honours HPC schedulers,
* resource / cache path resolution independent of the working directory,
* console output that never crashes on narrow encodings (cp1252, GBK, ...),
* detection of interactive sessions, so that ``input()`` is never reached in
  batch jobs.

All environment variables read here are documented in CHANGELOG.md:
PYUL_NUM_THREADS, PYUL_DATA_DIR, PYUL_CACHE_DIR, PYUL_FFT_BACKEND,
PYUL_PROGRESS, PYUL_INTERACTIVE.
"""

import os
import sys
import uuid

import numpy as np

# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def stream_can_encode(text, stream=None):
    """Return True if ``text`` can be written to ``stream`` without error."""
    if stream is None:
        stream = sys.stdout
    if stream is None:
        return False
    enc = getattr(stream, "encoding", None) or "ascii"
    try:
        text.encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def safe_print(*args, sep=" ", end="\n", file=None, flush=False):
    """``print`` that falls back to replacement characters instead of raising
    UnicodeEncodeError when the output stream cannot encode the text."""
    stream = sys.stdout if file is None else file
    if stream is None:  # e.g. pythonw without a console
        return
    try:
        print(*args, sep=sep, end=end, file=stream, flush=flush)
    except UnicodeEncodeError:
        text = sep.join(str(a) for a in args) + end
        enc = getattr(stream, "encoding", None) or "ascii"
        try:
            text = text.encode(enc, errors="replace").decode(enc, errors="replace")
        except LookupError:
            text = text.encode("ascii", errors="replace").decode("ascii")
        stream.write(text)
        if flush:
            stream.flush()


_UNICODE_GLYPHS = {"full": "●", "empty": "○", "head": "◎"}
_ASCII_GLYPHS = {"full": "#", "empty": "o", "head": "@"}


def progress_glyphs(stream=None):
    """Glyphs for the legacy progress bar; ASCII when the stream cannot encode
    the original Unicode circles."""
    if stream_can_encode("".join(_UNICODE_GLYPHS.values()), stream):
        return _UNICODE_GLYPHS
    return _ASCII_GLYPHS


def glyph(unicode_char, ascii_char, stream=None):
    """Return ``unicode_char`` if the stream can encode it, else ``ascii_char``."""
    return unicode_char if stream_can_encode(unicode_char, stream) else ascii_char


# ---------------------------------------------------------------------------
# Interactive session detection
# ---------------------------------------------------------------------------

_TRUE = ("1", "true", "yes", "on", "y")
_FALSE = ("0", "false", "no", "off", "n")


def in_ipython_kernel():
    """True inside a Jupyter/IPython kernel, where ``input()`` is forwarded to
    the front end although ``sys.stdin`` is not a TTY."""
    if "ipykernel" not in sys.modules:
        return False
    try:
        from IPython import get_ipython
        shell = get_ipython()
    except Exception:
        return False
    return shell is not None and hasattr(shell, "kernel")


def is_interactive():
    """Decide whether PyUL may ask questions with ``input()``.

    ``PYUL_INTERACTIVE=0|1`` overrides the detection.  Otherwise a session is
    interactive when stdin is a TTY or when running inside a Jupyter kernel.
    PBS/Slurm jobs, ``nohup``, pipes and CI are therefore non-interactive.
    """
    env = os.environ.get("PYUL_INTERACTIVE", "").strip().lower()
    if env in _TRUE:
        return True
    if env in _FALSE:
        return False
    try:
        if sys.stdin is not None and sys.stdin.isatty():
            return True
    except (AttributeError, ValueError, OSError):
        pass
    return in_ipython_kernel()


def stdin_is_tty():
    """TTY test used by GenerateConfig's run-name prompt (30.40 semantics:
    no prompt in Jupyter), honouring the PYUL_INTERACTIVE override."""
    env = os.environ.get("PYUL_INTERACTIVE", "").strip().lower()
    if env in _TRUE:
        return True
    if env in _FALSE:
        return False
    try:
        return bool(sys.stdin is not None and sys.stdin.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def ask():
    """Call ``input()`` and return the answer, or None on EOF / closed stdin."""
    try:
        return str(input())
    except (EOFError, OSError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# Progress output mode
# ---------------------------------------------------------------------------

PROGRESS_MODES = ("bar", "lines", "quiet")


def resolve_progress_mode(stream=None):
    """``PYUL_PROGRESS=bar|lines|quiet`` overrides; otherwise 'bar' on a TTY or
    in a Jupyter kernel, 'lines' (one compact line per snapshot) elsewhere."""
    env = os.environ.get("PYUL_PROGRESS", "").strip().lower()
    if env in PROGRESS_MODES:
        return env
    if stream is None:
        stream = sys.stdout
    try:
        if stream is not None and stream.isatty():
            return "bar"
    except (AttributeError, ValueError, OSError):
        pass
    if in_ipython_kernel():
        return "bar"
    return "lines"


# ---------------------------------------------------------------------------
# Optional third-party modules
# ---------------------------------------------------------------------------

def clear_output(wait=False):
    """IPython's clear_output when IPython is installed, otherwise a no-op.

    IPython is imported on first use only (it is not needed for batch runs;
    IPython >= 9 removed the old IPython.core.display.clear_output)."""
    try:
        from IPython.display import clear_output as _clear_output
    except Exception:
        return None
    return _clear_output(wait=wait)

NUMBA_IMPORT_ERROR = None
try:
    import numba  # noqa: F401
except Exception as _err:  # pragma: no cover - depends on the environment
    numba = None
    NUMBA_IMPORT_ERROR = f"{type(_err).__name__}: {_err}"


def jit(func):
    """``numba.jit(func)`` when numba is importable, else ``func`` unchanged.

    numba is only used for two helpers (initsoliton for embedded solitons and
    LUQuick for Momentum/AngMomentum saves); the default soliton path and the
    time evolution never call jitted code.
    """
    if numba is None:
        return func
    return numba.jit(func)


H5PY_IMPORT_ERROR = None
_h5py = None


def get_h5py():
    """Import h5py on first use (only needed for save_format='hdf5')."""
    global _h5py, H5PY_IMPORT_ERROR
    if _h5py is None:
        try:
            import h5py as _mod
        except Exception as err:
            H5PY_IMPORT_ERROR = f"{type(err).__name__}: {err}"
            raise ImportError(
                "save_format='hdf5' (and the hdf5 loaders) need the optional package h5py; "
                "install it with 'pip install h5py' or use save_format='npy'. "
                f"Import failed with {H5PY_IMPORT_ERROR}") from err
        _h5py = _mod
    return _h5py


# ---------------------------------------------------------------------------
# SciPy compatibility
# ---------------------------------------------------------------------------

def simpson(y, x=None, dx=1.0, axis=-1):
    """Composite Simpson rule with the legacy ``simps(y, x)`` call semantics.

    ``scipy.integrate.simps`` was removed in SciPy 1.14.  In SciPy 1.11-1.13
    ``simps`` is a thin alias that calls ``simpson(y, x=x, dx=dx, axis=axis)``,
    so this wrapper is identical to the legacy call on those versions.
    """
    import scipy.integrate as _si
    fn = getattr(_si, "simpson", None)
    if fn is None:  # SciPy < 1.6
        return _si.simps(y, x=x, dx=dx, axis=axis)
    return fn(y, x=x, dx=dx, axis=axis)


def _resolve_sph_harm():
    import scipy.special as _ss
    if hasattr(_ss, "sph_harm_y"):
        new = _ss.sph_harm_y

        def sph_harm(m, n, theta, phi):
            # legacy: theta = azimuth, phi = polar; new: sph_harm_y(n, m, polar, azimuth)
            return new(n, m, phi, theta)
        sph_harm.__doc__ = ("Legacy scipy.special.sph_harm(m, n, theta_azimuth, phi_polar) "
                            "implemented with scipy.special.sph_harm_y(n, m, phi_polar, theta_azimuth).")
        return sph_harm, "sph_harm_y"
    return _ss.sph_harm, "sph_harm"


def sph_harm(m, n, theta, phi):
    """Spherical harmonic with the LEGACY SciPy signature
    ``sph_harm(m, n, theta_azimuth, phi_polar)`` on every SciPy version."""
    global _SPH_IMPL
    if _SPH_IMPL is None:
        _SPH_IMPL = _resolve_sph_harm()[0]
    return _SPH_IMPL(m, n, theta, phi)


_SPH_IMPL = None


# ---------------------------------------------------------------------------
# Thread count
# ---------------------------------------------------------------------------

THREAD_ENV_ORDER = ("PYUL_NUM_THREADS", "OMP_NUM_THREADS", "SLURM_CPUS_PER_TASK", "NCPUS")


def _parse_positive_int(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    # OMP_NUM_THREADS may be a nested list such as "8,2": use the outer level.
    value = value.split(",")[0].strip()
    try:
        n = int(value)
    except ValueError:
        return None
    return n if n > 0 else None


def detect_cpu_threads():
    """(n, source) from the environment / scheduler / affinity / cpu_count."""
    for name in THREAD_ENV_ORDER:
        n = _parse_positive_int(os.environ.get(name))
        if n is not None:
            return n, name
    if hasattr(os, "sched_getaffinity"):
        try:
            n = len(os.sched_getaffinity(0))
            if n > 0:
                return n, "sched_getaffinity"
        except OSError:
            pass
    return (os.cpu_count() or 1), "cpu_count"


def resolve_num_threads(resol=None, small_grid_limit=128, small_grid_cap=4):
    """Thread count for pyFFTW, numexpr and scipy.fft.

    Order: PYUL_NUM_THREADS > OMP_NUM_THREADS > SLURM_CPUS_PER_TASK >
    NCPUS (PBS) > len(os.sched_getaffinity(0)) > os.cpu_count().
    As in 30.40, grids with resol < 128 are capped at 4 threads, unless the
    count was requested explicitly with PYUL_NUM_THREADS.
    Returns (n, source_description).
    """
    n, source = detect_cpu_threads()
    if resol is not None and resol < small_grid_limit and source != "PYUL_NUM_THREADS":
        if n > small_grid_cap:
            n = small_grid_cap
            source = f"{source}, capped at {small_grid_cap} for resol<{small_grid_limit}"
    return int(n), source


# ---------------------------------------------------------------------------
# FFT backends
# ---------------------------------------------------------------------------

class _ScipyPlan:
    """Callable that mimics the part of a ``pyfftw.FFTW`` object PyUL uses:
    ``plan(array)`` returns the transform of ``array`` along the planned axes.

    Unlike pyFFTW (which returns its internal output buffer on every call) a
    new array is returned each time; every PyUL call site rebinds the result
    before the next call of the same plan, so both behave identically.
    """

    def __init__(self, func, template, axes, threads):
        self._func = func
        self._shape = np.shape(template)
        self.axes = tuple(axes) if axes is not None else None
        self.threads = int(threads) if threads else 1

    def __call__(self, input_array=None, *args, **kwargs):
        if input_array is None:
            raise ValueError("scipy FFT backend plans must be called with an input array")
        return self._func(np.asarray(input_array), axes=self.axes, workers=self.threads)


class _ScipyBuilders:
    """Subset of ``pyfftw.builders`` (fftn, ifftn, rfftn, irfftn)."""

    def __init__(self):
        import scipy.fft as _sfft
        self._sfft = _sfft

    def fftn(self, a, axes=None, threads=1, **kwargs):
        return _ScipyPlan(self._sfft.fftn, a, axes, threads)

    def ifftn(self, a, axes=None, threads=1, **kwargs):
        return _ScipyPlan(self._sfft.ifftn, a, axes, threads)

    def rfftn(self, a, axes=None, threads=1, **kwargs):
        return _ScipyPlan(self._sfft.rfftn, a, axes, threads)

    def irfftn(self, a, axes=None, threads=1, **kwargs):
        return _ScipyPlan(self._sfft.irfftn, a, axes, threads)


class FFTBackend:
    def __init__(self, name, builders, zeros_aligned, version):
        self.name = name
        self.builders = builders
        self.zeros_aligned = zeros_aligned
        self.version = version

    def __repr__(self):
        return f"FFTBackend({self.name} {self.version})"


PYFFTW_IMPORT_ERROR = None
try:
    import pyfftw as _pyfftw
    import pyfftw.builders  # noqa: F401
except Exception as _err:  # pragma: no cover - depends on the environment
    _pyfftw = None
    PYFFTW_IMPORT_ERROR = f"{type(_err).__name__}: {_err}"


def _np_zeros_aligned(shape, dtype="float64", n=None):
    return np.zeros(shape, dtype=dtype)


def resolve_fft_backend(name=None):
    """Return the FFT backend: 'pyfftw' (default when importable) or 'scipy'.

    ``PYUL_FFT_BACKEND=pyfftw|scipy`` selects explicitly.  Requesting pyfftw
    when it cannot be imported raises ImportError with the original reason.
    """
    if name is None:
        name = os.environ.get("PYUL_FFT_BACKEND", "").strip().lower() or None
    if name not in (None, "pyfftw", "scipy"):
        raise ValueError(f"PYUL_FFT_BACKEND must be 'pyfftw' or 'scipy', not {name!r}")
    if name in (None, "pyfftw") and _pyfftw is not None:
        return FFTBackend("pyfftw", _pyfftw.builders, _pyfftw.zeros_aligned,
                          getattr(_pyfftw, "__version__", "?"))
    if name == "pyfftw":
        raise ImportError(f"PYUL_FFT_BACKEND=pyfftw but pyfftw cannot be imported ({PYFFTW_IMPORT_ERROR})")
    import scipy
    return FFTBackend("scipy", _ScipyBuilders(), _np_zeros_aligned, scipy.__version__)


def zeros_aligned(shape, dtype="float64"):
    """``pyfftw.zeros_aligned`` when pyfftw is importable, else ``numpy.zeros``."""
    if _pyfftw is not None:
        return _pyfftw.zeros_aligned(shape, dtype=dtype)
    return np.zeros(shape, dtype=dtype)


# ---------------------------------------------------------------------------
# Resource and cache paths
# ---------------------------------------------------------------------------

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PACKAGE_DIR)


def _dedup(paths):
    out, seen = [], set()
    for p in paths:
        if not p:
            continue
        key = os.path.normcase(os.path.abspath(p))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def user_cache_dir():
    """Per-user cache directory for PyUL (Green functions, custom profiles)."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Local")
        return os.path.join(base, "PyUltraLight2", "Cache")
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "PyUltraLight2")


def data_search_dirs():
    """Directories searched for shipped data ('Soliton Profile Files', ...):
    PYUL_DATA_DIR, then the working directory (legacy behaviour), then the
    repository root next to the package."""
    return _dedup([os.environ.get("PYUL_DATA_DIR"), os.getcwd(), REPO_ROOT])


def cache_dirs():
    """Writable cache roots in priority order: PYUL_CACHE_DIR, then the user cache."""
    return _dedup([os.environ.get("PYUL_CACHE_DIR"), user_cache_dir()])


def find_resource(relpath, include_cache=False):
    """First existing ``<dir>/<relpath>`` in the data search order, or None."""
    dirs = data_search_dirs() + (cache_dirs() if include_cache else [])
    for d in dirs:
        p = os.path.join(d, relpath)
        if os.path.exists(p):
            return p
    return None


def require_resource(relpath, include_cache=False):
    p = find_resource(relpath, include_cache=include_cache)
    if p is None:
        dirs = data_search_dirs() + (cache_dirs() if include_cache else [])
        raise FileNotFoundError(
            f"PyUL data file '{relpath}' not found. Searched: " + "; ".join(dirs)
            + ". Set PYUL_DATA_DIR to the folder that contains 'Soliton Profile Files'.")
    return p


def _dir_writable(d):
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        return False
    return os.access(d, os.W_OK)


def green_function_read_candidates(resol):
    """Locations searched for a cached IsoP Green function G{resol}.npy."""
    name = f"G{resol}.npy"
    cands = []
    env_cache = os.environ.get("PYUL_CACHE_DIR")
    if env_cache:
        cands.append(os.path.join(env_cache, "Green Functions", name))
    for d in data_search_dirs():
        cands.append(os.path.join(d, "Green Functions", name))
    cands.append(os.path.join(user_cache_dir(), "Green Functions", name))
    return _dedup(cands)


def repo_checkout_dir():
    """REPO_ROOT if the package runs from a source checkout (editable install
    or PYTHONPATH), else None.

    A checkout is recognised by the files that ship next to the package:
    pyproject.toml and the 'Green Functions' folder.  For a wheel install
    REPO_ROOT is site-packages, which has neither."""
    if (os.path.isfile(os.path.join(REPO_ROOT, "pyproject.toml"))
            and os.path.isdir(os.path.join(REPO_ROOT, "Green Functions"))):
        return REPO_ROOT
    return None


def green_function_write_path(resol):
    """Where a newly computed Green function is cached:
    PYUL_CACHE_DIR/Green Functions if set, else the existing 'Green Functions'
    folder of a source checkout if writable, else the user cache.

    The checkout folder is never created: for a non-editable install
    (REPO_ROOT = site-packages) the cache goes to the user cache, not into
    site-packages."""
    name = f"G{resol}.npy"
    env_cache = os.environ.get("PYUL_CACHE_DIR")
    if env_cache:
        d = os.path.join(env_cache, "Green Functions")
        if _dir_writable(d):
            return os.path.join(d, name)
    repo = repo_checkout_dir()
    if repo is not None:
        d = os.path.join(repo, "Green Functions")
        if os.path.isdir(d) and os.access(d, os.W_OK):
            return os.path.join(d, name)
    d = os.path.join(user_cache_dir(), "Green Functions")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def _self_and_ancestors(path):
    p = os.path.realpath(os.path.abspath(path))
    out = [p]
    while True:
        parent = os.path.dirname(p)
        if parent == p:
            return out
        out.append(parent)
        p = parent


def check_safe_rmtree(path, what="folder"):
    """Refuse (ValueError) a recursive delete of ``path`` when it resolves to
    the filesystem root, the home directory, the current working directory or
    one of its ancestors, or the repository root (REPO_ROOT) or one of its
    ancestors.  Returns the resolved path otherwise.

    A cheap guard against a mistyped save_path ('.', '~', '/', '..') in
    DSManagement(Force=True) and the other deletions of user paths (30.50)."""
    real = os.path.realpath(os.path.abspath(os.path.expanduser(os.fspath(path))))
    key = os.path.normcase(real)

    def same(a):
        return os.path.normcase(os.path.realpath(a)) == key

    reasons = []
    if os.path.dirname(real) == real:
        reasons.append("the filesystem root")
    if same(os.path.expanduser("~")):
        reasons.append("the home directory")
    if any(same(a) for a in _self_and_ancestors(os.getcwd())):
        reasons.append("the current working directory or one of its ancestors")
    if any(same(a) for a in _self_and_ancestors(REPO_ROOT)):
        reasons.append("the PyUL repository root or one of its ancestors")
    if reasons:
        raise ValueError(
            f"PyUL: refusing to delete the {what} {path!r} (resolves to {real}): it is "
            + " and ".join(reasons) + ". Choose a dedicated save folder.")
    return real


CUSTOM_PROFILE_REL = os.path.join("Soliton Profile Files", "Custom")


def custom_profile_dirs():
    """Folders searched for embedded-soliton profiles (f_<ratio>_Pro/Dra)."""
    dirs = [os.path.join(d, CUSTOM_PROFILE_REL) for d in data_search_dirs()]
    dirs += [os.path.join(d, CUSTOM_PROFILE_REL) for d in cache_dirs()]
    return _dedup(dirs)


def custom_profile_write_dir():
    """First existing, writable custom-profile folder (legacy: ./Soliton Profile
    Files/Custom/), otherwise a cache folder that is created on demand."""
    for d in [os.path.join(x, CUSTOM_PROFILE_REL) for x in data_search_dirs()]:
        if os.path.isdir(d) and os.access(d, os.W_OK):
            return d
    for d in [os.path.join(x, CUSTOM_PROFILE_REL) for x in cache_dirs()]:
        if _dir_writable(d):
            return d
    raise OSError("No writable folder for custom soliton profiles; set PYUL_CACHE_DIR.")


def atomic_save_npy(path, array):
    """np.save to a temporary file in the same folder, then os.replace: readers
    never see a truncated file and concurrent writers cannot corrupt it."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    tmp = os.path.join(folder, f".{os.path.basename(path)}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "wb") as fh:
            np.save(fh, array)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def file_sha256(path, normalise_newlines=False):
    import hashlib
    with open(path, "rb") as fh:
        data = fh.read()
    if normalise_newlines:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def library_versions():
    import platform
    out = {"Python": platform.python_version(), "Platform": platform.platform(),
           "NumPy": np.__version__}
    try:
        from importlib import metadata as _md
    except ImportError:  # pragma: no cover
        _md = None
    for mod in ("scipy", "numexpr", "pyfftw", "numba", "h5py"):
        m = sys.modules.get(mod)
        ver = getattr(m, "__version__", None) if m is not None else None
        if ver is None and _md is not None:
            try:
                ver = _md.version(mod if mod != "pyfftw" else "pyFFTW")
            except Exception:
                ver = None
        out[mod] = ver
    return out
