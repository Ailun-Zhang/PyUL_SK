#!/usr/bin/env bash
# PyUltraLight 2 performance benchmark (runs Benchmark.py from the repository root).
# Usage: ./Bench.sh [--resol 64|256|384|512|1024] [--yes]
# Set PYTHON to choose the interpreter (default: python3) and PYUL_NUM_THREADS
# to fix the FFT thread count.
set -euo pipefail
cd "$(dirname "$0")"
exec "${PYTHON:-python3}" Benchmark.py "$@"
