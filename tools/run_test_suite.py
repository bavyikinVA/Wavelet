"""Reproducible test entry point for Wavelets Analysis Studio.

From the project root:
    python tools/run_test_suite.py cpu
    python tools/run_test_suite.py gpu
    python tools/run_test_suite.py gui
    python tools/run_test_suite.py all

CPU does not need customtkinter or an available display. GPU and GUI runs
are explicitly gated instead of being reported as passing through skip-only
collections. Run `all` in a fully provisioned Windows 11 / Python 3.12 env.
"""
from __future__ import annotations

import argparse
import importlib.util
import platform
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# These tests depend on CustomTkinter and/or a window system, even when they
# also check CPU fallback or data handling.
GUI_TESTS = (
    "test_animation.py",
    "test_tk_lifecycle.py",
    "test_ml_run_report_gui.py",
    "test_progress_completion.py",
    "test_result_viewer.py",
    "test_results_integration.py",
    "test_run_control.py",
    "test_scrollable_dropdown.py",
    "test_cpu_without_cupy.py",
    "test_startup.py",
    "test_viewport.py",
)

# CPU suite also deselects GPU-only classes in mixed test files. Otherwise
# a GPU-equipped PC runs CUDA tests as part of the mandatory CPU suite, while
# a CPU-only PC silently skips those same cases. Keep results reproducible.
GPU_TESTS = (
    "test_gpu_extrema_envelope_parity.py",
    "test_knn_gpu_self_neighbor.py",
    "test_knn_gpu_spatial_grid.py",
    "test_backend_reproducibility.py",
    "test_morlet_2d.py",
)
GPU_ONLY_TESTS = GPU_TESTS[:3]
GPU_MIXED_NODES = (
    "tests/test_envelope_support.py::GPUEnvelopeSupportTests",
    "tests/test_backend_reproducibility.py::CPUGPURegressionTests",
    "tests/test_scientific_signals.py::ScientificGPUTests",
)


def _gpu_available() -> tuple[bool, str]:
    try:
        import cupy as cp
        count = cp.cuda.runtime.getDeviceCount()
        if count <= 0:
            return False, "CUDA devices not found"
        device = cp.asarray([1], dtype=cp.float32)
        cp.cuda.Stream.null.synchronize()
        return bool(device.size), f"CuPy {cp.__version__}, CUDA devices={count}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _gui_available() -> tuple[bool, str]:
    if importlib.util.find_spec("customtkinter") is None:
        return False, "customtkinter is missing (see requirements.txt)"
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        root.destroy()
    except Exception as exc:
        return False, f"Tk/display unavailable: {type(exc).__name__}: {exc}"
    return True, "customtkinter and Tk/display available"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("cpu", "gpu", "gui", "all"))
    args, extra = parser.parse_known_args()

    print(f"OS={platform.platform()} | Python={sys.version.split()[0]}", flush=True)
    try:
        import numpy as np
        print(f"NumPy={np.__version__}", flush=True)
    except ImportError:
        print("NumPy=missing", flush=True)
    if sys.version_info[:2] != (3, 12):
        print("WARNING: project's target is Python 3.12; results from this host are provisional.", flush=True)

    if args.suite in {"gpu", "all"}:
        ok, note = _gpu_available()
        print("GPU preflight:", note, flush=True)
        if not ok:
            print("GPU suite NOT RUN (exit code 2, not a passing skip).", flush=True)
            return 2
    if args.suite in {"gui", "all"}:
        ok, note = _gui_available()
        print("GUI preflight:", note, flush=True)
        if not ok:
            print("GUI suite NOT RUN (exit code 2, not a passing skip).", flush=True)
            return 2

    cmd = [sys.executable, "-m", "pytest", "-q"]
    if args.suite in {"gui", "all"}:
        # Windows Tcl may retain standard handles across Tk interpreters;
        # pytest fd capture closes/replaces them between tests. Python-level
        # capture preserves diagnostics without that handle churn.
        cmd.append("--capture=sys")
    if args.suite == "cpu":
        for name in (*GUI_TESTS, *GPU_ONLY_TESTS):
            cmd.append(f"--ignore=tests/{name}")
        for node in GPU_MIXED_NODES:
            cmd.append(f"--deselect={node}")
    elif args.suite == "gpu":
        cmd.extend(f"tests/{name}" for name in GPU_TESTS)
        cmd.extend(GPU_MIXED_NODES)
    elif args.suite == "gui":
        cmd.extend(f"tests/{name}" for name in GUI_TESTS)
    if args.suite in {"cpu", "all"}:
        cmd.append("tests")
    cmd.extend(extra)
    print("Test suite:", args.suite, "| command:", " ".join(cmd), flush=True)
    # Tcl stderr can report an error while pytest still returns zero (including
    # subprocess-based GUI tests). Keep diagnostics visible and reject that run.
    callback_error = False
    with subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace",
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"}) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            if 'invalid command name' in line or '("after" script)' in line:
                callback_error = True
        status = process.wait()
    if callback_error:
        print("GUI callback errors detected: suite NOT accepted.", flush=True)
        return status or 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
