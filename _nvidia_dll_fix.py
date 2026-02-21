"""
_nvidia_dll_fix.py — Ensure NVIDIA cuDNN / cuBLAS DLLs are discoverable.

On Windows, pip-installed nvidia-cudnn-cu12 / nvidia-cublas-cu12 place DLLs
under site-packages/nvidia/*/bin/.  ONNX Runtime (and InsightFace) won't find
them unless they're on PATH or registered via os.add_dll_directory.

Import this module **before** importing insightface or onnxruntime:

    import _nvidia_dll_fix   # noqa: F401  (side-effect import)
    from insightface.app import FaceAnalysis
"""

import os
import sys
import site
import glob

def _register_nvidia_dlls() -> None:
    """Add nvidia pip-package bin dirs to DLL search path (Windows only)."""
    if sys.platform != "win32":
        return

    # Find site-packages directory
    sp_dirs = site.getsitepackages() + [site.getusersitepackages()]
    for sp in sp_dirs:
        nvidia_root = os.path.join(sp, "nvidia")
        if not os.path.isdir(nvidia_root):
            continue

        # Look for bin/ dirs containing DLLs (cudnn, cublas, cuda_runtime, etc.)
        for bin_dir in glob.glob(os.path.join(nvidia_root, "*", "bin")):
            if os.path.isdir(bin_dir):
                try:
                    os.add_dll_directory(bin_dir)
                except OSError:
                    pass
                # Also prepend to PATH as fallback for older Python / subprocess
                current = os.environ.get("PATH", "")
                if bin_dir not in current:
                    os.environ["PATH"] = bin_dir + ";" + current


_register_nvidia_dlls()
