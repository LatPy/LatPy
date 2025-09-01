import ctypes
import importlib.resources
import sys
import pathlib

def _load_shared_lib():
    package = __package__  # "latpy"
    files = importlib.resources.files(package)

    # _latpy.*.so を探す
    candidates = [p for p in files.iterdir() if p.name.startswith("_latpy") and p.suffix == ".so"]
    if not candidates:
        raise ImportError(f"Shared library for _latpy not found in package {package}")
    return ctypes.CDLL(str(candidates[0]))

lib = _load_shared_lib()
