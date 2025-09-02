import ctypes
import importlib.resources

def _load_shared_lib():
    package = __package__  # "core"
    files = importlib.resources.files(package)

    # _core.*.so を探す
    candidates = [p for p in files.iterdir() if p.name.startswith("_core") and p.suffix == ".so"]
    if not candidates:
        raise ImportError(f"Shared library for _core not found in package {package}")
    return ctypes.CDLL(str(candidates[0]))

lib = _load_shared_lib()
