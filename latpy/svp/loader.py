import ctypes
import importlib.resources

def _load_shared_lib():
    package = __package__  # "svp"
    files = importlib.resources.files(package)

    # _svp.*.so を探す
    candidates = [p for p in files.iterdir() if p.name.startswith("_svp") and p.suffix == ".so"]
    if not candidates:
        raise ImportError(f"Shared library for _svp not found in package {package}")
    return ctypes.CDLL(str(candidates[0]))

lib = _load_shared_lib()
