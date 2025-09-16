from .loader import lib
import numpy as np
import math
import ctypes

def enum_sv(basis: np.ndarray[int], pruning: bool = False, alg: str = "gs") -> np.ndarray[int]:
    """Enumerates the shortest vector in a lattice basis using the SVP algorithm.

    Args:
        basis (np.ndarray[int]): The input lattice basis.
        pruning (bool, optional): Whether to use pruning. Defaults to False.
        alg (str, optional): The algorithm to use ('gs' for Gram-Schmidt, 'qr' for QR decomposition). Defaults to "gs".

    Returns:
        np.ndarray[int]: The shortest vector found in the lattice.
    """
    n, m = basis.shape

    lib.enumSV.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.POINTER(ctypes.c_long),                  # result
        ctypes.c_bool,                                  # pruning
        ctypes.c_long,                                  # n
        ctypes.c_long,                                  # m
    )
    lib.enumSV.restype = None
    lib.qrEnumSV.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.POINTER(ctypes.c_long),                  # result
        ctypes.c_bool,                                  # pruning
        ctypes.c_long,                                  # n
        ctypes.c_long,                                  # m
    )
    lib.qrEnumSV.restype = None

    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = (ctypes.c_long * m)()
        for j in range(m):
            basis_ptr[i][j] = ctypes.c_long(basis[i, j])

    coeff_ptr = (ctypes.c_long * n)()
    if alg == "gs":
        lib.enumSV(basis_ptr, coeff_ptr, ctypes.c_bool(pruning), n, m)
    elif alg == "qr":
        lib.qrEnumSV(basis_ptr, coeff_ptr, ctypes.c_bool(pruning), n, m)
    coeff = coeff_ptr[:]

    return np.array(coeff) @ basis
