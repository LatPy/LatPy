from .loader import lib
import numpy as np
import math
import ctypes

def lagrange(basis: np.ndarray[int]) -> np.ndarray[int]:
    """Performs Lagrange reduction on a 2D basis.

    Args:
        basis (np.ndarray[int]): The input 2D basis vectors.

    Returns:
        np.ndarray[int]: The Lagrange reduced basis.
    """
    if basis.shape[0] != 2:
        raise ValueError("Lagrange reduction is only defined for 2D bases.")

    n, m = basis.shape

    lib.lagrange.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.c_long,
        ctypes.c_long
    )
    lib.lagrange.restype = None

    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = basis[i].ctypes.data_as(ctypes.POINTER(ctypes.c_long))

    lib.lagrange(basis_ptr, n, m)

    reduced_basis = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            reduced_basis[i, j] = basis_ptr[i][j]

    return reduced_basis
