from .loader import lib
import numpy as np
import ctypes

def compute_gso(basis):
    n, m = basis.shape

    lib.computeGSO.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # mu
        ctypes.POINTER(ctypes.c_double),  # B
        ctypes.c_long,
        ctypes.c_long
    )
    lib.computeGSO.restype = None

    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = basis[i].ctypes.data_as(ctypes.POINTER(ctypes.c_long))

    mu = np.zeros((n, n), dtype=np.float64)
    mu_ptr = (ctypes.POINTER(ctypes.c_double) * n)()
    for i in range(n):
        mu_ptr[i] = mu[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    B = np.zeros(n, dtype=np.float64)
    B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    lib.computeGSO(basis_ptr, mu_ptr, B_ptr, n, m)

    return mu, B

def volume(basis):
    n, m = basis.shape

    lib.volume.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.c_long,
        ctypes.c_long
    )
    lib.volume.restype = ctypes.c_long

    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = basis[i].ctypes.data_as(ctypes.POINTER(ctypes.c_long))

    return lib.volume(basis_ptr, n, m)

def sl(basis):
    n, m = basis.shape

    lib.sl.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.c_long,
        ctypes.c_long
    )
    lib.sl.restype = ctypes.c_longdouble

    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = basis[i].ctypes.data_as(ctypes.POINTER(ctypes.c_long))

    return lib.sl(basis_ptr, n, m)
