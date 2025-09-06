from .loader import lib
import numpy as np
import pandas as pd
import ctypes
import os

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
        basis_ptr[i] = (ctypes.c_long * m)()
        for j in range(m):
            basis_ptr[i][j] = ctypes.c_long(basis[i, j])

    lib.lagrange(basis_ptr, n, m)

    reduced_basis = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            reduced_basis[i, j] = basis_ptr[i][j]

    return reduced_basis

def size(basis: np.ndarray[int], eta: float) -> np.ndarray[int]:
    """Performs size reduction on a basis with a given eta parameter.

    Args:
        basis (np.ndarray[int]): The input basis vectors.
        eta (float): The eta parameter for size reduction.

    Returns:
        np.ndarray[int]: The size reduced basis.
    """
    if eta < 0.5 or eta > 1.0:
        raise ValueError("Eta must be in the range [0.5, 1.0].")
    
    n, m = basis.shape

    lib.size.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.c_double,
        ctypes.c_long,
        ctypes.c_long
    )
    lib.size.restype = None

    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = (ctypes.c_long * m)()
        for j in range(m):
            basis_ptr[i][j] = ctypes.c_long(basis[i, j])

    lib.size(basis_ptr, ctypes.c_double(eta), n, m)

    reduced_basis = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            reduced_basis[i, j] = basis_ptr[i][j]

    return reduced_basis

def lll(basis: np.ndarray[int], delta: float = 0.99, eta: float = 0.55, output_sl_log: bool = False, output_rhf_log: bool = False) -> tuple[np.ndarray[int], list[float], list[float]]:
    """Performs LLL reduction on a basis with given delta and eta parameters.

    Args:
        basis (np.ndarray[int]): The input basis vectors.
        delta (float, optional): The delta parameter for LLL reduction. Defaults to 0.99.
        eta (float, optional): The eta parameter for size reduction within LLL. Defaults to 0.55.
        output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
        output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.

    Returns:
        np.ndarray[int]: The LLL reduced basis.
    """
    if delta <= 0.25 or delta >= 1:
        raise ValueError("Delta must be in the range (0.25, 1).")
    if eta < 0.5 or eta > np.sqrt(delta):
        raise ValueError("Eta must be in the range [0.5, sqrt(delta)].")
    
    n, m = basis.shape
    
    lib.LLL.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_long,
        ctypes.c_long
    )
    lib.LLL.restype = None

    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = (ctypes.c_long * m)()
        for j in range(m):
            basis_ptr[i][j] = ctypes.c_long(basis[i, j])

    lib.LLL(basis_ptr, ctypes.c_double(delta), ctypes.c_double(eta), output_sl_log, output_rhf_log, n, m)

    reduced_basis = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            reduced_basis[i, j] = basis_ptr[i][j]

    sl_log = []
    rhf_log = []
    if output_sl_log:
        if os.path.exists("sl_log.csv"):
            sl_log = list(pd.read_csv("sl_log.csv")["val"])
            os.remove("sl_log.csv")
    if output_rhf_log:
        if os.path.exists("rhf_log.csv"):
            rhf_log = list(pd.read_csv("rhf_log.csv")["val"])
            os.remove("rhf_log.csv")
    
    return reduced_basis, sl_log, rhf_log

def deep_lll(basis: np.ndarray[int], delta: float = 0.99, eta: float = 0.55, gamma: int = 20, output_sl_log: bool = False, output_rhf_log: bool = False) -> tuple[np.ndarray[int], list[float], list[float]]:
    """Performs Deep LLL reduction on a basis with given delta and eta parameters.

    Args:
        basis (np.ndarray[int]): The input basis vectors.
        delta (float, optional): The delta parameter for Deep LLL reduction. Defaults to 0.99.
        eta (float, optional): The eta parameter for size reduction within Deep LLL. Defaults to 0.55.
        gamma (int, optional): The gamma parameter for Deep LLL reduction. Defaults to 20.
        output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
        output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.

    Returns:
        np.ndarray[int]: The Deep LLL reduced basis.
    """
    if delta <= 0.25 or delta >= 1:
        raise ValueError("Delta must be in the range (0.25, 1).")
    if eta < 0.5 or eta > np.sqrt(delta):
        raise ValueError("Eta must be in the range [0.5, sqrt(delta)].")
    
    n, m = basis.shape

    lib.deepLLL.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_long,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_long,
        ctypes.c_long
    )
    lib.deepLLL.restype = None

    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = (ctypes.c_long * m)()
        for j in range(m):
            basis_ptr[i][j] = ctypes.c_long(basis[i, j])

    lib.deepLLL(basis_ptr, ctypes.c_double(delta), ctypes.c_double(eta), ctypes.c_long(gamma), output_sl_log, output_rhf_log, n, m)

    reduced_basis = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            reduced_basis[i, j] = basis_ptr[i][j]

    sl_log = []
    rhf_log = []
    if output_sl_log:
        if os.path.exists("sl_log.csv"):
            sl_log = list(pd.read_csv("sl_log.csv")["val"])
            os.remove("sl_log.csv")
    if output_rhf_log:
        if os.path.exists("rhf_log.csv"):
            rhf_log = list(pd.read_csv("rhf_log.csv")["val"])
            os.remove("rhf_log.csv")
    
    return reduced_basis, sl_log, rhf_log

def l2(basis: np.ndarray[int], delta: float = 0.99, eta: float = 0.55, output_sl_log: bool = False, output_rhf_log: bool = False) -> np.ndarray[int]:
    """Performs L2 reduction on a basis with given delta and eta parameters.

    Args:
        basis (np.ndarray[int]): The input basis vectors.
        delta (float, optional): The delta parameter for L2 reduction. Defaults to 0.99.
        eta (float, optional): The eta parameter for size reduction within L2. Defaults to 0.55.
        output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
        output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.

    Returns:
        np.ndarray[int]: The L2 reduced basis.
    """
    if delta <= 0.25 or delta >= 1:
        raise ValueError("Delta must be in the range (0.25, 1).")
    if eta < 0.5 or eta > np.sqrt(delta):
        raise ValueError("Eta must be in the range [0.5, sqrt(delta)].")
    
    n, m = basis.shape

    lib.L2.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_long,
        ctypes.c_long
    )
    lib.L2.restype = None

    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = (ctypes.c_long * m)()
        for j in range(m):
            basis_ptr[i][j] = ctypes.c_long(basis[i, j])

    lib.L2(basis_ptr, ctypes.c_double(delta), ctypes.c_double(eta), output_sl_log, output_rhf_log, n, m)

    reduced_basis = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            reduced_basis[i, j] = basis_ptr[i][j]

    sl_log = []
    rhf_log = []
    if output_sl_log:
        if os.path.exists("sl_log.csv"):
            sl_log = list(pd.read_csv("sl_log.csv")["val"])
            os.remove("sl_log.csv")
    if output_rhf_log:
        if os.path.exists("rhf_log.csv"):
            rhf_log = list(pd.read_csv("rhf_log.csv")["val"])
            os.remove("rhf_log.csv")

    return reduced_basis, sl_log, rhf_log

def bkz(basis: np.ndarray[int], delta: float = 0.99, beta: int = 20, max_loops: int = -1, pruning: bool = False, output_sl_log: bool = False, output_rhf_log: bool = False) -> tuple[np.ndarray[int], list[float], list[float]]:
    """Performs BKZ reduction on a basis with given delta and beta parameters.

    Args:
        basis (np.ndarray[int]): The input basis vectors.
        delta (float, optional): The delta parameter for BKZ reduction. Defaults to 0.99.
        beta (int, optional): The block size parameter for BKZ reduction. Defaults to 20.
        max_loops (int, optional): The maximum number of tours through the basis. Defaults to -1 (no limit).
        pruning (bool, optional): Whether to use pruning in the SVP solver. Defaults to False.
        output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
        output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.

    Returns:
        np.ndarray[int]: The BKZ reduced basis.
    """
    if delta <= 0.25 or delta >= 1:
        raise ValueError("Delta must be in the range (0.25, 1).")
    if beta < 2:
        raise ValueError("Beta must be at least 2.")
    if max_loops < -1:
        raise ValueError("max_loops must be -1 (no limit) or a non-negative integer.")
    
    n, m = basis.shape

    lib.BKZ.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.c_double,
        ctypes.c_long,
        ctypes.c_long,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_long,
        ctypes.c_long
    )
    lib.BKZ.restype = None

    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = (ctypes.c_long * m)()
        for j in range(m):
            basis_ptr[i][j] = ctypes.c_long(basis[i, j])

    lib.BKZ(basis_ptr, ctypes.c_double(delta), ctypes.c_long(beta), ctypes.c_long(max_loops), ctypes.c_bool(pruning), ctypes.c_bool(output_sl_log), ctypes.c_bool(output_rhf_log), n, m)

    reduced_basis = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            reduced_basis[i, j] = basis_ptr[i][j]

    sl_log = []
    rhf_log = []
    if output_sl_log:
        if os.path.exists("sl_log.csv"):
            sl_log = list(pd.read_csv("sl_log.csv")["val"])
            os.remove("sl_log.csv")
    if output_rhf_log:
        if os.path.exists("rhf_log.csv"):
            rhf_log = list(pd.read_csv("rhf_log.csv")["val"])
            os.remove("rhf_log.csv")

    return reduced_basis, sl_log, rhf_log
