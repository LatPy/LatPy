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

def seysen(basis: np.ndarray[int]) -> np.ndarray[int]:
    """Performs Seysen reduction on a basis.

    Args:
        basis (np.ndarray[int]): The input basis vectors.

    Returns:
        np.ndarray[int]: The Seysen reduced basis.
    """
    n, m = basis.shape

    lib.seysen.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.c_long,
        ctypes.c_long
    )
    lib.seysen.restype = None

    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = (ctypes.c_long * m)()
        for j in range(m):
            basis_ptr[i][j] = ctypes.c_long(basis[i, j])

    lib.seysen(basis_ptr, n, m)

    reduced_basis = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            reduced_basis[i, j] = basis_ptr[i][j]

    return reduced_basis

def lll(basis: np.ndarray[int], delta: float = 0.99, eta: float = 0.5, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[np.ndarray[int], list[float], list[float], float]:
    """Performs LLL reduction on a basis with given delta and eta parameters.

    Args:
        basis (np.ndarray[int]): The input basis vectors.
        delta (float, optional): The delta parameter for LLL reduction. Defaults to 0.99.
        eta (float, optional): The eta parameter for size reduction within LLL. Defaults to 0.5.
        output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
        output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
        output_err (bool, optional): Whether to output the error log. Defaults to False.

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

    lib.LLL(basis_ptr, ctypes.c_double(delta), ctypes.c_double(eta), output_sl_log, output_rhf_log, output_err, n, m)

    reduced_basis = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            reduced_basis[i, j] = basis_ptr[i][j]

    sl_log = []
    rhf_log = []
    err = 0
    if output_sl_log:
        if os.path.exists("sl_log.csv"):
            sl_log = list(pd.read_csv("sl_log.csv")["val"])
            os.remove("sl_log.csv")
    if output_rhf_log:
        if os.path.exists("rhf_log.csv"):
            rhf_log = list(pd.read_csv("rhf_log.csv")["val"])
            os.remove("rhf_log.csv")
    if os.path.exists("err.csv"):
        err = float(pd.read_csv("err.csv")["val"].iloc[-1])
        os.remove("err.csv")

    return reduced_basis, sl_log, rhf_log, err

def qr_lll(basis: np.ndarray[int], delta: float = 0.99, eta: float = 0.5, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[np.ndarray[int], list[float], list[float], float]:
    """Performs QR-based LLL reduction on a basis with given delta and eta parameters.

    Args:
        basis (np.ndarray[int]): The input basis vectors.
        delta (float, optional): The delta parameter for QR-LLL reduction. Defaults to 0.99.
        eta (float, optional): The eta parameter for size reduction within QR-LLL. Defaults to 0.5.
        output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
        output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
        output_err (bool, optional): Whether to output the error log. Defaults to False.

    Returns:
        np.ndarray[int]: The QR-LLL reduced basis.
    """
    if delta <= 0.25 or delta >= 1:
        raise ValueError("Delta must be in the range (0.25, 1).")
    if eta < 0.5 or eta > np.sqrt(delta):
        raise ValueError("Eta must be in the range [0.5, sqrt(delta)].")
    
    n, m = basis.shape

    lib.qrLLL.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_long,
        ctypes.c_long
    )
    lib.qrLLL.restype = None

    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = (ctypes.c_long * m)()
        for j in range(m):
            basis_ptr[i][j] = ctypes.c_long(basis[i, j])

    lib.qrLLL(basis_ptr, ctypes.c_double(delta), ctypes.c_double(eta), output_sl_log, output_rhf_log, output_err, n, m)

    reduced_basis = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            reduced_basis[i, j] = basis_ptr[i][j]

    sl_log = []
    rhf_log = []
    err = 0
    if output_sl_log:
        if os.path.exists("sl_log.csv"):
            sl_log = list(pd.read_csv("sl_log.csv")["val"])
            os.remove("sl_log.csv")
    if output_rhf_log:
        if os.path.exists("rhf_log.csv"):
            rhf_log = list(pd.read_csv("rhf_log.csv")["val"])
            os.remove("rhf_log.csv")
    if os.path.exists("err.csv"):
        err = float(pd.read_csv("err.csv")["val"].iloc[-1])
        os.remove("err.csv")

    return reduced_basis, sl_log, rhf_log, err

def deep_lll(basis: np.ndarray[int], delta: float = 0.99, eta: float = 0.5, gamma: int = 20, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[np.ndarray[int], list[float], list[float], float]:
    """Performs Deep LLL reduction on a basis with given delta and eta parameters.

    Args:
        basis (np.ndarray[int]): The input basis vectors.
        delta (float, optional): The delta parameter for Deep LLL reduction. Defaults to 0.99.
        eta (float, optional): The eta parameter for size reduction within Deep LLL. Defaults to 0.5.
        gamma (int, optional): The gamma parameter for Deep LLL reduction. Defaults to 20.
        output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
        output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
        output_err (bool, optional): Whether to output the error log. Defaults to False.

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

    lib.deepLLL(basis_ptr, ctypes.c_double(delta), ctypes.c_double(eta), ctypes.c_long(gamma), output_sl_log, output_rhf_log, output_err, n, m)

    reduced_basis = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            reduced_basis[i, j] = basis_ptr[i][j]

    sl_log = []
    rhf_log = []
    err = 0
    if output_sl_log:
        if os.path.exists("sl_log.csv"):
            sl_log = list(pd.read_csv("sl_log.csv")["val"])
            os.remove("sl_log.csv")
    if output_rhf_log:
        if os.path.exists("rhf_log.csv"):
            rhf_log = list(pd.read_csv("rhf_log.csv")["val"])
            os.remove("rhf_log.csv")
    if os.path.exists("err.csv"):
        err = float(pd.read_csv("err.csv")["val"].iloc[-1])
        os.remove("err.csv")

    return reduced_basis, sl_log, rhf_log, err

def qr_deep_lll(basis: np.ndarray[int], delta: float = 0.99, eta: float = 0.5, gamma: int = 20, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[np.ndarray[int], list[float], list[float], float]:
    """Performs QR-based Deep LLL reduction on a basis with given delta and eta parameters.

    Args:
        basis (np.ndarray[int]): The input basis vectors.
        delta (float, optional): The delta parameter for QR-Deep LLL reduction. Defaults to 0.99.
        eta (float, optional): The eta parameter for size reduction within QR-Deep LLL. Defaults to 0.5.
        gamma (int, optional): The gamma parameter for QR-Deep LLL reduction. Defaults to 20.
        output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
        output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
        output_err (bool, optional): Whether to output the error log. Defaults to False.

    Returns:
        np.ndarray[int]: The QR-Deep LLL reduced basis.
    """
    if delta <= 0.25 or delta >= 1:
        raise ValueError("Delta must be in the range (0.25, 1).")
    if eta < 0.5 or eta > np.sqrt(delta):
        raise ValueError("Eta must be in the range [0.5, sqrt(delta)].")
    
    n, m = basis.shape

    lib.qrDeepLLL.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_long,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_long,
        ctypes.c_long
    )
    lib.qrDeepLLL.restype = None

    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = (ctypes.c_long * m)()
        for j in range(m):
            basis_ptr[i][j] = ctypes.c_long(basis[i, j])

    lib.qrDeepLLL(basis_ptr, ctypes.c_double(delta), ctypes.c_double(eta), ctypes.c_long(gamma), output_sl_log, output_rhf_log, output_err, n, m)

    reduced_basis = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            reduced_basis[i, j] = basis_ptr[i][j]

    sl_log = []
    rhf_log = []
    err = 0
    if output_sl_log:
        if os.path.exists("sl_log.csv"):
            sl_log = list(pd.read_csv("sl_log.csv")["val"])
            os.remove("sl_log.csv")
    if output_rhf_log:
        if os.path.exists("rhf_log.csv"):
            rhf_log = list(pd.read_csv("rhf_log.csv")["val"])
            os.remove("rhf_log.csv")
    if os.path.exists("err.csv"):
        err = float(pd.read_csv("err.csv")["val"].iloc[-1])
        os.remove("err.csv")

    return reduced_basis, sl_log, rhf_log, err

def l2(basis: np.ndarray[int], delta: float = 0.99, eta: float = 0.55, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[np.ndarray[int], list[float], list[float], float]:
    """Performs L2 reduction on a basis with given delta and eta parameters.

    Args:
        basis (np.ndarray[int]): The input basis vectors.
        delta (float, optional): The delta parameter for L2 reduction. Defaults to 0.99.
        eta (float, optional): The eta parameter for size reduction within L2. Defaults to 0.55.
        output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
        output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
        output_err (bool, optional): Whether to output the error log. Defaults to False.

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

    lib.L2(basis_ptr, ctypes.c_double(delta), ctypes.c_double(eta), output_sl_log, output_rhf_log, output_err, n, m)

    reduced_basis = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            reduced_basis[i, j] = basis_ptr[i][j]

    sl_log = []
    rhf_log = []
    err = 0
    if output_sl_log:
        if os.path.exists("sl_log.csv"):
            sl_log = list(pd.read_csv("sl_log.csv")["val"])
            os.remove("sl_log.csv")
    if output_rhf_log:
        if os.path.exists("rhf_log.csv"):
            rhf_log = list(pd.read_csv("rhf_log.csv")["val"])
            os.remove("rhf_log.csv")
    if os.path.exists("err.csv"):
        err = float(pd.read_csv("err.csv")["val"].iloc[-1])
        os.remove("err.csv")

    return reduced_basis, sl_log, rhf_log, err

def deep_l2(basis: np.ndarray[int], delta: float = 0.99, eta: float = 0.55, gamma: int = 20, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[np.ndarray[int], list[float], list[float], float]:
    """Performs Deep L2 reduction on a basis with given delta and eta parameters.

    Args:
        basis (np.ndarray[int]): The input basis vectors.
        delta (float, optional): The delta parameter for Deep L2 reduction. Defaults to 0.99.
        eta (float, optional): The eta parameter for size reduction within Deep L2. Defaults to 0.55.
        gamma (int, optional): The gamma parameter for Deep L2 reduction. Defaults to 20.
        output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
        output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
        output_err (bool, optional): Whether to output the error log. Defaults to False.

    Returns:
        np.ndarray[int]: The Deep L2 reduced basis.
    """
    if delta <= 0.25 or delta >= 1:
        raise ValueError("Delta must be in the range (0.25, 1).")
    if eta < 0.5 or eta > np.sqrt(delta):
        raise ValueError("Eta must be in the range [0.5, sqrt(delta)].")
    
    n, m = basis.shape

    lib.deepL2.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_long,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_long,
        ctypes.c_long
    )
    lib.deepL2.restype = None

    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = (ctypes.c_long * m)()
        for j in range(m):
            basis_ptr[i][j] = ctypes.c_long(basis[i, j])

    lib.deepL2(basis_ptr, ctypes.c_double(delta), ctypes.c_double(eta), ctypes.c_long(gamma), output_sl_log, output_rhf_log, output_err, n, m)

    reduced_basis = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            reduced_basis[i, j] = basis_ptr[i][j]

    sl_log = []
    rhf_log = []
    err = 0
    if output_sl_log:
        if os.path.exists("sl_log.csv"):
            sl_log = list(pd.read_csv("sl_log.csv")["val"])
            os.remove("sl_log.csv")
    if output_rhf_log:
        if os.path.exists("rhf_log.csv"):
            rhf_log = list(pd.read_csv("rhf_log.csv")["val"])
            os.remove("rhf_log.csv")
    if os.path.exists("err.csv"):
        err = float(pd.read_csv("err.csv")["val"].iloc[-1])
        os.remove("err.csv")

    return reduced_basis, sl_log, rhf_log, err

def pot_lll(basis: np.ndarray[int], delta: float = 0.99, eta: float = 0.5, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[np.ndarray[int], list[float], list[float], float]:
    """Performs PotLLL reduction on a basis with given delta and eta parameters.

    Args:
        basis (np.ndarray[int]): The input basis vectors.
        delta (float, optional): The delta parameter for PotLLL reduction. Defaults to 0.99.
        eta (float, optional): The eta parameter for size reduction within PotLLL. Defaults to 0.5.
        output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
        output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
        output_err (bool, optional): Whether to output the error log. Defaults to False.

    Returns:
        np.ndarray[int]: The PotLLL reduced basis.
    """
    if delta <= 0.25 or delta >= 1:
        raise ValueError("Delta must be in the range (0.25, 1).")
    if eta < 0.5 or eta > np.sqrt(delta):
        raise ValueError("Eta must be in the range [0.5, sqrt(delta)].")
    
    n, m = basis.shape

    lib.potLLL.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_long,
        ctypes.c_long
    )
    lib.potLLL.restype = None

    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = (ctypes.c_long * m)()
        for j in range(m):
            basis_ptr[i][j] = ctypes.c_long(basis[i, j])

    lib.potLLL(basis_ptr, ctypes.c_double(delta), ctypes.c_double(eta), output_sl_log, output_rhf_log, output_err, n, m)

    reduced_basis = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            reduced_basis[i, j] = basis_ptr[i][j]

    sl_log = []
    rhf_log = []
    err = 0
    if output_sl_log:
        if os.path.exists("sl_log.csv"):
            sl_log = list(pd.read_csv("sl_log.csv")["val"])
            os.remove("sl_log.csv")
    if output_rhf_log:
        if os.path.exists("rhf_log.csv"):
            rhf_log = list(pd.read_csv("rhf_log.csv")["val"])
            os.remove("rhf_log.csv")
    if os.path.exists("err.csv"):
        err = float(pd.read_csv("err.csv")["val"].iloc[-1])
        os.remove("err.csv")
    return reduced_basis, sl_log, rhf_log, err

def bkz(basis: np.ndarray[int], delta: float = 0.99, beta: int = 20, max_loops: int = -1, pruning: bool = False, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[np.ndarray[int], list[float], list[float], float]:
    """Performs BKZ reduction on a basis with given delta and beta parameters.

    Args:
        basis (np.ndarray[int]): The input basis vectors.
        delta (float, optional): The delta parameter for BKZ reduction. Defaults to 0.99.
        beta (int, optional): The block size parameter for BKZ reduction. Defaults to 20.
        max_loops (int, optional): The maximum number of tours through the basis. Defaults to -1 (no limit).
        pruning (bool, optional): Whether to use pruning in the SVP solver. Defaults to False.
        output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
        output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
        output_err (bool, optional): Whether to output the error log. Defaults to False.

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

    lib.BKZ(basis_ptr, ctypes.c_double(delta), ctypes.c_long(beta), ctypes.c_long(max_loops), ctypes.c_bool(pruning), ctypes.c_bool(output_sl_log), ctypes.c_bool(output_rhf_log), ctypes.c_bool(output_err), n, m)

    reduced_basis = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            reduced_basis[i, j] = basis_ptr[i][j]

    sl_log = []
    rhf_log = []
    err = 0
    if output_sl_log:
        if os.path.exists("sl_log.csv"):
            sl_log = list(pd.read_csv("sl_log.csv")["val"])
            os.remove("sl_log.csv")
    if output_rhf_log:
        if os.path.exists("rhf_log.csv"):
            rhf_log = list(pd.read_csv("rhf_log.csv")["val"])
            os.remove("rhf_log.csv")
    if os.path.exists("err.csv"):
        err = float(pd.read_csv("err.csv")["val"].iloc[-1])
        os.remove("err.csv")

    return reduced_basis, sl_log, rhf_log, err

def deep_bkz(basis: np.ndarray[int], delta: float = 0.99, beta: int = 20, gamma: int = 20, max_loops: int = -1, pruning: bool = False, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[np.ndarray[int], list[float], list[float], float]:
    """Performs Deep BKZ reduction on a basis with given delta and beta parameters.

    Args:
        basis (np.ndarray[int]): The input basis vectors.
        delta (float, optional): The delta parameter for Deep BKZ reduction. Defaults to 0.99.
        beta (int, optional): The block size parameter for Deep BKZ reduction. Defaults to 20.
        gamma (int, optional): The gamma parameter for Deep BKZ reduction. Defaults to 20.
        max_loops (int, optional): The maximum number of tours through the basis. Defaults to -1 (no limit).
        pruning (bool, optional): Whether to use pruning in the SVP solver. Defaults to False.
        output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
        output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
        output_err (bool, optional): Whether to output the error log. Defaults to False.

    Returns:
        np.ndarray[int]: The Deep BKZ reduced basis.
    """
    if delta <= 0.25 or delta >= 1:
        raise ValueError("Delta must be in the range (0.25, 1).")
    if beta < 2:
        raise ValueError("Beta must be at least 2.")
    if gamma < 1:
        raise ValueError("Gamma must be at least 1.")
    if max_loops < -1:
        raise ValueError("max_loops must be -1 (no limit) or a non-negative integer.")
    
    n, m = basis.shape

    lib.deepBKZ.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.c_double,
        ctypes.c_long,
        ctypes.c_long,
        ctypes.c_long,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_long,
        ctypes.c_long
    )
    lib.deepBKZ.restype = None
    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = (ctypes.c_long * m)()
        for j in range(m):
            basis_ptr[i][j] = ctypes.c_long(basis[i, j])
            
    lib.deepBKZ(basis_ptr, ctypes.c_double(delta), ctypes.c_long(beta), ctypes.c_long(gamma), ctypes.c_long(max_loops), ctypes.c_bool(pruning), ctypes.c_bool(output_sl_log), ctypes.c_bool(output_rhf_log), ctypes.c_bool(output_err), n, m)
    
    reduced_basis = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            reduced_basis[i, j] = basis_ptr[i][j]
    sl_log = []
    rhf_log = []
    err = 0
    if output_sl_log:
        if os.path.exists("sl_log.csv"):
            sl_log = list(pd.read_csv("sl_log.csv")["val"])
            os.remove("sl_log.csv")
    if output_rhf_log:
        if os.path.exists("rhf_log.csv"):
            rhf_log = list(pd.read_csv("rhf_log.csv")["val"])
            os.remove("rhf_log.csv")
    if os.path.exists("err.csv"):
        err = float(pd.read_csv("err.csv")["val"].iloc[-1])
        os.remove("err.csv")
    return reduced_basis, sl_log, rhf_log, err

def qr_bkz(basis: np.ndarray[int], delta: float = 0.99, beta: int = 20, max_loops: int = -1, pruning: bool = False, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[np.ndarray[int], list[float], list[float], float]:
    """Performs QR-based BKZ reduction on a basis with given delta and beta parameters.

    Args:
        basis (np.ndarray[int]): The input basis vectors.
        delta (float, optional): The delta parameter for QR-BKZ reduction. Defaults to 0.99.
        beta (int, optional): The block size parameter for QR-BKZ reduction. Defaults to 20.
        max_loops (int, optional): The maximum number of tours through the basis. Defaults to -1 (no limit).
        pruning (bool, optional): Whether to use pruning in the SVP solver. Defaults to False.
        output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
        output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
        output_err (bool, optional): Whether to output the error log. Defaults to False.

    Returns:
        np.ndarray[int]: The QR-BKZ reduced basis.
    """
    if delta <= 0.25 or delta >= 1:
        raise ValueError("Delta must be in the range (0.25, 1).")
    if beta < 2:
        raise ValueError("Beta must be at least 2.")
    if max_loops < -1:
        raise ValueError("max_loops must be -1 (no limit) or a non-negative integer.")
    
    n, m = basis.shape

    lib.qrBKZ.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.c_double,
        ctypes.c_long,
        ctypes.c_long,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_long,
        ctypes.c_long
    )
    lib.qrBKZ.restype = None

    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = (ctypes.c_long * m)()
        for j in range(m):
            basis_ptr[i][j] = ctypes.c_long(basis[i, j])

    lib.qrBKZ(basis_ptr, ctypes.c_double(delta), ctypes.c_long(beta), ctypes.c_long(max_loops), ctypes.c_bool(pruning), ctypes.c_bool(output_sl_log), ctypes.c_bool(output_rhf_log), ctypes.c_bool(output_err), n, m)

    reduced_basis = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            reduced_basis[i, j] = basis_ptr[i][j]
    sl_log = []
    rhf_log = []
    err = 0
    if output_sl_log:
        if os.path.exists("sl_log.csv"):
            sl_log = list(pd.read_csv("sl_log.csv")["val"])
            os.remove("sl_log.csv")
    if output_rhf_log:
        if os.path.exists("rhf_log.csv"):
            rhf_log = list(pd.read_csv("rhf_log.csv")["val"])
            os.remove("rhf_log.csv")
    if os.path.exists("err.csv"):
        err = float(pd.read_csv("err.csv")["val"].iloc[-1])
        os.remove("err.csv")
    return reduced_basis, sl_log, rhf_log, err

def qr_deep_bkz(basis: np.ndarray[int], delta: float = 0.99, beta: int = 20, gamma: int = 20, max_loops: int = -1, pruning: bool = False, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[np.ndarray[int], list[float], list[float], float]:
    """Performs QR-based Deep BKZ reduction on a basis with given delta and beta parameters.

    Args:
        basis (np.ndarray[int]): The input basis vectors.
        delta (float, optional): The delta parameter for QR-Deep BKZ reduction. Defaults to 0.99.
        beta (int, optional): The block size parameter for QR-Deep BKZ reduction. Defaults to 20.
        gamma (int, optional): The gamma parameter for QR-Deep BKZ reduction. Defaults to 20.
        max_loops (int, optional): The maximum number of tours through the basis. Defaults to -1 (no limit).
        pruning (bool, optional): Whether to use pruning in the SVP solver. Defaults to False.
        output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
        output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
        output_err (bool, optional): Whether to output the error log. Defaults to False.

    Returns:
        np.ndarray[int]: The QR-Deep BKZ reduced basis.
    """
    if delta <= 0.25 or delta >= 1:
        raise ValueError("Delta must be in the range (0.25, 1).")
    if beta < 2:
        raise ValueError("Beta must be at least 2.")
    if gamma < 1:
        raise ValueError("Gamma must be at least 1.")
    if max_loops < -1:
        raise ValueError("max_loops must be -1 (no limit) or a non-negative integer.")
    
    n, m = basis.shape

    lib.qrDeepBKZ.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.c_double,
        ctypes.c_long,
        ctypes.c_long,
        ctypes.c_long,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_long,
        ctypes.c_long
    )
    lib.qrDeepBKZ.restype = None

    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = (ctypes.c_long * m)()
        for j in range(m):
            basis_ptr[i][j] = ctypes.c_long(basis[i, j])
    lib.qrDeepBKZ(basis_ptr, ctypes.c_double(delta), ctypes.c_long(beta), ctypes.c_long(gamma), ctypes.c_long(max_loops), ctypes.c_bool(pruning), ctypes.c_bool(output_sl_log), ctypes.c_bool(output_rhf_log), ctypes.c_bool(output_err), n, m)
    reduced_basis = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            reduced_basis[i, j] = basis_ptr[i][j]
    sl_log = []
    rhf_log = []
    err = 0
    if output_sl_log:
        if os.path.exists("sl_log.csv"):
            sl_log = list(pd.read_csv("sl_log.csv")["val"])
            os.remove("sl_log.csv")
    if output_rhf_log:
        if os.path.exists("rhf_log.csv"):
            rhf_log = list(pd.read_csv("rhf_log.csv")["val"])
            os.remove("rhf_log.csv")
    if os.path.exists("err.csv"):
        err = float(pd.read_csv("err.csv")["val"].iloc[-1])
        os.remove("err.csv")
    return reduced_basis, sl_log, rhf_log, err

def pot_bkz(basis: np.ndarray[int], delta: float = 0.99, beta: int = 20, max_loops: int = -1, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[np.ndarray[int], list[float], list[float], float]:
    """Performs PotBKZ reduction on a basis with given delta and beta parameters.

    Args:
        basis (np.ndarray[int]): The input basis vectors.
        delta (float, optional): The delta parameter for PotBKZ reduction. Defaults to 0.99.
        beta (int, optional): The block size parameter for PotBKZ reduction. Defaults to 20.
        max_loops (int, optional): The maximum number of tours through the basis. Defaults to -1 (no limit).
        output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
        output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.

    Returns:
        np.ndarray[int]: The PotBKZ reduced basis.
    """
    if delta <= 0.25 or delta >= 1:
        raise ValueError("Delta must be in the range (0.25, 1).")
    if beta < 2:
        raise ValueError("Beta must be at least 2.")
    if max_loops < -1:
        raise ValueError("max_loops must be -1 (no limit) or a non-negative integer.")
    
    n, m = basis.shape

    lib.potBKZ.argtypes = (
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
    lib.potBKZ.restype = None

    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = (ctypes.c_long * m)()
        for j in range(m):
            basis_ptr[i][j] = ctypes.c_long(basis[i, j])

    lib.potBKZ(basis_ptr, ctypes.c_double(delta), ctypes.c_long(beta), ctypes.c_long(max_loops), ctypes.c_bool(output_sl_log), ctypes.c_bool(output_rhf_log), ctypes.c_bool(output_err), n, m)

    reduced_basis = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            reduced_basis[i, j] = basis_ptr[i][j]
            
    sl_log = []
    rhf_log = []
    err = 0
    if output_sl_log:
        if os.path.exists("sl_log.csv"):
            sl_log = list(pd.read_csv("sl_log.csv")["val"])
            os.remove("sl_log.csv")
    if output_rhf_log:
        if os.path.exists("rhf_log.csv"):
            rhf_log = list(pd.read_csv("rhf_log.csv")["val"])
            os.remove("rhf_log.csv")
    if os.path.exists("err.csv"):
        err = float(pd.read_csv("err.csv")["val"].iloc[-1])
        os.remove("err.csv")
    return reduced_basis, sl_log, rhf_log, err

def hkz(basis: np.ndarray[int], pruning: bool = False, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[np.ndarray[int], list[float], list[float], float]:
    """Performs HKZ reduction on a basis.
    
    ## Reference
        - C.-P. Schnorr and M. Euchner. Lattice basis reduction: Improved practical algorithms and solving subset sum problems. 1994

    Args:
        basis (np.ndarray[int]): The input basis vectors.
        pruning (bool, optional): Whether to use pruning in the SVP solver. Defaults to False.
        output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
        output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
        output_err (bool, optional): Whether to output the error log. Defaults to False.

    Returns:
        np.ndarray[int]: The HKZ reduced basis.
    """
    n, m = basis.shape

    lib.HKZ.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_long,
        ctypes.c_long
    )
    lib.HKZ.restype = None

    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = (ctypes.c_long * m)()
        for j in range(m):
            basis_ptr[i][j] = ctypes.c_long(basis[i, j])

    lib.HKZ(basis_ptr, ctypes.c_bool(pruning), ctypes.c_bool(output_sl_log), ctypes.c_bool(output_rhf_log), ctypes.c_bool(output_err), n, m)

    reduced_basis = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            reduced_basis[i, j] = basis_ptr[i][j]

    sl_log = []
    rhf_log = []
    err = 0
    if output_sl_log:
        if os.path.exists("sl_log.csv"):
            sl_log = list(pd.read_csv("sl_log.csv")["val"])
            os.remove("sl_log.csv")
    if output_rhf_log:
        if os.path.exists("rhf_log.csv"):
            rhf_log = list(pd.read_csv("rhf_log.csv")["val"])
            os.remove("rhf_log.csv")
    if os.path.exists("err.csv"):
        err = float(pd.read_csv("err.csv")["val"].iloc[-1])
        os.remove("err.csv")
    return reduced_basis, sl_log, rhf_log, err

def dual_lll(basis: np.ndarray[int], delta: float = 0.99, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[np.ndarray[int], list[float], list[float], float]:
    """Performs Dual LLL reduction on a basis with given delta and eta parameters.

    Args:
        basis (np.ndarray[int]): The input basis vectors.
        delta (float, optional): The delta parameter for Dual LLL reduction. Defaults to 0.99.
        output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
        output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
        output_err (bool, optional): Whether to output the error log. Defaults to False.

    Returns:
        np.ndarray[int]: The Dual LLL reduced basis.
    """
    if delta <= 0.25 or delta >= 1:
        raise ValueError("Delta must be in the range (0.25, 1).")
    
    n, m = basis.shape

    lib.dualLLL.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.c_double,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_long,
        ctypes.c_long
    )
    lib.dualLLL.restype = None

    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = (ctypes.c_long * m)()
        for j in range(m):
            basis_ptr[i][j] = ctypes.c_long(basis[i, j])

    lib.dualLLL(basis_ptr, ctypes.c_double(delta), output_sl_log, output_rhf_log, output_err, n, m)

    reduced_basis = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            reduced_basis[i, j] = basis_ptr[i][j]

    sl_log = []
    rhf_log = []
    err = 0
    if output_sl_log:
        if os.path.exists("sl_log.csv"):
            sl_log = list(pd.read_csv("sl_log.csv")["val"])
            os.remove("sl_log.csv")
    if output_rhf_log:
        if os.path.exists("rhf_log.csv"):
            rhf_log = list(pd.read_csv("rhf_log.csv")["val"])
            os.remove("rhf_log.csv")
    if os.path.exists("err.csv"):
        err = float(pd.read_csv("err.csv")["val"].iloc[-1])
        os.remove("err.csv")
    return reduced_basis, sl_log, rhf_log, err

def dual_deep_lll(basis: np.ndarray[int], delta: float = 0.99, gamma: int = 20, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[np.ndarray[int], list[float], list[float], float]:
    """Performs Dual Deep LLL reduction on a basis with given delta and eta parameters.

    Args:
        basis (np.ndarray[int]): The input basis vectors.
        delta (float, optional): The delta parameter for Dual Deep LLL reduction. Defaults to 0.99.
        gamma (int, optional): The gamma parameter for Dual Deep LLL reduction. Defaults to 20.
        output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
        output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
        output_err (bool, optional): Whether to output the error log. Defaults to False.

    Returns:
        np.ndarray[int]: The Dual Deep LLL reduced basis.
    """
    if delta <= 0.25 or delta >= 1:
        raise ValueError("Delta must be in the range (0.25, 1).")
    if gamma < 1:
        raise ValueError("Gamma must be at least 1.")
    
    n, m = basis.shape

    lib.dualDeepLLL.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.c_double,
        ctypes.c_long,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_long,
        ctypes.c_long
    )
    lib.dualDeepLLL.restype = None

    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = (ctypes.c_long * m)()
        for j in range(m):
            basis_ptr[i][j] = ctypes.c_long(basis[i, j])

    lib.dualDeepLLL(basis_ptr, ctypes.c_double(delta), ctypes.c_long(gamma), output_sl_log, output_rhf_log, output_err, n, m)

    reduced_basis = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            reduced_basis[i, j] = basis_ptr[i][j]

    sl_log = []
    rhf_log = []
    err = 0
    if output_sl_log:
        if os.path.exists("sl_log.csv"):
            sl_log = list(pd.read_csv("sl_log.csv")["val"])
            os.remove("sl_log.csv")
    if output_rhf_log:
        if os.path.exists("rhf_log.csv"):
            rhf_log = list(pd.read_csv("rhf_log.csv")["val"])
            os.remove("rhf_log.csv")
    if os.path.exists("err.csv"):
        err = float(pd.read_csv("err.csv")["val"].iloc[-1])
        os.remove("err.csv")
    return reduced_basis, sl_log, rhf_log, err

def dual_pot_lll(basis: np.ndarray[int], delta: float = 0.99, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[np.ndarray[int], list[float], list[float], float]:
    """Performs Dual PotLLL reduction on a basis with given delta and eta parameters.
    Args:
        basis (np.ndarray[int]): The input basis vectors.
        delta (float, optional): The delta parameter for Dual PotLLL reduction. Defaults to 0.99.
        eta (float, optional): The eta parameter for size reduction within Dual PotLLL. Defaults to 0.5.
        output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
        output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
        output_err (bool, optional): Whether to output the error log. Defaults to False.
    Returns:
        np.ndarray[int]: The Dual PotLLL reduced basis.
    """
    if delta <= 0.25 or delta >= 1:
        raise ValueError("Delta must be in the range (0.25, 1).")

    n, m = basis.shape
    lib.dualPotLLL.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.c_double,                                # delta
        ctypes.c_bool,                                  # output_sl_log
        ctypes.c_bool,                                  # output_rhf_log
        ctypes.c_bool,                                  # output_err
        ctypes.c_long,                                  # n
        ctypes.c_long                                   # m
    )
    lib.dualPotLLL.restype = None
    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = (ctypes.c_long * m)()
        for j in range(m):
            basis_ptr[i][j] = ctypes.c_long(basis[i, j])
    lib.dualPotLLL(basis_ptr, ctypes.c_double(delta), ctypes.c_bool(output_sl_log), ctypes.c_bool(output_rhf_log), ctypes.c_bool(output_err), n, m)
    reduced_basis = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            reduced_basis[i, j] = basis_ptr[i][j]
    sl_log = []
    rhf_log = []
    err = 0
    if output_sl_log:
        if os.path.exists("sl_log.csv"):
            sl_log = list(pd.read_csv("sl_log.csv")["val"])
            os.remove("sl_log.csv")
    if output_rhf_log:
        if os.path.exists("rhf_log.csv"):
            rhf_log = list(pd.read_csv("rhf_log.csv")["val"])
            os.remove("rhf_log.csv")
    if os.path.exists("err.csv"):
        err = float(pd.read_csv("err.csv")["val"].iloc[-1])
        os.remove("err.csv")
    return reduced_basis, sl_log, rhf_log, err
