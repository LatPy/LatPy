from .loader import lib
import numpy as np
import math
import ctypes

def compute_gso(basis: np.ndarray[int]) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """Computes the Gram-Schmidt orthogonalization of a basis.

    Args:
        basis (np.ndarray[int]): The input basis vectors.

    Returns:
        tuple[np.ndarray[float], np.ndarray[float]]: The orthogonalized basis and the coefficients.
    """
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

def volume(basis: np.ndarray[int]) -> int:
    """Computes the volume of lattice spanned by input basis vectors.

    Args:
        basis (np.ndarray[int]): The input basis vectors.

    Returns:
        int: The volume of the lattice.
    """
    n, m = basis.shape

    lib.volume.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.c_long,
        ctypes.c_long
    )
    lib.volume.restype = ctypes.c_char_p # ctypes.c_long

    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = basis[i].ctypes.data_as(ctypes.POINTER(ctypes.c_long))

    return int(lib.volume(basis_ptr, n, m).decode("utf-8"))

def sl(basis: np.ndarray[int]) -> float:
    """Computes the GSA-slope of the lattice basis.

    Args:
        basis (np.ndarray[int]): The input basis vectors.

    Returns:
        float: The GSA-slope of the lattice basis.
    """
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

def pot(basis: np.ndarray[int]) -> float:
    """Computes the potential of the lattice basis.

    Args:
        basis (np.ndarray[int]): The input basis vectors.

    Returns:
        float: The potential of the lattice basis.
    """
    n, m = basis.shape

    lib.pot.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.c_long,
        ctypes.c_long
    )
    lib.pot.restype = ctypes.c_char_p

    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = basis[i].ctypes.data_as(ctypes.POINTER(ctypes.c_long))

    return float(lib.pot(basis_ptr, n, m).decode("utf-8"))

def hf(basis: np.ndarray[int]) -> float:
    """Computes Hermite-factor of the lattice basis.

    Args:
        basis (np.ndarray[int]): The input basis vectors.

    Returns:
        float: The Hermite-factor of the lattice basis.
    """
    n, m = basis.shape

    lib.hf.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.c_long,
        ctypes.c_long
    )
    lib.hf.restype = ctypes.c_longdouble

    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = basis[i].ctypes.data_as(ctypes.POINTER(ctypes.c_long))

    return lib.hf(basis_ptr, n, m)

def rhf(basis: np.ndarray[int]) -> float:
    """Computes the root of Hermite-factor of the lattice basis.

    Args:
        basis (np.ndarray[int]): The input basis vectors.

    Returns:
        float: The root of Hermite-factor of the lattice basis.
    """
    n, m = basis.shape

    lib.rhf.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.c_long,
        ctypes.c_long
    )
    lib.rhf.restype = ctypes.c_longdouble

    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = basis[i].ctypes.data_as(ctypes.POINTER(ctypes.c_long))

    return lib.rhf(basis_ptr, n, m)

def gh(basis: np.ndarray[int]) -> float:
    """Computes the Gaussian heuristic of the lattice basis.

    Args:
        basis (np.ndarray[int]): The input basis vectors.

    Returns:
        float: The Gaussian heuristic of the lattice basis.
    """
    n = basis.shape[0]

    return (math.gamma(n * 0.5 + 1) ** (1.0 / n)) / math.sqrt(math.pi)

def od(basis: np.ndarray[int]) -> float:
    """Computes the orthogonality defect of the lattice basis.

    Args:
        basis (np.ndarray[int]): The input basis vectors.

    Returns:
        float: The orthogonality defect of the lattice basis.
    """
    n, m = basis.shape

    lib.od.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.c_long,
        ctypes.c_long
    )
    lib.od.restype = ctypes.c_longdouble

    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = basis[i].ctypes.data_as(ctypes.POINTER(ctypes.c_long))

    return lib.od(basis_ptr, n, m)

def is_size(basis: np.ndarray[int]) -> bool:
    """Checks if the lattice basis is of a certain size.

    Args:
        basis (np.ndarray[int]): The input basis vectors.

    Returns:
        bool: True if the basis is of the desired size, False otherwise.
    """
    n, m = basis.shape
    
    lib.isSize.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.c_long,
        ctypes.c_long
    )
    lib.isSize.restype = ctypes.c_bool

    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = basis[i].ctypes.data_as(ctypes.POINTER(ctypes.c_long))

    return lib.isSize(basis_ptr, n, m)

def is_weakly_lll(basis: np.ndarray[int], delta: float) -> bool:
    """Checks if the lattice basis is weakly LLL-reduced, that is, satisfies Lovasz condition with a given delta parameter.

    Args:
        basis (np.ndarray[int]): The input basis vectors.
        delta (float): The delta parameter for the weakly LLL condition.

    Returns:
        bool: True if the basis is weakly LLL-reduced, False otherwise.
    """
    if delta < 0.25 or delta >= 1.0:
        raise ValueError("Delta must be in the range [0.25, 1.0).")
    
    n, m = basis.shape

    lib.isWeaklyLLL.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.c_double,
        ctypes.c_long,
        ctypes.c_long
    )
    lib.isWeaklyLLL.restype = ctypes.c_bool

    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = basis[i].ctypes.data_as(ctypes.POINTER(ctypes.c_long))

    return lib.isWeaklyLLL(basis_ptr, ctypes.c_double(delta), n, m)

def is_seysen(basis: np.ndarray[int]) -> bool:
    """Checks if the lattice basis is Seysen-reduced.

    Args:
        basis (np.ndarray[int]): The input basis vectors.

    Returns:
        bool: True if the basis is Seysen-reduced, False otherwise.
    """
    n, m = basis.shape

    lib.isSeysen.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.c_long,
        ctypes.c_long
    )
    lib.isSeysen.restype = ctypes.c_bool

    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = basis[i].ctypes.data_as(ctypes.POINTER(ctypes.c_long))

    return lib.isSeysen(basis_ptr, n, m)

def is_weakly_deep_lll(basis: np.ndarray[int], delta: float) -> bool:
    """Checks if the lattice basis is weakly DeepLLL-reduced, that is, satisfies deep-exchange condition with given delta parameters.

    Args:
        basis (np.ndarray[int]): The input basis vectors.
        delta (float): The delta parameter for the DeepLLL condition.
        eta (float): The eta parameter for the DeepLLL condition.
        gamma (int): The gamma parameter for the DeepLLL condition.

    Returns:
        bool: True if the basis is weakly DeepLLL-reduced, False otherwise.
    """
    if delta < 0.25 or delta >= 1.0:
        raise ValueError("Delta must be in the range [0.25, 1.0).")
    
    n, m = basis.shape

    lib.isWeaklyDeepLLL.argtypes = (
        ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # basis
        ctypes.c_double,
        ctypes.c_long,
        ctypes.c_long
    )
    lib.isWeaklyDeepLLL.restype = ctypes.c_bool

    basis_ptr = (ctypes.POINTER(ctypes.c_long) * n)()
    for i in range(n):
        basis_ptr[i] = basis[i].ctypes.data_as(ctypes.POINTER(ctypes.c_long))

    return lib.isWeaklyDeepLLL(basis_ptr, ctypes.c_double(delta), n, m)
