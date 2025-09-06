from __future__ import annotations

import numpy as np

from . import core
from . import reduction
from . import svp

class LatPy:
    """
    A class for representing a lattice in n-dimensional space.
    """
    def __init__(self, basis) -> None:
        """Initialize the LatPy class.

        Args:
            basis (array like): The basis vectors of the lattice.
        """
        self.basis = basis
        self.n, self.m = basis.shape

    def __repr__(self) -> str:
        """Return a string representation of the LatPy object.

        Returns:
            str: A string representation of the LatPy object.
        """
        return f"LatPy(basis={self.basis})"

    def __str__(self) -> str:
        """Return a string representation of the LatPy object.

        Returns:
            str: A string representation of the LatPy object.
        """
        return f"{self.n}-dimensional lattice with basis:\n{self.basis}"

    def compute_gso(self) -> tuple[np.ndarray[float], np.ndarray[float]]:
        """Compute the Gram-Schmidt orthogonalization of the lattice basis.

        Returns:
            np.ndarray: The GSO basis.
        """
        return core.compute_gso(self.basis)
    
    def gram_schmidt(self) -> tuple[np.ndarray[float], np.ndarray[float]]:
        """Alias for compute_gso method.

        Returns:
            np.ndarray: The GSO basis.
        """
        return self.compute_gso()

    def volume(self) -> int:
        """Calculate the volume of the lattice.

        Returns:
            int: The volume of the lattice.
        """
        return core.volume(self.basis)
    
    def vol(self) -> int:
        """Alias for volume method.

        Returns:
            int: The volume of the lattice.
        """
        return self.volume()
    
    def det(self) -> int:
        """Alias for volume method.

        Returns:
            int: The volume of the lattice.
        """
        return self.volume()
    
    def determinant(self) -> int:
        """Alias for volume method.

        Returns:
            int: The volume of the lattice.
        """
        return self.volume()
    
    def sl(self) -> float:
        """Calculate the GSA-slope of the lattice.

        Returns:
            float: The GSA-slope of the lattice basis.
        """
        return core.sl(self.basis)

    def pot(self) -> float:
        """Calculate the potential of the lattice.

        Returns:
            float: The potential of the lattice basis.
        """
        return core.pot(self.basis)
    
    def potential(self) -> float:
        """Alias for pot method.

        Returns:
            float: The potential of the lattice basis.
        """
        return self.pot()

    def hf(self) -> float:
        """Calculate the Hermite-factor of the lattice.

        Returns:
            float: The Hermite-factor of the lattice basis.
        """
        return core.hf(self.basis)
    
    def hermite_factor(self) -> float:
        """Alias for hf method.

        Returns:
            float: The Hermite-factor of the lattice basis.
        """
        return self.hf()

    def rhf(self) -> float:
        """Calculate the root of Hermite-factor of the lattice.

        Returns:
            float: The root of Hermite-factor of the lattice basis.
        """
        return core.rhf(self.basis)
    
    def root_hermite_factor(self) -> float:
        """Alias for rhf method.

        Returns:
            float: The root of Hermite-factor of the lattice basis.
        """
        return self.rhf()
    
    def gh(self) -> float:
        """Calculate the Gaussian heuristic of the lattice.

        Returns:
            float: The Gaussian heuristic of the lattice basis.
        """
        return core.gh(self.basis)
    
    def gaussian_heuristic(self) -> float:
        """Alias for gh method.

        Returns:
            float: The Gaussian heuristic of the lattice basis.
        """
        return self.gh()

    def od(self) -> float:
        """Calculate the orthogonality defect of the lattice.

        Returns:
            float: The orthogonality defect of the lattice basis.
        """
        return core.od(self.basis)
    
    def orthogonality_defect(self) -> float:
        """Alias for od method.

        Returns:
            float: The orthogonality defect of the lattice basis.
        """
        return self.od()

    def is_size(self) -> bool:
        """Check if the basis is size reduced.

        Returns:
            bool: True if the basis is size reduced, False otherwise.
        """
        return core.is_size(self.basis)
    
    def is_size_reduced(self) -> bool:
        """Alias for is_size method.

        Returns:
            bool: True if the basis is size reduced, False otherwise.
        """
        return self.is_size()
    
    def is_weakly_lll(self, delta: float) -> bool:
        """Check if the basis is weakly LLL reduced with a given delta parameter.

        Args:
            delta (float, optional): The delta parameter for weakly LLL reduction. Defaults to 0.99.

        Returns:
            bool: True if the basis is weakly LLL reduced, False otherwise.
        """
        return core.is_weakly_lll(self.basis, delta)
    
    def is_weakly_lll_reduced(self, delta: float) -> bool:
        """Alias for is_weakly_lll method.

        Args:
            delta (float, optional): The delta parameter for weakly LLL reduction. Defaults to 0.99.

        Returns:
            bool: True if the basis is weakly LLL reduced, False otherwise.
        """
        return self.is_weakly_lll(delta)
    
    def is_lll(self, delta: float = 0.99) -> bool:
        """Check if the basis is LLL reduced with a given delta parameter.

        Args:
            delta (float, optional): The delta parameter for LLL reduction. Defaults to 0.99.

        Returns:
            bool: True if the basis is LLL reduced, False otherwise.
        """
        return self.is_size() and self.is_weakly_lll(delta)
    
    def is_lll_reduced(self, delta: float = 0.99) -> bool:
        """Alias for is_lll method.

        Args:
            delta (float, optional): The delta parameter for LLL reduction. Defaults to 0.99.

        Returns:
            bool: True if the basis is LLL reduced, False otherwise.
        """
        return self.is_lll(delta)
    
    def lagrange(self) -> LatPy:
        """Perform Lagrange reduction on the lattice basis.

        Returns:
            LatPy: The reduced basis.
        """
        return LatPy(reduction.lagrange(self.basis))

    def gauss(self) -> LatPy:
        """Alias for lagrange method.

        Returns:
            LatPy: The reduced basis.
        """
        return self.lagrange()

    def lagrange_gauss(self) -> LatPy:
        """Alias for lagrange method.

        Returns:
            LatPy: The reduced basis.
        """
        return self.lagrange()

    def size(self, eta: float) -> LatPy:
        """Perform size reduction on the lattice basis with a given eta parameter.

        Args:
            eta (float): The eta parameter for size reduction.

        Returns:
            np.ndarray[int]: The reduced basis.
        """
        return LatPy(reduction.size(self.basis, eta))

    def lll(self, delta: float = 0.99, eta: float = 0.55, output_sl_log: bool = False, output_rhf_log: bool = False) -> tuple[LatPy, list[float], list[float]]:
        """Perform LLL reduction on the lattice basis with given delta and eta parameters.

        Args:
            delta (float, optional): The delta parameter for LLL reduction. Defaults to 0.99.
            eta (float, optional): The eta parameter for size reduction. Defaults to 0.55.

        Returns:
            LatPy: The reduced basis.
        """
        reduced_basis, sl_log, rhf_log = reduction.lll(self.basis, delta, eta, output_sl_log, output_rhf_log)
        return LatPy(reduced_basis), sl_log, rhf_log
    
    def l3(self, delta: float = 0.99, eta: float = 0.55, output_sl_log: bool = False, output_rhf_log: bool = False) -> tuple[LatPy, list[float], list[float]]:
        """Alias for lll method.

        Args:
            delta (float, optional): The delta parameter for LLL reduction. Defaults to 0.99.
            eta (float, optional): The eta parameter for size reduction. Defaults to 0.55.

        Returns:
            LatPy: The reduced basis.
        """
        return self.lll(delta, eta, output_sl_log, output_rhf_log)
    
    def lenstra_lenstra_lovasz(self, delta: float = 0.99, eta: float = 0.55, output_sl_log: bool = False, output_rhf_log: bool = False) -> tuple[LatPy, list[float], list[float]]:
        """Alias for lll method.

        Args:
            delta (float, optional): The delta parameter for LLL reduction. Defaults to 0.99.
            eta (float, optional): The eta parameter for size reduction. Defaults to 0.55.

        Returns:
            LatPy: The reduced basis.
        """
        return self.lll(delta, eta, output_sl_log, output_rhf_log)
    
    def l2(self, delta: float = 0.99, eta: float = 0.55, output_sl_log: bool = False, output_rhf_log: bool = False) -> tuple[LatPy, list[float], list[float]]:
        """Perform L2 reduction on the lattice basis with given delta and eta parameters.

        Args:
            delta (float, optional): The delta parameter for L2 reduction. Defaults to 0.99.
            eta (float, optional): The eta parameter for size reduction. Defaults to 0.55.
            output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
            output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.

        Returns:
            LatPy: The reduced basis.
        """
        reduced_basis, sl_log, rhf_log = reduction.l2(self.basis, delta, eta, output_sl_log, output_rhf_log)
        return LatPy(reduced_basis), sl_log, rhf_log

    def deep_lll(self, delta: float = 0.99, eta: float = 0.55, gamma: int = 20, output_sl_log: bool = False, output_rhf_log: bool = False) -> tuple[LatPy, list[float], list[float]]:
        """Perform Deep LLL reduction on the lattice basis with given delta, eta, and gamma parameters.

        Args:
            delta (float, optional): The delta parameter for Deep LLL reduction. Defaults to 0.99.
            eta (float, optional): The eta parameter for size reduction. Defaults to 0.55.
            gamma (int, optional): The gamma parameter for Deep LLL reduction. Defaults to 20.
            output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
            output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.

        Returns:
            LatPy: The reduced basis.
        """
        reduced_basis, sl_log, rhf_log = reduction.deep_lll(self.basis, delta, eta, gamma, output_sl_log, output_rhf_log)
        return LatPy(reduced_basis), sl_log, rhf_log
    
    def lll_with_deep_insertions(self, delta: float = 0.99, eta: float = 0.55, gamma: int = 20, output_sl_log: bool = False, output_rhf_log: bool = False) -> tuple[LatPy, list[float], list[float]]:
        """Alias for deep_lll method.

        Args:
            delta (float, optional): The delta parameter for Deep LLL reduction. Defaults to 0.99.
            eta (float, optional): The eta parameter for size reduction. Defaults to 0.55.
            gamma (int, optional): The gamma parameter for Deep LLL reduction. Defaults to 20.
            output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
            output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.

        Returns:
            LatPy: The reduced basis.
        """
        return self.deep_lll(delta, eta, gamma, output_sl_log, output_rhf_log)
    
    def enum_sv(self, pruning: bool = False) -> np.ndarray[int]:
        """Enumerates the shortest vector in the lattice basis using the SVP algorithm.

        Args:
            pruning (bool, optional): Whether to use pruning. Defaults to False.

        Returns:
            np.ndarray[int]: The shortest vector found in the lattice.
        """
        return svp.enum_sv(self.basis, pruning)
