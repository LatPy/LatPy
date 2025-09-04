from . import core
import numpy as np

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
