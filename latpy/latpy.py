from . import core
import numpy as np

class LatPy:
    """
    A class for representing a lattice in n-dimensional space.
    """
    def __init__(self, basis):
        self.basis = basis
        self.n, self.m = basis.shape

    def __repr__(self):
        return f"LatPy(basis={self.basis})"

    def __str__(self):
        return f"{self.n}-dimensional lattice with basis:\n{self.basis}"

    def compute_gso(self) -> tuple[np.ndarray[float], np.ndarray[float]]:
        """Compute the Gram-Schmidt orthogonalization of the lattice basis.

        Returns:
            np.ndarray: The GSO basis.
        """
        return core.compute_gso(self.basis)

    def volume(self) -> int:
        """Calculate the volume of the lattice.

        Returns:
            int: The volume of the lattice.
        """
        return core.volume(self.basis)
    
    def sl(self) -> float:
        """Calculate the GSA-slope of the lattice.

        Returns:
            float: The GSA-slope of the lattice basis.
        """
        return core.sl(self.basis)

    def hf(self) -> float:
        """Calculate the Hermite-factor of the lattice.

        Returns:
            float: The Hermite-factor of the lattice basis.
        """
        return core.hf(self.basis)

    def rhf(self) -> float:
        """Calculate the root of Hermite-factor of the lattice.

        Returns:
            float: The root of Hermite-factor of the lattice basis.
        """
        return core.rhf(self.basis)
