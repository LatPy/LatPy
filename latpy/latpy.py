from __future__ import annotations

import numpy as np
import importlib.resources as resources

from . import core
from . import reduction
from . import svp

class Lattice:
    """
    A class for representing a lattice in n-dimensional space.
    """
    def __init__(self, basis) -> None:
        """Initialize the Lattice class.

        Args:
            basis (array like): The basis vectors of the lattice.
        """
        self.basis = basis
        self.n, self.m = basis.shape

    def __repr__(self) -> str:
        """Return a string representation of the Lattice object.

        Returns:
            str: A string representation of the Lattice object.
        """
        return f"Lattice(basis={self.basis})"

    def __str__(self) -> str:
        """Return a string representation of the Lattice object.

        Returns:
            str: A string representation of the Lattice object.
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
    
    def is_seysen(self) -> bool:
        """Check if the basis is Seysen reduced.

        Returns:
            bool: True if the basis is Seysen reduced, False otherwise.
        """
        return core.is_seysen(self.basis)

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
    
    def is_weakly_deep_lll(self, delta: float = 0.99) -> bool:
        """Check if the basis is weakly DeepLLL reduced with given delta and gamma parameters.

        Args:
            delta (float, optional): The delta parameter for DeepLLL reduction. Defaults to 0.99.

        Returns:
            bool: True if the basis is weakly DeepLLL reduced, False otherwise.
        """
        return core.is_weakly_deep_lll(self.basis, delta)
    
    def is_weakly_deep_lll_reduced(self, delta: float = 0.99) -> bool:
        """Alias for is_weakly_deep_lll method.

        Args:
            delta (float, optional): The delta parameter for DeepLLL reduction. Defaults to 0.99.

        Returns:
            bool: True if the basis is weakly DeepLLL reduced, False otherwise.
        """
        return self.is_weakly_deep_lll(delta)
    
    def is_deep_lll(self, delta: float = 0.99) -> bool:
        """Check if the basis is DeepLLL reduced with given delta and gamma parameters.

        Args:
            delta (float, optional): The delta parameter for DeepLLL reduction. Defaults to 0.99.

        Returns:
            bool: True if the basis is DeepLLL reduced, False otherwise.
        """
        return self.is_size() and self.is_weakly_deep_lll(delta)
    
    def is_deep_lll_reduced(self, delta: float = 0.99) -> bool:
        """Alias for is_deep_lll method.

        Args:
            delta (float, optional): The delta parameter for DeepLLL reduction. Defaults to 0.99.

        Returns:
            bool: True if the basis is DeepLLL reduced, False otherwise.
        """
        return self.is_deep_lll(delta)
    
    def lagrange(self) -> Lattice:
        """Perform Lagrange reduction on the lattice basis.
        
        ## Reference
        - J.-L. Lagrange. Recherches d'arithmétique. 1773

        Returns:
            Lattice: The reduced basis.
        """
        return Lattice(reduction.lagrange(self.basis))

    def gauss(self) -> Lattice:
        """Alias for lagrange method.
        
        ## Reference
        - J.-L. Lagrange. Recherches d'arithmétique. 1773

        Returns:
            Lattice: The reduced basis.
        """
        return self.lagrange()

    def lagrange_gauss(self) -> Lattice:
        """Alias for lagrange method.
        
        ## Reference
        - J.-L. Lagrange. Recherches d'arithmétique. 1773

        Returns:
            Lattice: The reduced basis.
        """
        return self.lagrange()

    def size(self, eta: float) -> Lattice:
        """Perform size reduction on the lattice basis with a given eta parameter.

        Args:
            eta (float): The eta parameter for size reduction.

        Returns:
            np.ndarray[int]: The reduced basis.
        """
        return Lattice(reduction.size(self.basis, eta))

    def seysen(self) -> Lattice:
        """Perform Seysen reduction on the lattice basis.

        Returns:
            Lattice: The reduced basis.
        """
        return Lattice(reduction.seysen(self.basis))
    
    def lll(self, delta: float = 0.99, eta: float = 0.5, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[Lattice, list[float], list[float], float]:
        """Perform LLL reduction on the lattice basis with given delta and eta parameters.
        
        ## Reference
        - A. K. Lenstra and H. W. Lenstra and L. Lovász, Factoring polynomials with rational coefficients, 1982
        - R. Bottesch and M. W. Haslbeck and R. Thiemann, A verified efficient implementation of the LLL basis reduction algorithm, 2018

        Args:
            delta (float, optional): The delta parameter for LLL reduction. Defaults to 0.99.
            eta (float, optional): The eta parameter for size reduction. Defaults to 0.5.
            output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
            output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
            output_err (bool, optional): Whether to output the error. Defaults to False.

        Returns:
            Lattice: The reduced basis.
        """
        reduced_basis, sl_log, rhf_log, err = reduction.lll(self.basis, delta, eta, output_sl_log, output_rhf_log, output_err)
        return Lattice(reduced_basis), sl_log, rhf_log, err

    def qr_lll(self, delta: float = 0.99, eta: float = 0.5, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[Lattice, list[float], list[float], float]:
        """Perform QR-LLL reduction on the lattice basis with given delta and eta parameters.

        Args:
            delta (float, optional): The delta parameter for QR-LLL reduction. Defaults to 0.99.
            eta (float, optional): The eta parameter for size reduction. Defaults to 0.5.
            output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
            output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
            output_err (bool, optional): Whether to output the error. Defaults to False.

        Returns:
            Lattice: The reduced basis.
        """
        reduced_basis, sl_log, rhf_log, err = reduction.qr_lll(self.basis, delta, eta, output_sl_log, output_rhf_log, output_err)
        return Lattice(reduced_basis), sl_log, rhf_log, err

    def l3(self, delta: float = 0.99, eta: float = 0.5, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[Lattice, list[float], list[float], float]:
        """Alias for lll method.
        
        ## Reference
        - A. K. Lenstra and H. W. Lenstra and L. Lovász, Factoring polynomials with rational coefficients, 1982
        - R. Bottesch and M. W. Haslbeck and R. Thiemann, A verified efficient implementation of the LLL basis reduction algorithm, 2018

        Args:
            delta (float, optional): The delta parameter for LLL reduction. Defaults to 0.99.
            eta (float, optional): The eta parameter for size reduction. Defaults to 0.5.
            output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
            output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
            output_err (bool, optional): Whether to output the error. Defaults to False.

        Returns:
            Lattice: The reduced basis.
        """
        return self.lll(delta, eta, output_sl_log, output_rhf_log, output_err)

    def lenstra_lenstra_lovasz(self, delta: float = 0.99, eta: float = 0.5, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[Lattice, list[float], list[float], float]:
        """Alias for lll method.
        
        ## Reference
        - A. K. Lenstra and H. W. Lenstra and L. Lovász, Factoring polynomials with rational coefficients, 1982
        - R. Bottesch and M. W. Haslbeck and R. Thiemann, A verified efficient implementation of the LLL basis reduction algorithm, 2018

        Args:
            delta (float, optional): The delta parameter for LLL reduction. Defaults to 0.99.
            eta (float, optional): The eta parameter for size reduction. Defaults to 0.5.
            output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
            output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
            output_err (bool, optional): Whether to output the error. Defaults to False.

        Returns:
            Lattice: The reduced basis.
        """
        return self.lll(delta, eta, output_sl_log, output_rhf_log, output_err)

    def l2(self, delta: float = 0.99, eta: float = 0.55, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[Lattice, list[float], list[float], float]:
        """Perform L2 reduction on the lattice basis with given delta and eta parameters.
        
        ## Reference
        - P. Q. Nguyen and D. Stehlé. Floating-point LLL revisited. 2005
        - P. Q. Nguyen and D. Stehlé. An LLL algorithm with quadratic complexity. 2009
        Args:
            delta (float, optional): The delta parameter for L2 reduction. Defaults to 0.99.
            eta (float, optional): The eta parameter for size reduction. Defaults to 0.55.
            output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
            output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
            output_err (bool, optional): Whether to output the error. Defaults to False.

        Returns:
            Lattice: The reduced basis.
        """
        reduced_basis, sl_log, rhf_log, err = reduction.l2(self.basis, delta, eta, output_sl_log, output_rhf_log, output_err)
        return Lattice(reduced_basis), sl_log, rhf_log, err

    def deep_lll(self, delta: float = 0.99, eta: float = 0.5, gamma: int = 20, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[Lattice, list[float], list[float], float]:
        """Perform Deep LLL reduction on the lattice basis with given delta, eta, and gamma parameters.
        
        ## Reference
        - C.-P. Schnorr and M. Euchner. Lattice basis reduction: Improved practical algorithms and solving subset sum problems. 1994

        Args:
            delta (float, optional): The delta parameter for Deep LLL reduction. Defaults to 0.99.
            eta (float, optional): The eta parameter for size reduction. Defaults to 0.5.
            gamma (int, optional): The gamma parameter for Deep LLL reduction. Defaults to 20.
            output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
            output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
            output_err (bool, optional): Whether to output the error. Defaults to False.

        Returns:
            Lattice: The reduced basis.
        """
        reduced_basis, sl_log, rhf_log, err = reduction.deep_lll(self.basis, delta, eta, gamma, output_sl_log, output_rhf_log, output_err)
        return Lattice(reduced_basis), sl_log, rhf_log, err

    def deep_l2(self, delta: float = 0.99, eta: float = 0.55, gamma: int = 20, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[Lattice, list[float], list[float], float]:
        """Perform Deep L2 reduction on the lattice basis with given delta, eta, and gamma parameters.

        Args:
            delta (float, optional): The delta parameter for Deep L2 reduction. Defaults to 0.99.
            eta (float, optional): The eta parameter for size reduction. Defaults to 0.55.
            gamma (int, optional): The gamma parameter for Deep L2 reduction. Defaults to 20.
            output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
            output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
            output_err (bool, optional): Whether to output the error. Defaults to False.

        Returns:
            Lattice: The reduced basis.
        """
        reduced_basis, sl_log, rhf_log, err = reduction.deep_l2(self.basis, delta, eta, gamma, output_sl_log, output_rhf_log, output_err)
        return Lattice(reduced_basis), sl_log, rhf_log, err

    def l2_with_deep_insertions(self, delta: float = 0.99, eta: float = 0.55, gamma: int = 20, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[Lattice, list[float], list[float], float]:
        """Alias for deep_l2 method.

        Args:
            delta (float, optional): The delta parameter for Deep L2 reduction. Defaults to 0.99.
            eta (float, optional): The eta parameter for size reduction. Defaults to 0.55.
            gamma (int, optional): The gamma parameter for Deep L2 reduction. Defaults to 20.
            output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
            output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
            output_err (bool, optional): Whether to output the error. Defaults to False.

        Returns:
            Lattice: The reduced basis.
        """
        return self.deep_l2(delta, eta, gamma, output_sl_log, output_rhf_log, output_err)

    def lll_with_deep_insertions(self, delta: float = 0.99, eta: float = 0.5, gamma: int = 20, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[Lattice, list[float], list[float], float]:
        """Alias for deep_lll method.
        
        ## Reference
        - C.-P. Schnorr and M. Euchner. Lattice basis reduction: Improved practical algorithms and solving subset sum problems. 1994

        Args:
            delta (float, optional): The delta parameter for Deep LLL reduction. Defaults to 0.99.
            eta (float, optional): The eta parameter for size reduction. Defaults to 0.5.
            gamma (int, optional): The gamma parameter for Deep LLL reduction. Defaults to 20.
            output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
            output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
            output_err (bool, optional): Whether to output the error. Defaults to False.

        Returns:
            Lattice: The reduced basis.
        """
        return self.deep_lll(delta, eta, gamma, output_sl_log, output_rhf_log, output_err)

    def qr_deep_lll(self, delta: float = 0.99, eta: float = 0.5, gamma: int = 20, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[Lattice, list[float], list[float]]:
        """Perform QR-Deep LLL reduction on the lattice basis with given delta, eta, and gamma parameters.

        Args:
            delta (float, optional): The delta parameter for QR-Deep LLL reduction. Defaults to 0.99.
            eta (float, optional): The eta parameter for size reduction. Defaults to 0.5.
            gamma (int, optional): The gamma parameter for QR-Deep LLL reduction. Defaults to 20.
            output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
            output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
            output_err (bool, optional): Whether to output the error. Defaults to False.

        Returns:
            Lattice: The reduced basis.
        """
        reduced_basis, sl_log, rhf_log, err = reduction.qr_deep_lll(self.basis, delta, eta, gamma, output_sl_log, output_rhf_log, output_err)
        return Lattice(reduced_basis), sl_log, rhf_log, err

    def pot_lll(self, delta: float = 0.99, eta: float = 0.5, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[Lattice, list[float], list[float], float]:
        """Perform Potential LLL reduction on the lattice basis with given delta and eta parameters.
        
        ## Reference
        - F. Fontein and M. Schneider and U. Wagner. PotLLL: A polynomial time version of LLL with deep insertions. 2014

        Args:
            delta (float, optional): The delta parameter for Potential LLL reduction. Defaults to 0.99.
            eta (float, optional): The eta parameter for size reduction. Defaults to 0.5.
            output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
            output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
            output_err (bool, optional): Whether to output the error. Defaults to False.

        Returns:
            Lattice: The reduced basis.
        """
        reduced_basis, sl_log, rhf_log, err = reduction.pot_lll(self.basis, delta, eta, output_sl_log, output_rhf_log, output_err)
        return Lattice(reduced_basis), sl_log, rhf_log, err
    
    def potential_lll(self, delta: float = 0.99, eta: float = 0.5, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[Lattice, list[float], list[float], float]:
        """Alias for pot_lll method.
        
        ## Reference
        - F. Fontein and M. Schneider and U. Wagner. PotLLL: A polynomial time version of LLL with deep insertions. 2014

        Args:
            delta (float, optional): The delta parameter for Potential LLL reduction. Defaults to 0.99.
            eta (float, optional): The eta parameter for size reduction. Defaults to 0.5.
            output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
            output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
            output_err (bool, optional): Whether to output the error. Defaults to False.

        Returns:
            Lattice: The reduced basis.
        """
        return self.pot_lll(delta, eta, output_sl_log, output_rhf_log, output_err)
    
    def enum_sv(self, pruning: bool = False) -> np.ndarray[int]:
        """Enumerates the shortest vector in the lattice basis using the SVP algorithm.
        
        ## Reference
        - N. Gama and P. Q. Nguyen and O. Regev. Lattice enumeration using extreme pruning. 2010

        Args:
            pruning (bool, optional): Whether to use pruning. Defaults to False.

        Returns:
            np.ndarray[int]: The shortest vector found in the lattice.
        """
        return svp.enum_sv(self.basis, pruning)

    def bkz(self, delta: float = 0.99, beta: int = 20, max_loops: int = -1, pruning: bool = False, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[Lattice, list[float], list[float], float]:
        """Perform BKZ reduction on the lattice basis with given delta and beta parameters.
        
        ## Reference
        - C.-P. Schnorr and M. Euchner. Lattice basis reduction: Improved practical algorithms and solving subset sum problems. 1994

        Args:
            delta (float, optional): The delta parameter for BKZ reduction. Defaults to 0.99.
            beta (int, optional): The block size parameter for BKZ reduction. Defaults to 20.
            max_loops (int, optional): The maximum number of tours through the basis. Defaults to -1 (no limit).
            pruning (bool, optional): Whether to use pruning in the SVP solver. Defaults to False.
            output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
            output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
            output_err (bool, optional): Whether to output the error. Defaults to False.

        Returns:
            Lattice: The reduced basis.
        """
        reduced_basis, sl_log, rhf_log, err = reduction.bkz(self.basis, delta, beta, max_loops, pruning, output_sl_log, output_rhf_log, output_err)
        return Lattice(reduced_basis), sl_log, rhf_log, err

    def block_korkine_zolotarev(self, delta: float = 0.99, beta: int = 20, max_loops: int = -1, pruning: bool = False, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[Lattice, list[float], list[float], float]:
        """Alias for bkz method.
        
        ## Reference
        - C.-P. Schnorr and M. Euchner. Lattice basis reduction: Improved practical algorithms and solving subset sum problems. 1994

        Args:
            delta (float, optional): The delta parameter for BKZ reduction. Defaults to 0.99.
            beta (int, optional): The block size parameter for BKZ reduction. Defaults to 20.
            max_loops (int, optional): The maximum number of tours through the basis. Defaults to -1 (no limit).
            pruning (bool, optional): Whether to use pruning in the SVP solver. Defaults to False.
            output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
            output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
            output_err (bool, optional): Whether to output the error. Defaults to False.

        Returns:
            Lattice: The reduced basis.
        """
        return self.bkz(delta, beta, max_loops, pruning, output_sl_log, output_rhf_log, output_err)
    
    def deep_bkz(self, delta: float = 0.99, beta: int = 20, gamma: int = 20, max_loops: int = -1, pruning: bool = False, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[Lattice, list[float], list[float], float]:
        """Perform Deep BKZ reduction on the lattice basis with given delta, beta, and gamma parameters.

        Args:
            delta (float, optional): The delta parameter for Deep BKZ reduction. Defaults to 0.99.
            beta (int, optional): The block size parameter for Deep BKZ reduction. Defaults to 20.
            gamma (int, optional): The gamma parameter for Deep BKZ reduction. Defaults to 20.
            max_loops (int, optional): The maximum number of tours through the basis. Defaults to -1 (no limit).
            pruning (bool, optional): Whether to use pruning in the SVP solver. Defaults to False.
            output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
            output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
            output_err (bool, optional): Whether to output the error. Defaults to False.

        Returns:
            Lattice: The reduced basis.
        """
        reduced_basis, sl_log, rhf_log, err = reduction.deep_bkz(self.basis, delta, beta, gamma, max_loops, pruning, output_sl_log, output_rhf_log, output_err)
        return Lattice(reduced_basis), sl_log, rhf_log, err
    
    def qr_bkz(self, delta: float = 0.99, beta: int = 20, max_loops: int = -1, pruning: bool = False, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[Lattice, list[float], list[float], float]:
        """Perform QR-BKZ reduction on the lattice basis with given delta and beta parameters.

        Args:
            delta (float, optional): The delta parameter for QR-BKZ reduction. Defaults to 0.99.
            beta (int, optional): The block size parameter for QR-BKZ reduction. Defaults to 20.
            max_loops (int, optional): The maximum number of tours through the basis. Defaults to -1 (no limit).
            pruning (bool, optional): Whether to use pruning in the SVP solver. Defaults to False.
            output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
            output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
            output_err (bool, optional): Whether to output the error. Defaults to False.

        Returns:
            Lattice: The reduced basis.
        """
        reduced_basis, sl_log, rhf_log, err = reduction.qr_bkz(self.basis, delta, beta, max_loops, pruning, output_sl_log, output_rhf_log, output_err)
        return Lattice(reduced_basis), sl_log, rhf_log, err
    
    def qr_deep_bkz(self, delta: float = 0.99, beta: int = 20, gamma: int = 20, max_loops: int = -1, pruning: bool = False, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[Lattice, list[float], list[float], float]:
        """Perform QR-Deep BKZ reduction on the lattice basis with given delta, beta, and gamma parameters.

        Args:
            delta (float, optional): The delta parameter for QR-Deep BKZ reduction. Defaults to 0.99.
            beta (int, optional): The block size parameter for QR-Deep BKZ reduction. Defaults to 20.
            gamma (int, optional): The gamma parameter for QR-Deep BKZ reduction. Defaults to 20.
            max_loops (int, optional): The maximum number of tours through the basis. Defaults to -1 (no limit).
            pruning (bool, optional): Whether to use pruning in the SVP solver. Defaults to False.
            output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
            output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
            output_err (bool, optional): Whether to output the error. Defaults to False.
        Returns:
            Lattice: The reduced basis.
        """
        reduced_basis, sl_log, rhf_log, err = reduction.qr_deep_bkz(self.basis, delta, beta, gamma, max_loops, pruning, output_sl_log, output_rhf_log, output_err)
        return Lattice(reduced_basis), sl_log, rhf_log, err
    
    def hkz(self, pruning: bool = False, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[Lattice, list[float], list[float], float]:
        """Perform HKZ reduction on the lattice basis with given parameters.
        
        ## Reference
        - C.-P. Schnorr and M. Euchner. Lattice basis reduction: Improved practical algorithms and solving subset sum problems. 1994

        Args:
            pruning (bool, optional): Whether to use pruning in the SVP solver. Defaults to False.
            output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
            output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
            output_err (bool, optional): Whether to output the error log. Defaults to False.

        Returns:
            Lattice: The reduced basis.
        """
        reduced_basis, sl_log, rhf_log, err = reduction.hkz(self.basis, pruning, output_sl_log, output_rhf_log, output_err)
        return Lattice(reduced_basis), sl_log, rhf_log, err

    def pot_bkz(self, delta: float = 0.99, beta: int = 20, max_loops: int = -1, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[Lattice, list[float], list[float], float]:
        """Perform Potential BKZ reduction on the lattice basis with given delta and beta parameters.
        
        ## Reference
        - 佐藤 新 and 安田 雅哉, 自己双対型PotBKZ基底簡約の提案とBKZとの比較, 2025

        Args:
            delta (float, optional): The delta parameter for Potential BKZ reduction. Defaults to 0.99.
            beta (int, optional): The block size parameter for Potential BKZ reduction. Defaults to 20.
            max_loops (int, optional): The maximum number of tours through the basis. Defaults to -1 (no limit).
            output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
            output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
            output_err (bool, optional): Whether to output the error. Defaults to False.

        Returns:
            Lattice: The reduced basis.
        """
        reduced_basis, sl_log, rhf_log, err = reduction.pot_bkz(self.basis, delta, beta, max_loops, output_sl_log, output_rhf_log, output_err)
        return Lattice(reduced_basis), sl_log, rhf_log, err
    
    def dual_deep_lll(self, delta: float = 0.99, gamma: int = 20, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[Lattice, list[float], list[float], float]:
        """Perform Dual Deep LLL reduction on the lattice basis with given delta and gamma parameters.

        Args:
            delta (float, optional): The delta parameter for Dual Deep LLL reduction. Defaults to 0.99.
            gamma (int, optional): The gamma parameter for Dual Deep LLL reduction. Defaults to 20.
            output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
            output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
            output_err (bool, optional): Whether to output the error. Defaults to False.

        Returns:
            Lattice: The reduced basis.
        """
        reduced_basis, sl_log, rhf_log, err = reduction.dual_deep_lll(self.basis, delta, gamma, output_sl_log, output_rhf_log, output_err)
        return Lattice(reduced_basis), sl_log, rhf_log, err
    
    def dual_pot_lll(self, delta: float = 0.99, output_sl_log: bool = False, output_rhf_log: bool = False, output_err: bool = False) -> tuple[Lattice, list[float], list[float], float]:
        """Perform Dual Potential LLL reduction on the lattice basis with given delta and eta parameters.
        
        ## Reference
        - 佐藤 新 and 安田 雅哉, 自己双対型PotBKZ基底簡約の提案とBKZとの比較, 2025

        Args:
            delta (float, optional): The delta parameter for Dual Potential LLL reduction. Defaults to 0.99.
            output_sl_log (bool, optional): Whether to output the GSA-slope log. Defaults to False.
            output_rhf_log (bool, optional): Whether to output the RHF log. Defaults to False.
            output_err (bool, optional): Whether to output the error. Defaults to False.

        Returns:
            Lattice: The reduced basis.
        """
        reduced_basis, sl_log, rhf_log, err = reduction.dual_pot_lll(self.basis, delta, output_sl_log, output_rhf_log, output_err)
        return Lattice(reduced_basis), sl_log, rhf_log, err

def svp_challenge(dim: int, seed: int) -> Lattice:
    """Return the basis of the SVP challenge lattice.

    Returns:
        Lattice: The basis of the SVP challenge lattice.
    """
    with resources.files("latpy.svp_challenge_list").joinpath(f"svp_challenge_{dim}_{seed}.txt").open('r') as f:
        basis = np.loadtxt(f, dtype=np.int64)[: dim, :dim]
    return Lattice(basis)
