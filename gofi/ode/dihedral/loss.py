# definiraj grad_L
import numpy as np
from numpy.linalg import norm
from numpy.linalg import matrix_power as pow
from abc import ABC, abstractmethod

# helper functions to compute the derivative of the loss function
def Q(X, n):
    """
    Returns 1/2 * (  d/dX (||X^n - I||^2) )^T
    """
    dim = X.shape[0]
    ans = np.zeros(X.shape)
    for i in range(n):
        ans += pow(X, n - i - 1) @ pow(X.transpose(), n) @ pow(X, i)
    ans -= n * pow(X, n - 1)
    return ans


def norm_square_RS_minus_I_by_R(R, S):
    """
    Returns 2(  SRS(RS)^{T^2} + S(RS)^{T^2} RS - 2SRS) ^T
    """
    RST = pow((R @ S).transpose(), 2)
    ans = S @ R @ S @ RST + S @ RST @ R @ S - 2 * S @ R @ S
    return 2 * ans.transpose()


def norm_square_RS_minus_I_by_S(R, S):
    """
    Returns  2  (RS (RS)^{T^2} R + (RS)^{T^2}RSR - 2RSR) ^T
    """
    RST = pow((R @ S).transpose(), 2)
    ans = R @ S @ RST @ R + RST @ R @ S @ R - 2 * R @ S @ R
    return 2 * ans.transpose()


#############################################################
## Loss functions
#############################################################
class LossDihedral(ABC):
    """Abstract class for loss functions."""
    @abstractmethod
    def __init__(self, *args):
        """Initialize the loss function with given arguments."""
        pass
    @abstractmethod
    def __call__(self, R, S):
        """Compute the loss given the parameters R and S."""
        pass
    @abstractmethod    
    def dR(self, R, S):
        """Compute the gradient d/dR of the loss with respect to R."""
        pass
    @abstractmethod    
    def dS(self, R, S):
        """Compute the gradient d/dS of the loss with respect to S."""
        pass


class RelationLoss(LossDihedral):
    """Relation loss 1/3(||R^n - I||^2 + ||S^2 - I||^2 + ||RS^2 - I||^2)"""

    def __init__(self, n):
        """
        Initialize the relation loss with respect to the dihedral group D_2n.

        Parameters:
        ----------
        n : int
            Order of rotation. Order of dihedral group D_2n is equal to 2n.
        """
        self.n = n

    def __call__(self, R, S):
        """
        Computes relation loss of the mapping (r -> R, s -> S) with respect to the dihedral group D_2n.
        The relation loss is defined as:
        1/3 * (||R^n - I||^2 + ||S^2 - I||^2 + ||RS^2 - I||^2)
        """
        dim = R.shape[0]
        eye = np.eye(dim)
        return (1 / 3) * (
            norm(pow(R, self.n) - eye, ord="fro") ** 2
            + norm(pow(S, 2) - eye, ord="fro") ** 2
            + norm(pow(R @ S, 2) - eye, ord="fro") ** 2
        )

    def dR(self, R, S):
        """Compute the gradient d/dR of the loss with respect to R."""
        return (1 / 3) * (
            2 * Q(R, self.n).transpose() + norm_square_RS_minus_I_by_R(R, S)
        )

    def dS(self, R, S):
        """
        Computes the gradient d/dS of the relation loss with respect to S.
        """
        return (1 / 3) * (2 * Q(S, 2).transpose() + norm_square_RS_minus_I_by_S(R, S))


def norm_chi(R, S, n):
    """
    Computes the norm of the character of the representation (R, S) with respect to the dihedral group D_2n.
    The norm is defined as:
    1/2n * (sum_{i=0}^{n-1} |tr(R^i)|^2 + sum_{i=0}^{n-1} |tr(R^i S)|^2)
    """
    ans = 0
    for i in range(n):
        ans += np.abs(np.trace(pow(R, i))) ** 2
        ans += np.abs(np.trace(pow(R, i) @ S)) ** 2
    ans /= 2 * n
    return ans


class IrreducibilityLoss(LossDihedral):
    """Irreducibility loss (|character(R,S)| - 1)^2"""

    def __init__(self, n):
        """
        Initialize the irreducibility loss with respect to the dihedral group D_2n.

        Parameters:
        ----------
        n : int
            Order of rotation. Order of dihedral group D_2n is equal to 2n.
        """
        self.n = n

    def __call__(self, R, S):
        """
        Computes irreducibility loss of the mapping (r -> R, s -> S) with respect to the dihedral group D_2n.
        The irreducibility loss is defined as (|chi(R,S,n)|^2 - 1)^2
        """
        return (norm_chi(R, S, self.n) - 1) ** 2

    def dR(self, R, S):
        """
        Computes the gradient d/dR of the irreducibility loss with respect to R.
        """
        dim = R.shape[0]
        ans = np.zeros((dim, dim))
        for i in range(1, self.n):
            ans += np.trace(pow(R, i)) * i * pow(R, i - 1)
            ans += np.trace(pow(R, i) @ S) * (
                sum([pow(R, i - j - 1) @ S @ pow(R, j) for j in range(i)])
            )
        ans = ans.transpose()
        ans *= 2 * (norm_chi(R, S, self.n) - 1) / self.n
        return ans

    def dS(self, R, S):
        """
        Computes the gradient d/dS of the irreducibility loss with respect to S.
        """
        dim = R.shape[0]
        ans = np.zeros((dim, dim))
        for i in range(0, self.n):
            ans += np.trace(pow(R, i) @ S) * pow(R, i)
        ans = ans.transpose()
        ans *= 2 * (norm_chi(R, S, self.n) - 1) / self.n
        return ans


def unitary_norm(X):
    """
    Returns d/dx (||X^T X - I||^2)
    """
    return 4 * (X @ X.transpose() @ X - X)


class UnitaryLoss(LossDihedral):
    """Unitary loss 1/2 * (||R^T R - I||^2 + ||S^T S - I||^2)"""

    def __init__(self, *args):
        pass

    def __call__(self, R, S):
        """
        Computes unitary loss of the mapping (r -> R, s -> S) with respect to the dihedral group D_2n.
        The unitary loss is defined as:

        Returns 1/2 * (||R^T R - I||^2 + ||S^T S - I||^2)
        """
        dim = R.shape[0]
        eye = np.eye(dim)
        return (1 / 2) * (
            norm(R.transpose() @ R - eye, ord="fro") ** 2
            + norm(S.transpose() @ S - eye, ord="fro") ** 2
        )

    def dR(self, R, S):
        """
        Computes the gradient d/dR of the unitary loss with respect to R.
        """
        return 0.5 * unitary_norm(R)

    def dS(self, R, S):
        """
        Computes the gradient d/dS of the unitary loss with respect to S.
        """
        return 0.5 * unitary_norm(S)

class OrtogonalLoss(LossDihedral):
    def __init__(self, n : int, R0 : np.array , S0 : np.array):
        """
        Creates loss for ortogonality of characthers. OrtogonalLoss(n, R0, S0)(R, S) is equal to the square of  scalar product 
        between mapping (r -> R0, s -> S0) and (r -> R, s -> S).


        Parameters
        ----------
        n : int 
            Dihedral group D2n number.
        R1 : np.array
            Map of rotation
        S1 : np.array 
            Map of flip
        """
        self.n = n 
        self.R0 = R0 
        self.S0 = S0
    def prod(self, R, S):
        """
        Returns scalar product between mapping rho:(r -> R, s -> S) and rho1:(r -> R1, s -> S1)
        [rho, rho1] = 1/(2n) sum_{g in Dn} tr(rho(g) tr(rho1(g) 
        """
        ans = 0
        for i in range(self.n):
            ans += np.trace(pow(self.R0, i)) * np.trace(pow(R, self.n-i))
            ans += np.trace(pow(self.R0, i) @ self.S0) * np.trace(S @ pow(R, self.n-i))
        ans /= 2 * self.n
        return ans
    def __call__(self, R, S):
        """
        Returns square of  scalar product between mapping rho:(r -> R, s -> S) and rho1:(r -> R1, s -> S1)
        [rho, rho1] = 1/(2n) sum_{g in Dn} tr(rho(g) tr(rho1(g) 
        """ 
        return self.prod(R, S) ** 2

    def dR(self, R, S):
        """
        Returns d/dR(OrtogonalLoss)
        """ 
        ans = np.zeros(R.shape)
        for i in range(self.n):
            ans += np.trace(pow(self.R0, i)) * (self.n - i) * pow(R, self.n - i - 1)
            ans += np.trace(pow(self.R0, i) @ self.S0) * S * (self.n - i) * pow(R, self.n - i - 1)
        ans /= 2 * self.n
        ans *= self.prod(R, S)
        return ans  
    def dS(self,  R, S):
        """
        Returns d/dS(OrtogonalLoss)
        """ 
        ans = np.zeros(S.shape)
        for i in range(self.n):
            ans += np.trace(pow(self.R0, i) @ self.S0) *  pow(R, self.n - i)
        ans /= 2 * self.n
        ans *= self.prod(R, S)
        return ans