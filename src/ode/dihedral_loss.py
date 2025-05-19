# definiraj grad_L
import numpy as np

# from numpy.linalg import matrix_power as pow
from numpy.linalg import norm


def pow(A, n):
    dim = A.shape[0]
    X = np.eye(dim)
    for i in range(n):
        X = X @ A
    return X

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

def Q(X, n):
    """
    Returns 1/2 * (  d/dX (||X^n - I||^2) )^T
    """
    dim = X.shape[0]
    ans = np.zeros(X.shape)
    for i in range(n):
        ans += pow(X, n - i - 1) @ pow(X.transpose(), n) @ pow(X, i)
    ans -= n * pow(X, n-1)
    return ans


def d_Lrel_dR(R, S, n):
    return (1 / 3) *( 2 * Q(R, n).transpose() +   norm_square_RS_minus_I_by_R(R, S) )


def d_Lrel_dS(R, S):
    return (1 / 3) * (2 * Q(S, 2).transpose() + norm_square_RS_minus_I_by_S(R, S))


def Lrel(R, S, n):
    dim = R.shape[0]
    eye = np.eye(dim)
    return (1 / 3) * (norm(pow(R,n) - eye, ord='fro')**2 
                    + norm(pow(S,2) - eye, ord='fro')**2
                    + norm(pow(R@S,2) - eye, ord='fro') **2)

def norm_chi(R, S, n):
    ans = 0
    for i in range(n):
        ans += np.abs(np.trace(pow(R, i))) ** 2
        ans += np.abs(np.trace(pow(R, i) @ S)) ** 2
    ans /= 2 * n
    return ans


def Lirr(R, S, n):
    return (norm_chi(R, S, n) - 1) ** 2


def d_Lirr_dR(R, S, n):
    """
    Returns
      2 * (  d/dR (||chi(R,S,n)||^2 - 1) )^T, which is equal to
    2/n(1 - |chi|) 
(
sum_{i=1}^{n-1} tr(R^i)iR^{i-1} + 
tr(R^iS)(sum_{j=0}^{i-1}  R^{i-j-1}SR^j)         
)^T$
    """
    dim = R.shape[0]
    ans = np.zeros((dim, dim))
    for i in range(1, n):
        ans += np.trace(pow(R, i)) * i * pow(R, i - 1)
        ans += np.trace(pow(R, i) @ S) * (sum([pow(R, i - j - 1) @ S @ pow(R, j) for j in range(i)]))
    ans = ans.transpose()
    ans *= 2 * (norm_chi(R, S, n) - 1) / n
    return ans


def d_Lirr_dS(R, S, n):
    """
    Returns 2/n(1 - |chi|)
  (sum_{i=0}^{n-1} tr(R^iS)R^i      ) ^T
    """
    dim = R.shape[0]
    ans = np.zeros((dim, dim))
    for i in range(0, n):
        ans += np.trace( pow(R, i)@S) * pow(R, i)
    ans = ans.transpose()
    ans *= 2 * (norm_chi(R, S, n) -1) / n
    return ans

def Lunitary(R, S):
    """
    Returns 1/2 * (||R^T R - I||^2 + ||S^T S - I||^2)
    """
    dim = R.shape[0]
    eye = np.eye(dim)
    return (1 / 2) * (norm(R.transpose() @ R - eye, ord='fro')**2 + norm(S.transpose() @ S - eye, ord='fro')**2)
def unitary_norm(X):
    """
    Returns d/dx (||X^T X - I||^2)
    """ 
    return 4 * (X @ X.transpose() @ X - X )

def d_Lunitary_dR(R):
    """
    Returns 2 * (d/dR (||R^T R - I||^2))^T
    """
    return 0.5 *unitary_norm(R)
def d_Lunitary_dS(S):
    """
    Returns 2 * (d/dS (||S^T S - I||^2))^T
    """
    return 0.5 * unitary_norm(S)

def L(R, S, n):
    """
    Returns Lrel + Lirr + Lunitary
    """
    return Lrel(R, S, n) + Lirr(R, S, n) + Lunitary(R, S)
def d_L_dR(R, S, n):
    """
    Returns d/dR (Lrel + Lirr + Lunitary)
    """ 
    return d_Lrel_dR(R, S, n) + d_Lirr_dR(R, S, n) + d_Lunitary_dR(R)   

def d_L_dS(R, S, n):
    """
    Returns d/dS (Lrel + Lirr + Lunitary)
    """ 
    return d_Lrel_dS(R, S) + d_Lirr_dS(R, S, n) + d_Lunitary_dS(S)