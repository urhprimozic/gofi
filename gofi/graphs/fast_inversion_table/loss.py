import torch
from functools import lru_cache
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pa naj bo matrika velikosti (n+1)x(n+1)
# Pa[m][z] = P(a_m = z)

def S(Pa, i, j, m):
    if j < i:
        return 0.0
    return torch.sum(Pa[m, i:j+1])

@lru_cache(maxsize=None)
def q(Pa, j, h, m, n):
    if j <= 0:
        return 0.0
    if m == h:
        return Pa[m, j]
    if m == n:
        return 1.0
    return (q(Pa, j-1, h, m+1, n) * S(Pa, 1, j-1, m) +
            q(Pa, j, h, m+1, n) * S(Pa, j+1, n-m, m))

@lru_cache(maxsize=None)
def P(Pa, i, k, j, h, m, n):
    if i <= 0 or j <= 0:
        return 0.0
    if m == n:
        return 1.0
    if m == k:
        return Pa[k, i] * q(Pa, j-1, h, k+1, n)
    if m == h:
        return Pa[h, j] * q(Pa, i, k, h+1, n)
    return (P(Pa, i-1, k, j-1, h, m+1, n) * S(Pa, 1, i-1, m) +
            P(Pa, i, k, j-1, h, m+1, n) * S(Pa, i+1, j-1, m) +
            P(Pa, i, k, j, h, m+1, n) * S(Pa, j+1, n-m, m))

def loss(Pa, M):
    n = M.shape[0]
    L = 0.0
    for i in range(1, n+1):
        for j in range(1, n+1):
            inner_sum = 0.0
            for k in range(1, n+1):
                for h in range(1, n+1):
                    inner_sum += P(Pa, i, k, j, h, 1, n)
            L += (M[i-1, j-1] - inner_sum)**2
    return L


def monte_carlo_loss(Pa, M, num_samples=100):
    """
    Monte Carlo loss: vzorči naključne (i,j) pare in računa stochastic loss.
    
    Args:
        Pa : matrika verjetnosti P(a_m = z)
        M : ciljna matrika (n x n)
        num_samples : število vzorčenih (i,j) parov
        
    Returns:
        float, približek lossa
    """
    n = M.shape[0]
    L = 0.0
    for _ in range(num_samples):
        i = random.randint(1, n)
        j = random.randint(1, n)
        
        inner_sum = 0.0
        for k in range(1, n+1):
            for h in range(1, n+1):
                inner_sum += P(Pa, i, k, j, h, 1, n)
        
        L += (M[i-1, j-1] - inner_sum)**2
    
    # normaliziramo, da je primerljivo z "polnim" lossom
    return (n**2 / num_samples) * L