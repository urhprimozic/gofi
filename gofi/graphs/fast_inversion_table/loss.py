import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch







def loss_slow(Pa, M1, M2):
    """
    Loss in O(n^4). 
    Loss = 1/n^2 sum_{i, j} |M1_{i,j} - P(f(i) ~ f(j))|
    """
    print("loss_slow is deprecated, use loss instead")
    


def mask(Pa):
    """
    1 where Pa!=0, 0 where Pa==0
    """
    return (Pa != 0).float()


def S_low(Pa):
    """
    S_low(Pa, M1, M2)[m, i] = sum_{z = 1} ^{i} Pa[m, z]


    """
    return torch.cumsum(Pa, dim=1) * mask(Pa)
    
def S_high(Pa):
    """
    S_high(Pa, M1, M2)[m, i] = sum_{z = i} ^{n} Pa[m, z]
    """
    return torch.cumsum(torch.flip(Pa, dims=[1]), dim=1).flip(dims=[1]) * mask(Pa)

def S_mid(Pa):
    """
    S_mid(Pa, M1, M2)[m, i, j] = sum_{z = i} ^{j} Pa[m, z]
    """
    n = Pa.shape[1]
    S_low_mat = S_low(Pa).unsqueeze(1).repeat(1, n, 1)
    S_high_mat = S_high(Pa).unsqueeze(2).repeat(1, 1, n)
    return torch.min(S_low_mat, S_high_mat) * (mask(Pa).unsqueeze(1).repeat(1, n, 1))

def q(Pa, M1, M2):
    n = Pa.shape[0]
    
    Q = torch.zeros((n,n, n)).to(device) 

    # base - q(j, m, m) = P(m, j)
    m_idx = torch.arange(n, device=Q.device)
    j_idx = torch.arange(n, device=Q.device)
    J, M = torch.meshgrid(j_idx, m_idx, indexing='ij')  # shape (n,n)
    Q[J, M, M] = Pa[M, J]

    # recursive case
    # q(j, h, m) = q(j-1, h, m+1) * S_low[m-1, i-1] + q(j, h, m+1) * S_high[m-1, j]
    for j in range(1, n):
        for m in range(n - 1):
            for h in range(m + 1):
                Q[j, h, m] = (
                    Q[j - 1, h, m + 1] * S_low(Pa)[m - 1, j - 1]
                    + Q[j, h, m + 1] * S_high(Pa)[m - 1, j]
                )

def q_recursive(Pa, M1, M2):
    n = Pa.shape[0]
    
    Q = torch.fill((n,n, n), torch.nan).to(device)

    Slow = S_low(Pa)
    Shigh = S_high(Pa)

    def f(j, h, m):
        if not torch.isnan(Q[j, h, m]):
            return Q[j, h, m]
        ans = 0
        if h < m:
            ans  = 0
        elif h == m:
            ans = Pa[m -1, j -1]
        else:
            ans = f(j - 1, h, m + 1) * Slow[m - 1, j - 1] + f(j, h, m + 1) * Shigh[m - 1, j]
        Q[j, h, m] = ans
        return ans 
    for h in range(1, n+1):
        f(n, h, 1)
    return Q

def p_recursive(Pa, M1, M2):
    n = Pa.shape[0]
  
    P = torch.fill((n, n, n, n), torch.nan).to(device)

    Q = q_recursive(Pa, M1, M2)
    Slow = S_low(Pa)
    S_high = S_high(Pa)
    Smid = S_mid(Pa)

    def g(i, k, j, m):
        if i > j:
            i,j = j,i
            k,h = h,k
        if k == h or k < m or h < m or k > n or h > n or m > n:
            return 0
        if not torch.isnan(P[i - 1, k - 1, j - 1, h - 1]):
            return P[i - 1, k - 1, j - 1, h - 1]
        ans = 0
        if m == k:
            ans = Q[j - 1, h, m + 1] * Pa[m-1, i-1]
        elif m == h:
            ans = Q[i, k, m + 1] * Pa[m-1, j-1]
        else:
            ans = g(i-1, k, j-1, h, m+1) * Slow[m-1, i-1] 
            + g(i, k, j+1, h, m+1) * Smid[m-1, i, j-2] 
            + g(i, k, j, h, m+1) * S_high[m-1, j] 
        P[i - 1, k - 1, j - 1, h - 1] = ans
        return ans
    for i in range(1, n+1):
        raise NotImplementedError("Not finished")        
    

    

        
    

if "__main__" == __name__:
    # simple test
    _mask = torch.flip(torch.triu(torch.ones(4, 4)), dims=[1])
    X = torch.arange(1, 17).float().reshape(4, 4) * _mask


    



