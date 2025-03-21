'''
Demo showcasing different speeds of convergence for 
model of S3 -> O(2).

- O(2) is parametrised by real numbers --> only two parameters!
- only relation loss and loss of irreducability are used, since all the matrices are already ortogonal
'''
import torch

def sign(x):
    if x > 0:
        return 1
    return -1

def param_to_matrix(a, b_sign=1, c_sign=1, d_sign=1):
    '''
    Maps real numbers to matrices as:
    a --> [[a,  b]
           [c,  d]], 
    where a^2+c^2=b^2+d^2=1 and ab + cd = 0

    

    '''
    assert abs(b_sign) == 1 and abs(c_sign == 1) and abs(d_sign) == 1
    assert a.shape == torch.Size([])

    if a == 0:
        return torch.tensor([[ a , b_sign * 1],
                              [ c_sign * 1,     0]])
    else:
        c = c_sign * torch.sqrt(1 - a**2)
        b = -  c 
        d = d_sign * torch.sqrt(1 - b**2)

        return torch.tensor([[ a , b],
                              [ c,     d]])

