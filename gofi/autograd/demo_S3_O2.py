'''
Demo showcasing different speeds of convergence for 
model of S3 -> O(2).

- O(2) is parametrised by real numbers --> only two parameters!
- only relation loss and loss of irreducability are used, since all the matrices are already ortogonal
'''
import torch
from gofi.autograd.models import GeneratorModel
from gofi.autograd.loss import irr_loss_generator, relation_loss_generator
from gofi.groups import demo_S3
from gofi.autograd.grid import generate_grid

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
    assert a.shape in [torch.Size([]), torch.Size([1]) ]
    
    row1 = torch.concat((a , -torch.sqrt(1 - a**2)))
    row2 = torch.concat((torch.sqrt(1 - a**2) , -a))

    m = torch.concat((row1, row2), dim=0)
    m = torch.reshape(m, (2,2))
    return  m

def loss_function(model : GeneratorModel):
    return irr_loss_generator(model) + relation_loss_generator(model)

