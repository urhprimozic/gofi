from sympy.combinatorics.fp_groups import FpGroup
import sympy
from typing import List
from sympy import MatrixSymbol, Matrix

def matrix_element_symbol(name : str, generator : any, row :int, column : int) -> sympy.Symbol:
    """
    Returns a symbolic representation of name_{generator}_{row;column}, which is real.

    Args:
        name (str): The name of the group element.
        generator (int): Generator
        row (int): The row index in the matrix.
        column (int): The column index in the matrix.

    Returns:
        sympy.Symbol: A symbolic representation of the group element.
    """
    return sympy.Symbol(f"{{{name}_{generator}}}_{{{row};{column}}}", real=True)

def FpGroup_matrices(group: FpGroup, dim: int) -> dict[sympy.Symbol, sympy.Matrix]:
    """
    Returns list of smybolic dim x dim matrices  `[M_g for g in group.generators]` for a finitly presented group.

    Args:
        group (FpGroup): The free group.
        dim (int): The dimension.

    Returns:
        dict[sympy.Symbol, sympy.Matrix]: Dictionary mapping generators to corresponding matrices.
    """
    # hash using simbol, nbot using free group element
    
    # use this line for matrix symbols:
    #g_to_matrix = {g.letter_form[0] : Matrix(MatrixSymbol(f'M_{g}', dim, dim)) for g in group.generators}
   
   
    # g_to_matrix = {g.letter_form[0] : MatrixSymbol(f'M_{g}', dim, dim) for g in group.generators}


   #use this lines for x + iy

    g_to_matrix = {}
    for g in group.generators:
        
        elements = [(matrix_element_symbol("x", g, row, column) + sympy.I * matrix_element_symbol("y", g, row, column)) for row in range(dim) for column in range(dim)]
        matrix = Matrix(dim, dim, elements)

        g_to_matrix[g.letter_form[0]] = matrix
    return g_to_matrix


def FpGroup_L_irr(group: FpGroup, dim: int, expand=False) -> sympy.Symbol:
    """
    Returns symbolic loss L_irr for a finitly presented group and dimension dim.

    Args:
        group (FpGroup): Finitely generated group
        dim (int):  dimension of the representation
        expand (bool) : If True, loss.expand() is called() at the end. 

    Returns:
        sympy.Symbol: Loss of the  group.
    """
    # get matrices of generators
    matrices = FpGroup_matrices(group, dim)

    #L_irr = (  [1/|G| * sum_{g in G} |tr(M_g)|^2 ] -1)**2


    # where M_g is the matrix corresponding to the group element g
    
    
    character = 0
    for g in group.elements:
        #M_g = rho(g) = prod_{g_i in g} M_{g_i}
        
        matrix_of_element = sympy.Mul(sympy.eye(dim), *[matrices[generator]**power for (generator, power) in list(g)])
        character += sympy.Abs(sympy.trace(matrix_of_element))**2
    character = 1 / group.order() * character

    L_irr = (character - 1) ** 2
    if expand:
        #expand - this usually gets rid of the imaginary part
        L_irr = L_irr.expand()
        if sympy.im(L_irr) != 0:
            print("Warning: sympy.im(L_irr) != 0")
    return L_irr

def FpGroup_L_rel(group: FpGroup, dim: int, expand=False) -> sympy.Symbol:
    """
    Returns symbolic loss L_rel for a finitly presented group and dimension dim.

    Args:
        group (FpGroup): The free group.
        dim (int): The dimension.

    Returns:
        sympy.Symbol: The loss of the free group.
    """
    # get matrices of generators
    matrices = FpGroup_matrices(group, dim)

    #L_rel = 1/|R| * sum_{r in R} ||M_r - I||^2
    # where M_r is the matrix corresponding to the relator r
    L_rel = 0

    # iterate over relations
    for rel in group.relators:        
        matrix_of_relator = sympy.Mul(*[matrices[generator]**power  for (generator, power) in list(rel)])
        m_minus_eye = matrix_of_relator - sympy.eye(dim)
        L_rel += m_minus_eye.norm() ** 2
   
    # normalise loss
    L_rel = 1 / len(group.relators) * L_rel

    if expand:
        #expand - this usually gets rid of the imaginary part 
        L_rel = L_rel.expand()
        if sympy.im(L_rel) != 0:
            print("Warning: sympy.im(L_rel) != 0")
    return L_rel

from sympy.combinatorics.free_groups import free_group, vfree_group, xfree_group
from sympy.combinatorics.fp_groups import FpGroup, CosetTable, coset_enumeration_r

