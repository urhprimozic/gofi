import numpy as np

def get_matrices(solution_y, index, dim):
    """
    Returns the matrices R and S from the solution_y array at the given index.
    The solution_y array is expected to be in the format where the first half
    of the array corresponds to the flattened matrix R and the second half
    corresponds to the flattened matrix S.
    """ 
    z = solution_y.transpose()[index]
    R = np.reshape(z[:len(z)//2], (dim, dim))
    S = np.reshape(z[len(z)//2:], (dim, dim))
    return R, S

def get_characters(solution_y, dim):
    """
    Returns the characters of the matrices R and S from the solution_y array.
    The solution_y array is expected to be in the format where the first half
    of the array corresponds to the flattened matrix R and the second half
    corresponds to the flattened matrix S.
    """
    solutions = solution_y.transpose()
    char_R = []
    char_S=[]
    for z in solutions:
            R = np.reshape(z[:len(z)//2], (dim, dim))
            S = np.reshape(z[len(z)//2:], (dim, dim))
            char_R.append(np.trace(R))
            char_S.append(np.trace(S))

    return char_R, char_S