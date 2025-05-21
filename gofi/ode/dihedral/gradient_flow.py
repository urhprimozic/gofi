import numpy as np
from scipy.integrate import solve_ivp

class GradientFlow:
    """
    Implements gradient flow equation. Supports solving it on initial points.
    Converts matrix losses to numpy arrays and solve until convergence. 
    """
    def __init__(self, dim,  loss_function, clipping_limit=None):
        """
        Create new gradient flow object.
        Parameters
        -----------
        dim : int
            Dimension of the matrices R and S.
        loss_function : LossDihedral
            Loss function to be used for the gradient flow.
        clipping_limit : float
            Clipping limit for the gradients. If None, no clipping is applied.

        """
        self.loss = loss_function
        self.dim = dim
        self.clipping_limit = clipping_limit
        
    def neg_grad(self, t, z):
        # collect matrices
        R = np.reshape(z[:len(z)//2], (self.dim, self.dim))
        S = np.reshape(z[len(z)//2:], (self.dim, self.dim))

        # calculate gradient
        dR = self.loss.dR(R,S)
        dS = self.loss.dS(R,S)
        
        # stack gradient back to vector form
        dR = dR.flatten()
        dS =  dS.flatten()

        P = np.concatenate((dR, dS), axis=None)

        # clip to avoid too big gradients 
        if self.clipping_limit is not None:
            P = np.clip(P, -self.clipping_limit, self.clipping_limit)
        return P

    def vec_to_matrix(self, P):
        """
        Converts vector P to two matrices R and S.
        """
        R = np.reshape(P[:len(P)//2], (self.dim, self.dim))
        S = np.reshape(P[len(P)//2:], (self.dim, self.dim))
        return (R, S)
    
    def matrix_to_vec(self, R, S):
        """
        Converts two matrices R and S to vector P.
        """
        return np.concatenate((R.flatten(), S.flatten()), axis=None)
        
    def solve(self, R0, S0, eps=0.0001, t_max=50):
        """
        Solves system of ODE (dR = loss.dR, dS = loss.dS) with initial values R(0)=0, S(0)=0. 
        Stops integrating, when norm(loss.dR) + norm(loss.dS) < eps (converged to something)
        or when system reaches the end of the integrating interval.

        Returns
        -----------
        Same as scipy.solve_ivp()
        """
        # create a new event function
        def event(t, z):
            # get size of gradients 
            grads = self.neg_grad(t,z)
            norm_grads = np.linalg.norm(grads)
            return norm_grads - eps 
        # set event properties
        event.terminal = True  # stop integration, when reaching norm_grads == eps 
        event.direction = -1 # only triggers, when event goes from positive to negative --> when this happens, norm_grad < eps!!

        # solve 
        P0 = self.matrix_to_vec(R0, S0)
        return solve_ivp(self.neg_grad, (0, t_max), P0, events=event)
