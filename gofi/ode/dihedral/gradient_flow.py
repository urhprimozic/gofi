import numpy as np
from scipy.integrate import solve_ivp
import multiprocessing as mp
from tqdm import tqdm

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
        # negate
        P = -P

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
            
    def solve_random(self, min_value=-1, max_value=1, eps=0.0001, t_max=50):
            """
            Solves the gradient flow for a random initial point.
            """
            # generate random initial point
            R0 = np.random.uniform(min_value, max_value, (self.dim, self.dim))
            S0 = np.random.uniform(min_value, max_value, (self.dim, self.dim))
            
            # solve the gradient flow
            return self.solve(R0, S0, eps=eps, t_max=t_max)
    
    
    def solve_on_uniform_sample(self, n_samples : int, min_value :  float = -1,max_value : float=1, multiprocess=False, eps=0.0001, t_max=50, verbose=False):
        """
        Samples n_sample different initial parameters and runs solve for each one. 
        Every group element inside initial parameters is selected uniformly between min_value and max_value.

        parameters
        -----------
        n_samples : int
            Number of samples to be generated.
        min_value : float
            Minimum value for the random sample. Default is -1.
        max_value : float
            Maximum value for the random sample. Default is 1.
        multiprocess : bool
            If True, uses multiprocessing to speed up the computation. Default is False.
        eps : float
            Convergence threshold for the integration. Default is 0.0001.
        t_max : float
            Maximum time for the integration. Default is 50.
        
        verbose : bool
            If True, prints the progress of the integration. Default is False.
        """
        solutions = []



        # dummy function to be used for multiprocessing
        f = lambda _ : self.solve_random(min_value=min_value, max_value=max_value, eps=eps, t_max=t_max)

        results = []
        
        if multiprocess:
            raise NotImplementedError("Multiprocessing is not implemented yet.")
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = pool.map(f, range(n_samples)) 
        else:
            iterator = range(n_samples) 
            if verbose:
                iterator = tqdm(iterator, desc="Solving on uniform sample..", total=n_samples)
            for i in iterator:
                results.append(f(i))
        
        return results


