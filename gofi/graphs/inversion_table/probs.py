import torch


class PermModel:
    def __init__(self, model, n):
        """
        Parameters
        ------------
        model
            torch model. model() Should be a list of categorical distributions of lenghts n, n-1, ..., 1
        """
        self.model = model
        self.n = n

    def P(self, m, s):
        '''
        Returns the probability P(a_m = s) of placing m on the s-th empty file.
        '''
        probs =  self.model()[m-1] # we index with 0!
        return probs[s-1] # we index with zero!
    def P_sum(self, m , start, end):
        '''
        Returns SUM_{s=start --> end} P(a_m = s). That equals to the probability of placing m anywhere between (and including ) start and end.
        '''
        # get probabilities for placing m
        probs = self.model()[m-1] # index with zero
        # sum 
        return torch.sum(probs[start -1 : end]) #index by zero, end -1 is the last one

    
    def q(self, j, h, m):
        '''
        probability of placing h on the j-th empty file, if m-1 numbers were already placed.
        '''
    
        ### border cases ### 
        # j outside of scope
        if j <= 0 or j > self.n-m+1:
            return 0
        # j in scope but m is n --> just one option
        if m == self.n:
            return 1 
        
        ## place m on j-th place
        if m == h:
            return self.P(m, j)

        ### recursive ###
        ans = 0
        ans += self.q(j-1, h, m+1) * self.P_sum(m, 1, j-1)
        ans += self.q(j, h, m+1) * self.P_sum(m ,j+1, self.n-m+1)
        return ans  

    def p(self, i, k, j, h, m):
        """
        probability of placing k on the i-th empty file and j on the j-th empty file, if m-1 numbers were already placed.

        TODO"""

        # i < j
        if i >= j:
            # i should be SMALLER than j!
            i, j = j , i
            k, h = h , k

        ### border cases ###
        # i or j out of scope
        if i <= 0 or j <= 0 or i >  self.n-m+1 or  j > self.n-m+1:
            # outside of the region placed to many on one side
            return 0
        # m maximal
        if m == self.n:
            # just one tile to place 
            return 1
    
        ### place m ###
        if m == k:
            return self.P(k, i) * self.q(j-1, h, k+1)
        if m == h:
            return self.P(h, j) *  self.q(i, k, h+1)
        
        ### recursive ###
        ans = 0
        ans += self.p(i-1, k, j-1, h, m+1) * self.P_sum(m, 1, i-1)
        ans += self.p(i, k, j-1, h, m+1) * self.P_sum(m, i+1, j-1)
        ans += self.p(i, k, j, h, m+1) * self.P_sum(m , j+1, self.n-m+1)
        return ans

    def prob(self, i, k, j, h):
        """
        Returns probabiity of generating a permutation, where k is palced on the i-th file and h is placed on the j-th file
        """
        return self.p(i,k,j,h,1)
    