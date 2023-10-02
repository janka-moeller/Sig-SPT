# -*- coding: utf-8 -*-

import numpy as np
import copy
from scipy.linalg import sqrtm, inv
import iisignature

import Sig_optim_functions as func



            
class GenerateSigMarket:
    """
    A class to generate a Sig-market. 
   

    Attributes
    ----------
    n_stocks : int
        The number of stocks in the market. 
    dim_BM : int
        Must be equal to the number of stocks.
    T : int
        Number of time steps to simulate.
    Cov : np.array or None, optional
        The market's covariance matrix. If None, a random matrix is 
        initailized. Default is None.
    a_coeff_list : list or None, optional
        A list with coefficients of the linear function of the signature. If
        None, a new list is created. Default is None.
    order : int, optional
        The order of the signature. Default is 3. 
    S0 : np.array or 1, optional 
        The inital values of the capitalization process. Default is 1. 

    Methods
    -------
    generate_prices()
        Generates the price/capitalization-process of the specified market.
    """
    
    
    def __init__(
            self,n_stocks, dim_BM, T, Cov=None, a_coeff_list=None, order=3, 
            S0=1):
        
        self.n = n_stocks
        self.dim= dim_BM
        self.T= T
        self.dt= 1/self.T
        self.dW= np.zeros((self.dim, self.T))
        self.BM= np.zeros((self.dim, self.T))
        
        self.MktCov= Cov
        self.L=0
        self.alpha=None
        self.price= np.zeros((self.n, self.T))
        self.order=order
        self.a_coeff_list=a_coeff_list
        self.sig_term=[]
        self.sig_term_list=[]
        
        if S0==1:
            self.price[:,0]= np.ones((self.n))
        else:
            self.price[:,0]= S0
    
        
    def _BM(self):
        for t in range(self.T): 
            self.dW[:,t]= np.random.normal(0,1,self.dim) 
        self.dW= self.dW/np.sqrt(self.T) 
        self.BM= np.cumsum(self.dW, axis=1) 
        
                
    def _MktCov(self): 
        self.L = np.tril(np.random.normal(5.6,2.5, [self.n, self.dim]))
        self.MktCov= np.matmul(self.L.transpose(), self.L)
        
    def _mu(self):
        self.mu= np.random.uniform(0.001,0.01,self.n)
        
    def _fill_a_coeff_list(self, nparam):
        self.a_coeff_list=[]
        words= func.create_word_list(self.n+1,self.order)
        for i in range(self.n):
            tmp_a_coeff=[]
            for w in words:
                if len(w)==0:
                    tmp_a_coeff.append(np.random.uniform(-0.001,0.001,nparam))
                elif len(w)==1:
                    tmp_a_coeff.append(np.random.uniform(-0.001,0.001,nparam))
                elif np.all(np.array(w[1:])==1):
                    tmp_a_coeff.append(np.random.uniform(-0.001,0.001,nparam))
                else:
                    tmp_a_coeff.append(0)
                    
    def _get_new_sig_term(self,t):
        weights= func.get_market_weights(self.price[:,:t+1], rank_based=False)
        process= func.get_integrator_and_or_helper_path(
            mkt_weights=weights, mkt_weights_quad_var=None, 
            quad_var_labels=None, timestep=self.dt, result="helper")
        signature_full= iisignature.sig(process,self.order)
        signature_full= np.concatenate((np.array([1]), signature_full))
    
        if self.a_coeff_list==None:
            nparam= len(signature_full)
            self._fill_a_coeff_list(nparam)
        self.sig_term=[]
        for i in range(self.n):
            self.sig_term.append(self.a_coeff_list[i]@signature_full)
        
        
    def generate_prices(self):
        if np.any(self.MktCov==None):
            self._MktCov()
        
        else: 
            self.L= sqrtm(self.MktCov)
            
        self.L= sqrtm(self.MktCov)
            
        self._BM()
        diffusion= np.matmul(self.L,self.dW)
        
        sigma_sq= np.diag((self.L@self.L.T))
        
        self._get_new_sig_term(0)
        
        for t in range(0,self.T-1):
            self.price[:, t+1] = self.price[:, t]*np.exp(
                (np.array(self.sig_term)- 0.5*sigma_sq)*self.dt+ diffusion[:,t])
            self.sig_term_list.append(copy.deepcopy(self.sig_term))
            self._get_new_sig_term(t+1)





class GenerateVolStabMarket:
    """
    A class to generate a Volatility-Stabilized market. 
   

    Attributes
    ----------
    n_stocks : int
        The number of stocks in the market. 
    dim_BM : int
        Must be equal to the number of stocks.
    T : int
        Number of time steps to simulate.
    beta : positive float or None, 
        Model parameter of the market.
    order : int, optional
        Redundant parameter. 
    S0 : np.array or 1, optional 
        The inital values of the capitalization process. Default is 1. 

    Methods
    -------
    generate_prices()
        Generates the price/capitalization-process of the specified market.
    """
    def __init__(self,n_stocks, dim_BM, T, beta=None, order=3, S0=1):
        self.n = n_stocks
        self.dim= dim_BM
        self.T= T
        self.dt= 1/self.T
        self.dW= np.zeros((self.dim, self.T))
        self.BM= np.zeros((self.dim, self.T))
        
        self.beta= beta
        self.price= np.zeros((self.n, self.T))
        self.order=order
        self.t_term_list=[]
        self.W_term_list=[]
        
        if S0==1:
            self.price[:,0]= np.ones(self.n)
        else:
            self.price[:,0]= S0
    
        
    def _BM(self):
        for t in range(self.T): 
            self.dW[:,t]= np.random.normal(0,1,self.dim) 
        self.dW= self.dW/np.sqrt(self.T) 
        self.BM= np.cumsum(self.dW, axis=1) 
        
                        
    def _beta(self):
        self.beta= 2.1
            
    def _get_new_term(self,t):
        if self.beta==None:
            self._beta()
        total_cap= np.sum(self.price[:, t])
        self.t_term= []
        self.W_term=[]
        for i in range(self.n):
            self.t_term.append(total_cap/self.price[i,t])
            self.W_term.append(np.sqrt(total_cap/self.price[i,t]))
        self.W_term= np.diag(self.W_term)
        self.t_term= np.array(self.t_term)
        
    def generate_prices(self):
 
        self._BM()
        
        self._get_new_term(0)
              
        for t in range(0,self.T-1):
            self.price[:, t+1] = self.price[:, t]*np.exp(
                (self.beta)/2*self.t_term*self.dt + self.W_term@self.dW[:,t])
            self.t_term_list.append(copy.deepcopy((self.beta+1)/2*self.t_term))
            self.W_term_list.append(copy.deepcopy(self.W_term))
            self._get_new_term(t+1)
        
        
class GenerateMarket:
    
    """
    A class to generate a (correlated) Black-Scholes market. 
   

    Attributes
    ----------
    n_stocks : int
        The number of stocks in the market. 
    dim_BM : int
        Must be equal to the number of stocks.
    T : int
        Number of time steps to simulate.
    Cov : np.array 
        The market's covariance matrix. 
    mu : np.array
        The drift-component of the capitalization process. 
    S0 : np.array or 1, optional 
        The inital values of the capitalization process. Default is 1. 

    Methods
    -------
    generate_prices()
        Generated the price/capitalization-process of the specified market.
    """
    
    
    
    def __init__(self,n_stocks, dim_BM, T, Cov, mu, S0=1):
        
        self.n = n_stocks
        self.dim= dim_BM
        self.T= T
        self.dt= 1/self.T
        self.dW= np.zeros((self.dim, self.T))
        self.BM= np.zeros((self.dim, self.T))
        
        self.MktCov= Cov
        self.L=0
        self.alpha=None
        self.drift= np.random.rand()
        self.mu=mu
        self.price= np.zeros((self.n, self.T))
        
        if np.all(S0)==1:
            self.price[:,0]= np.ones((self.n))
        else:
            self.price[:,0]= S0
    
        
    def _BM(self):
        for t in range(self.T): 
            self.dW[:,t]= np.random.normal(0,1,self.dim) 
        self.dW= self.dW/np.sqrt(self.T) 
        self.BM= np.cumsum(self.dW, axis=1) 
                      
        
    def generate_prices(self):

        self.L= sqrtm(self.MktCov)

        sigma= np.sqrt(np.diag(self.MktCov)) 
        if np.any(self.mu==None):
            self._mu()
        drift= (self.mu - 0.5 * sigma**2)*(self.dt) 
        self.alpha= self.mu - 0.5 * sigma**2

        self._BM()
        diffusion= np.matmul(self.L,self.dW)
        
        for t in range(1,self.T):
            self.price[:, t] = self.price[:, t-1]*np.exp(
                drift + diffusion[:,t])
        
        
            
        

    
class GrowthOptimal:
    """
    A class to obtain the theoretical growth-optimal porfolio of a market. 
   

    Attributes
    ----------
    d : int
        The number of stocks in the market. 
    m : int
        Number of Brownian Motions must be at least as many as stocks in the
        market. 
    T : int
        Number of time steps to simulate.
    a_list : list of np.arrays (1-dim) or None, optional 
        The process of drift-vectors of the market, if the vector is constant, 
        a list of length 1 can be passed. Default is None. 
    Sigma_list : list of np.arrays (2-dim) or None, optional 
        The process of variance-matrices of the market, if the vector is 
        constant, a list of length 1 can be passed. Default is None. 
    mkt_weights : np.array or None, optional
        The process of market weights. Default is None.

    Methods
    -------
    get_go_weights()
        Generates the growth-optimal weights of the specified market.
    """
    def __init__(self,d,m,T,a_list=None,Sigma_list=None, mkt_weights=None):
        self.d= d
        self.m=m 
        self.T= T
        self.a_list= a_list
        self.Sigma_list= Sigma_list
        self.mkt_weights= mkt_weights
        self.all_constant= False
        
        if len(Sigma_list)==1 and len(a_list)!=1:
            self.Sigma_list= self.Sigma_list*T
        elif len(Sigma_list)!=1 and len(a_list)==1: 
            self.a_list=self.a_list*T
        elif len(Sigma_list)==1 and len(a_list)==1: 
            self.all_constant=True
            
    def _growth_optimal_weight(self, Sigma, a):
        S_inv= inv(Sigma@Sigma.T)
        kappa = (np.sum(S_inv@a) - 1)/(np.sum(S_inv))
        kappa_array= np.array([kappa]*len(a))
        pi= S_inv@(a-kappa_array)
        return pi
    
    def get_GO_weights(self):
        if self.all_constant: 
            pi= self._growth_optimal_weight(self.Sigma_list[0], self.a_list[0])
            self.go_weights= [pi]*self.T
        else:
            self.go_weights=[]
            for t in range(self.T-1):
                self.go_weights.append(self._growth_optimal_weight(
                    self.Sigma_list[t], self.a_list[t])) 
               
        self.go_weights=np.array(self.go_weights)
        
        
        
        
