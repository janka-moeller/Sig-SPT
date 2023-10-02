# -*- coding: utf-8 -*-


import numpy as np
import copy
from joblib import Parallel, delayed
import itertools as it 
import iisignature


from Signature_portfolios_functions import *



### JL signature ###

def get_JL_signature_batch(path, level, rand_mat):
    """

    Parameters
    ----------
    path : np.array
        The (batch) of the process of which the JL-signature shall be 
        calculated.
    level : integer
        The level of the true signature to be calculated.
    rand_mat : np.array (2-dim)
        The JL-projection matrix.


    Returns
    -------
    JL_proj : np.array
        Path of the JL-signature.

    """
  
    full_sig=iisignature.sig(path,level,2)
    
    zeroth= [[0 for i in range(full_sig.shape[1]) ]]
    
    full_sig= np.concatenate((np.array(zeroth), full_sig), axis=0)
        
    JL_proj= rand_mat@(full_sig.T)
    
    return JL_proj


def get_JL_signature(order_sig, proj_dim, n_jobs, helper_path, rand_mat=None):
    """
    Calculates the path of the JL-signature.

    Parameters
    ----------
    order_sig : integer
        Order of the signature.
    proj_dim : integer
        Dimension of the JL-projection.
    n_jobs : integer
        Number of workers for parallelization.
    helper_path : np.array
        The process of which the JL-signature shall be calculated.
    rand_mat : np.array or None, optional
        The random matrix corresponding to the JL-projection. If None, a new 
        matrix is created. The default is None.

    Returns
    -------
    random_sig_dict : dictionary
        Dictionary of the path of the JL-signature.
        
    random_sig_keys : list of strings
        The keys of random_sig_dict.
        
    rand_mat : list of np.arrays 
        Slices of the projection matrix. Only returned if no matrix was 
        provided. 

    """
    
    indices= [i for i in range(0, len(helper_path[0,:]))]
    comb_list=[]
    was_none=False
    if rand_mat==None:
        was_none=True
        rand_mat=[]
        rand_mat.append(np.random.normal(0, 1/np.sqrt(proj_dim), 
                                          size=(proj_dim))) # the empty word
        
        
    for i in range(1,order_sig+1):
        tmp_comb= list(it.combinations(indices, i))
        comb_list+= tmp_comb
        
        sig_words=create_word_list(i,order_sig)
        
        if was_none:
            #select the redundant words
            to_zero=[]
            for w in sig_words:
                if len(w)!=0 and len(w)<i: 
                    to_zero.append(True)
                elif len(w)>=i and (i - len(set(w)))>=1: 
                    to_zero.append(True)
                elif len(w)!=0:
                    to_zero.append(False)
            to_zero=tuple(to_zero)    
            
        if was_none:
            for c in tmp_comb:
                size_full_sig=len(sig_words[1:])
                tmp_rand_mat= np.random.normal(0, 1/np.sqrt(proj_dim), 
                                                size=(proj_dim, size_full_sig))

                if np.any(to_zero):
                    tmp_rand_mat[:, to_zero] = np.zeros((proj_dim, 
                                                          np.sum(to_zero)))
                rand_mat.append(tmp_rand_mat)
                
    def step_for_batch(comb_list_batch, rand_mat_batch, return_rand_mat=was_none):
        JL_proj_of_batch=0
        

        for i, c in enumerate(comb_list_batch):
            path= helper_path[:,c]
            JL_proj= get_JL_signature_batch(path, order_sig, 
                                            rand_mat=rand_mat_batch[i], 
                                            proj_dim=proj_dim)
            
            JL_proj_of_batch+=JL_proj

        return JL_proj_of_batch
        
    job_step= int(len(comb_list)/n_jobs+1)
    
    JL_proj_list= Parallel(n_jobs=n_jobs)(delayed(step_for_batch)(comb_list[
        n*job_step: min((n+1)*job_step, len(comb_list))], rand_mat[
            1+n*job_step: 1+min((n+1)*job_step, len(comb_list))]) 
                for n in range(n_jobs))
    
    JL_proj= np.array([rand_mat[0] for _ in range(np.shape(helper_path)[0])]) #empty word
    
    for JL in JL_proj_list:
        if type(JL)!=int:
            JL_proj+= JL.T

    random_sig_keys=[str(i) for i in range(len(JL_proj[0,:]))]
    random_sig_dict= {}
    
    for i, I in enumerate(random_sig_keys):
        random_sig_dict[I]= JL_proj[:, i]
    
    if was_none:
        return random_sig_dict, random_sig_keys, rand_mat
    else:
        return random_sig_dict, random_sig_keys



##### randomized signature ######

def get_R_sig(
        proj_dim, helper_path, rand_mat_list=None, init_val=None, 
        activation=np.tanh):
    
    """
    Parameters
    ----------
    proj_dim : integer
        Dimension of the randomized signature.
    helper_path : np.array
        Process of which the randomized signature shall be calculated.
    rand_mat_list : list of np.arrays, optional
        List of matrices and vectors determining the randomized signature. If 
        None, new coefficients are initalized. The default is None.
    init_val : np.array, optional
        The inital value of the randomized signature. The default is None.
    activation : callable, optional
        Activation function for the randomized signature. The default 
        is np.tanh.

    Returns
    -------
    random_sig_dict : dictionary 
        Dictionary containing the path of the randomized signature.
        
    random_sig_keys : list of stings
        The keys of random_sig_dict.
        
    rand_mat_list : list of np.arrays
        List of matrices and vectors determining the randomized signature. 
        Only returned of none were provided. 

    """
    
    
    was_none=False
    Input= helper_path.T
    
    dim_input= np.shape(Input)[0]
    dim_t= np.shape(Input)[1]
    
    if rand_mat_list==None:
        b= np.random.normal(0, 1/np.sqrt(proj_dim), 
                            size=(dim_input, proj_dim))
        A= np.random.normal(0, 1/np.sqrt(proj_dim), 
                            size=(dim_input,proj_dim, proj_dim))
        rand_mat_list=[b,A]
        was_none=True
    else:
        b= copy.deepcopy(rand_mat_list[0])
        A= copy.deepcopy(rand_mat_list[1])
    
    if np.any(init_val)==None:
        init_val= np.zeros(proj_dim)
    
    RS=np.zeros((proj_dim, dim_t))
    RS[:,0]= init_val
    for t in range(dim_t-1):
        predictor= np.tensordot((Input[:,t+1] - Input[:, t]), 
                    activation(b+ np.tensordot(A,RS[:,t], axes=1)), axes=1)

        RS[:, t+1]= RS[:,t] + activation(b[0,:]+ A[0,:,:]@RS[:,t])*(
            Input[0,t+1] - Input[0, t])+ np.tensordot(
                (Input[1:,t+1] - Input[1:, t]), 0.5*(
                    activation(b+ np.tensordot(A,RS[:,t], axes=1))+activation(
                        b+ np.tensordot(A,predictor, axes=1)))[1:], axes=1)
        
    random_sig_keys=[str(i) for i in range(proj_dim)]
    random_sig_dict= {}
    
    for i, I in enumerate(random_sig_keys):
        random_sig_dict[I]= RS[i,:]
    
    if was_none:
        return random_sig_dict, random_sig_keys, rand_mat_list
    else:
        return random_sig_dict, random_sig_keys
    
    
    