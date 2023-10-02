# -*- coding: utf-8 -*-


import numpy as np
import os
import copy
from scipy.optimize import fsolve, minimize
import matplotlib.pyplot as plt
from datetime import date, datetime
import pickle
import gurobipy
from gurobipy import GRB
from joblib import Parallel, delayed

import itertools as it 

import iisignature

import Signature_portfolios_classes as classes
from Randomize_signatures import *


### Basic Functions #####

def create_word_list(dim,order):
    """
    Created the list of words associated to the elements of the signature. 
    Each word is given as a list of integers. 

    Parameters
    ----------
    dim : int
        the dimension of the process of which the signature should be calculated.
    order : int
        the order of the signature.

    Returns
    -------
    res : list of list of int
        list of the words.

    """
    res=[[]]
    for i in range(order):
        if i==0: 
            start_idx=0
        else:
            start_idx=dim**(i-1)+old_start_idx
        for r in res[start_idx:]:
            for d in range(1,dim+1):
                res.append(r+[d])
        old_start_idx=start_idx
    return res

def get_market_weights(capitalizations, rank_based=False):
    """
    Calculates the market weights of a process "capitalizations".

    Parameters
    ----------
    capitalizations : np.array of shape (D, T)
        Capitalization process of D stocks over T timesteps.
    rank_based : boolean, optional
        Whether the ranked-weights should be computed. The default is False.

    Returns
    -------
    np.array of shape (D, T)
        Process of market weights (columns sum to 1). 

    """
    
    if rank_based: #here the weights are ordered in a rank-based fashion
        weights= capitalizations/np.sum(capitalizations, axis=0) 
        ranked_weights= -np.sort(-weights, axis=0) #order weights column-wise in descending order (first row is always largest weight)
        return ranked_weights
        
    else:
        weights= capitalizations/np.sum(capitalizations, axis=0)
        return weights


def quad_var_dict_label(i, j, add_time_comp=True):
    """
    Creates the labels of the quadratic variation.

    Parameters
    ----------
    i : integer
        Index of the component of the first process in the quadratic variation.
    j : integer
        Index of the component of the second process in the quadratic variation.
    add_time_comp : boolean, optional
        Whether process has a time-augmentation as its first component. 
        The default is True.

    Returns
    -------
    string
        Label assocuated to quadratic variation. "<j,i>" if j>=i else "<i,j>"

    """
    #if there is a time-augmentation, everything needs to be shifted by one, 
    #since time is the first component of the path
    if add_time_comp: 
        i= i+1
        j=j+1
    #take into account that we only calcualted quadratic variation for j>=i 
    #and make use of symmetry otherwise
    if j >= i:
        return '<'+ str(j)+','+str(i)+'>'
    else:
        return '<'+ str(i)+','+str(j)+'>'
    
    
def get_quadratic_variation(
        process, add_time_comp=True, MOD=False):
    """
    Calculates the (discretized) quadratic variation of a process.

    Parameters
    ----------
    process : np.array of shape (D, T)
        Process of which the quadratic variation should be computed.
    add_time_comp : boolean, optional
        Whether the process has a time-augmentation in the first component. 
        The default is True.
    MOD : boolean, optional
        If true, the type II weights are considered. The default is False.
    
    Returns
    -------
    
    quad_var_dict : dictionary
        Contains the components of the quadratic variation. 
    quad_var_label_list : list of strings
        List of the keys of quad_var_dict.
    quad_var_tuple_list : list of tuples
        Only returned if MOD==True. Containes tuples associated to the 
        components of the quadratic variation. 
    quad_var_index : dictionary
        Assigns to each component of the quadratic variation an integer count. 

    """

    quad_var_dict= {}
    quad_var_label_list= []
    quad_var_tuple_list=[]
    quad_var_index= {}
    dim= np.shape(process)[0]
    count=0
    for i in range(0,dim): 
        for j in range(i,dim):
            temp_quad_var= np.cumsum((process[i,1:]-process[i,:-1])*
                                     (process[j,1:]-process[j,:-1])) 
            
            #first element of quadratic variation is 0 
            temp_quad_var= np.concatenate([[0],temp_quad_var]) 
            
            #obtain the correct label for the quadratic variation
            label= quad_var_dict_label(i,j, add_time_comp) 
            quad_var_dict[label]= temp_quad_var 
            
            #append the label to a list of labels 
            #(this is fast map for index-> label)
            quad_var_label_list.append(label) 
            
            #append current iteration count to dictionary of label
            #( this is fast map for label->index)
            quad_var_index[label]= count 
            
            if MOD:
                quad_var_tuple_list.append((i,j))
                
            count +=1
    
    if MOD:
        return (quad_var_dict, quad_var_label_list, quad_var_tuple_list, 
                quad_var_index)
    
    return quad_var_dict, quad_var_label_list, quad_var_index
    

def qvar_idx_to_letter(qvar_idx, n_stocks, add_time=True, incl_MCAP=False):
    """
    Maps the index of the quadratic variation to the corresponding letter of 
    the signature.

    Parameters
    ----------
    qvar_idx : integer
        The index of the quadratic variation component in the path.
    n_stocks : integer
        Number of stocks (corresponds to the number of components in the 
        underlying process).
    add_time : boolean, optional
        Whether a time-component is added to the augmented path (before the 
        quadratic variation). The default is True.
   
    Returns
    -------
    integer
        The letter in the signature corresponding to the provided index.

    """
    
    if add_time: 
        # the letter is +1 to account for time-component and +1 
        # because letters start at 1
        return qvar_idx+n_stocks+2 
    else:
        return qvar_idx+n_stocks+1 #the letter is +1 bc letters start at 1 
    

def dicrete_Stratonovic(X, W, full_hist=False, t_start=0):
    """
    Calculates the Stratonovich integral for dicrete timeseries. 
    This is a mid-point-approximation.
    
    Parameters
    ----------
    X : list or np.array
        The integrand-process.
    W : list or np.array
        The integrator-process.
    full_hist : boolean, optional
        Whether all intermiate integrals should be returned as well or just 
        the final one. The default is False.
    t_start : integer, optional
        From which position to start integrating. The default is 0.

    Returns
    -------
    int or np.array
        Either the value for the integral or np.array of all intermediate 
        integrals.

    """
    
    if full_hist:
        return np.cumsum((X[1+t_start:]+X[t_start:-1])/2*(W[1+t_start:]-W[t_start:-1])) #return all the "intermediate" integrals aswell
    else:
        return np.sum((X[1+t_start:]+X[t_start:-1])/2*(W[1+t_start:]-W[t_start:-1])) #return the full integral only 
    
    
def Ito_integral(X, W, full_hist=False, t_start=0):
    """
    Calculates the Ito integral for discrete timeseries. This is a 
    left-point-approximation. 

    Parameters
    ----------
    X : list or np.array
        The integrand-process.
    W : list or np.array
        The integrator-process.
    full_hist : boolean, optional
        Whether all intermiate integrals should be returned as well or just 
        the final one. The default is False.
    t_start : integer, optional
        From which position to start integrating. The default is 0.

    Returns
    -------
    int or np.array
        Either the value for the integral or np.array of all intermediate 
        integrals.

    """
    
    if full_hist: 
        return np.cumsum((X[t_start:-1])*(W[1+t_start:]-W[t_start:-1])) #return all the "intermediate" integrals aswell
    else:
        return np.sum((X[t_start:-1])*(W[1+t_start:]-W[t_start:-1]))#return the full integral only 



def get_integrator_and_or_helper_path(
        mkt_weights=None, mkt_weights_quad_var=None, quad_var_labels=None, 
         timestep=1, result="both", total_cap=None, incl_MCAP=False):
    """
    Parameters
    ----------
    mkt_weights : np.array of shape (D,T), optional
        Process of market weights. The default is None.
    mkt_weights_quad_var : dictionary, optional
        The quadratic variation. Only needed if result=="both" or "integrator". 
        The default is None.
    quad_var_labels : list, optional
        The labels of the quadratic variation. Only needed if result=="both" 
        or "integrator".  The default is None.
    timestep : float, optional
        The increments of time. The default is 1.
    result : string, optional
        Whether to just return helper, integrator or both. The default is "both".
    total_cap : np.array of shape (T), optional
        The total market capitalization. The default is None.
    incl_MCAP : boolean, optional
        Whether to include the total capitalization. The default is False.

    Raises
    ------
    NotImplementedError
        Raises Error if result is not in ["both", "helper", "integrator"].

    Returns
    -------
    np.array(s)
        Either the helper or integrator process or both.

    """
    
    if result=="both":
        get_helper=True
        get_integrator=True
    elif result=="helper":
        get_helper=True
        get_integrator=False
    elif result=="integrator":
        get_helper=False
        get_integrator=True
    else:
        raise NotImplementedError("result must either be 'both', 'helper' or 'integrator'")
    
    n_stocks= len(mkt_weights[:,0])
    
    if get_helper:
        t= len(mkt_weights[1,:])
        dim_helper_path= n_stocks+1
        if incl_MCAP:
            dim_helper_path+=1
        helper_path= np.zeros((t, dim_helper_path))
    if get_integrator:
        var_dim= len(mkt_weights_quad_var)
        dim_integ_path= var_dim+n_stocks
        integrator_path= np.zeros((t, dim_integ_path))
       
    for i in range(n_stocks):
        if get_helper:
            if incl_MCAP:
                helper_path[:,i+2]= mkt_weights[i,:]
            else:
                helper_path[:,i+1]= mkt_weights[i,:]
        if get_integrator:
            integrator_path[:,i]= mkt_weights[i,:]
            
    if get_integrator:
        for i in range(n_stocks,dim_integ_path):
            integrator_path[:,i]=  mkt_weights_quad_var[quad_var_labels[i-n_stocks]]
    if get_helper:
        helper_path[:,0]= [timestep*i for i in range(0, t)] #in helper_path the zeroth component is time
        if incl_MCAP:
            helper_path[:,1]= total_cap
    
    
    if result=="both":
        return integrator_path, helper_path
    elif result=="helper":
        return helper_path
    elif result=="integrator":
        return integrator_path



def get_signature_full(process, order):
    """
    Returns the path of the signature of a process and the corresponding 
    signature keys.

    Parameters
    ----------
    process : np.array shape (T, D)
        Process of which the signature shall be calculated. T is the number of 
        time-steps and D the dimension of the process.
    order : integer
        Order of the signature to be calculated.

    Returns
    -------
    signature_dict_full : dictionary
        The path of the siganture. The keys specify the elements of the 
        signature.
    signature_keys_full : list of lists
        Each element denotes a word of the signature by a list of ints 
        i.e. [1,4,2].
    signature_keys_full_STR : list of strings
        Each element is the string of a word by converting the word to a 
        string i.e. '[1,4,2]'.

    """

    dim_process= len(process[0,:])
    signature_keys_full= create_word_list(dim_process, order)
    signature_full= iisignature.sig(process,order,2)
    
    zeroth= [[0 for i in range(signature_full.shape[1]) ]]
    empty_word= [[1 for i in range(signature_full.shape[0]+1) ]]

    signature_full= np.concatenate((np.array(zeroth), signature_full), axis=0)
    signature_full= np.concatenate((np.array(empty_word).T, signature_full), 
                                   axis=1)
    
    signature_dict_full= {}
    signature_keys_full_STR=[]
    for i in range(len(signature_full[0,:])):
        signature_dict_full[str(signature_keys_full[i])]= signature_full[:,i]
        signature_keys_full_STR.append(str(signature_keys_full[i]))
    
    return signature_dict_full, signature_keys_full, signature_keys_full_STR

    
    
    
def version_my_dir(mkt_dir):
    """
    Check whether a directory already exists and if so, appends a version 
    number to it. 

    Parameters
    ----------
    mkt_dir : string
        Directory to be checked/versioned.

    Returns
    -------
    mkt_dir : string
        Versioned directory.

    """
    if os.path.isdir(mkt_dir):
        check=True
        count=1
        while(check):
            tmp_mkt_dir=mkt_dir+"_"+str(count)
            if os.path.isdir(tmp_mkt_dir): 
                count+=1
            else:
                check=False
                mkt_dir=tmp_mkt_dir
                
    return mkt_dir



def initialize_Q_mem_optim(n_stocks, signature_keys_tolearn, integrator, 
                           signature_dict, quad_var_index, t_start=0, 
                           port_type=1):
    
    """
    Initialize the Q matrix for the log-value-optimization. 

    Parameters
    ----------
    n_stocks : integer
        Number of stocks.
    signature_keys_tolearn : list
        Keys of the feature maps.
    integrator : np.array shape (D, T)
        Process of the integrator.
    signature_dict : dictionary
        Dictionary of the feature maps.
    quad_var_index : dictionary
        Dictionary of index corresponding to a label of the quadratic 
        variation.
    t_start: integer, optional
        Index where to start integration. Default is 0. 
    port_type: 1 or 2, optional
        Type of path-functional portfolios. Default is 1. 

    Returns
    -------
    Q : np.array
        Quadratic matrix for the log-value optimization problem.

    """

    n_param= len(signature_keys_tolearn)*n_stocks
    Q= np.zeros((n_param, n_param))
    row=0
    col=0
    for i in range(n_stocks):
        for l, L in enumerate(signature_keys_tolearn):
            for j in range(n_stocks):
                for k, K in enumerate(signature_keys_tolearn):
    
                    if port_type==1:
                        integrand= signature_dict[L]*signature_dict[K]
                    elif port_type==2:
                        integrand= signature_dict[L]/integrator[:,i]*signature_dict[K]/integrator[:,j]
                        
                    qvar_letter= qvar_idx_to_letter(
                        quad_var_index[quad_var_dict_label(j,i, True)], 
                        n_stocks, add_time=True)
                    
                    integral= Ito_integral(
                        integrand,integrator[:,qvar_letter-2], 
                        t_start=t_start)
                    
                    Q[row, col]=  integral

                    col+=1
            row+=1
            col=0
    return Q


def initialize_c_Ito_mem_optim(
        n_stocks, signature_keys_tolearn, signature_dict, integrator, 
        quad_var_index, t_start=0, port_type=1):
    
    """
    Initialize the c vector for the log-value-optimization. 

    Parameters
    ----------
    n_stocks : integer
        Number of stocks.
    signature_keys_tolearn : list
        Keys of the feature maps.
    integrator : np.array shape (D, T)
        Process of the integrator.
    signature_dict : dictionary
        Dictionary of the feature maps.
    quad_var_index : dictionary
        Dictionary of index corresponding to a label of the quadratic 
        variation.
    t_start: integer, optional
        Index where to start integration. Default is 0. 
    port_type: 1 or 2, optional
        Type of path-functional portfolios. Default is 1. 

    Returns
    -------
    c : np.array
        Vector for the log-value optimization problem.

    """
    
    n_param= len(signature_keys_tolearn)*n_stocks
    c= np.zeros((n_param))
    row=0
    
    if port_type==1:
        for i in range(n_stocks): 
            for l, L in enumerate(signature_keys_tolearn):
                integral= Ito_integral(signature_dict[L],
                                       integrator[:,i], t_start=t_start)
                c[row]= (-1)*integral
                row+=1
        return c   
    
    elif port_type==2:
        for i in range(n_stocks): 
            for l, L in enumerate(signature_keys_tolearn):
                integral= Ito_integral(signature_dict[L]/integrator[:,i],
                                       integrator[:,i], t_start=t_start)
                c[row]= (-1)*integral
                row+=1
        return c 


def initialize_Q_c_MV(n_stocks, signature_keys_tolearn, signature_dict, weights, t_start):
    
    """
    Initialize the Q matrix and c vector for the mean-variance-optimization 
    (for portfolios of type 1). 

    Parameters
    ----------
    n_stocks : integer
        Number of stocks.
    signature_keys_tolearn : list
        Keys of the feature maps.
    signature_dict : dictionary
        Dictionary of the feature maps.
    weights : np.array
        Market weights.
    t_start: integer
        Index where to start investing. 
 
    Returns
    -------
    Q, c : np.array
        Matrix and vector for the mean-variance optimization problem.

    """
    
    n_param= len(signature_keys_tolearn)*n_stocks
    time= len(weights[0,:])-t_start-1
    
    row=0
    Y_process=np.zeros((n_param, time))
    for i in range(n_stocks):
        for l, L in enumerate(signature_keys_tolearn):
            L=str(L)
            
            Y_process[row, :]= signature_dict[L][(t_start):-1]*(
                weights[i,(t_start+1):]-weights[i,(t_start):-1] )
            row+=1
    Cov= np.cov(Y_process)
    mean= np.mean(Y_process, axis=1)
    return Cov, -1*mean


def initialize_TC_mat(n_stocks, signature_keys_tolearn, signature_dict, weights, t_start, n_jobs=1):
    
    """
    Initialize the matrix for the regularization for transaction costs (for 
    portfolios of type 1).

    Parameters
    ----------
    n_stocks : integer
        Number of stocks.
    signature_keys_tolearn : list
        Keys of the feature maps.
    signature_dict : dictionary
        Dictionary of the feature maps.
    weights : np.array
        Market weights.
    t_start: integer
        Index where to start investing. 
    n_jobs : integer
        Number of workers for paralellization.
 
    Returns
    -------
    T : np.array
        Matrix for the regularization for transaction costs.

    """
    
    n_param= len(signature_keys_tolearn)*n_stocks
    nr_keys= len(signature_keys_tolearn)
    T= np.zeros((n_param, n_param))

    
    n_row_list=[0]+[n_param-i for i in range(n_param)]
    n_row_list=list(np.cumsum(n_row_list))
    
    n_entries= int(n_param*(n_param+1)/2)
    n_entries_list=list(range(n_entries))
    
    def T_step(batch):
        T= np.zeros((n_param, n_param))
        for n in batch:
            row=0
            for r in range(n_param):
                if n< n_row_list[r+1]:
                    row=copy.copy(r)
                    break

            new_n= n-n_row_list[row]
            col = row+new_n
            
            i= int(col/nr_keys)
            l= col%nr_keys
                        
            j= int(row/nr_keys)
            k= row%nr_keys
    
            L= str(signature_keys_tolearn[l])
            K= str(signature_keys_tolearn[k])
            
            if i==j:
                T1= initialize_T1(K, L, signature_dict, t_start)
            else:
                T1=0
            T2= initialize_T2(i,j,K, L, signature_dict, weights, t_start)
            T3= initialize_T3(i,j,K, L, signature_dict, weights, t_start)
            
            T[row, col]= T1+T2+n_stocks*T3
            T[col, row]= T1+T2+n_stocks*T3
        return T
    print("time before parallel-loop:", datetime.now(), flush=True)
    job_step= int(n_entries/n_jobs+1)
    res= Parallel(n_jobs=-1)(delayed(T_step)(n_entries_list[
        n*job_step: min((n+1)*job_step, n_entries)]) for n in range(n_jobs))
    print("time after parallel-loop:", datetime.now(), flush=True)
    
    for r in res:
        T+=r
            
    print("time after constructing T:", datetime.now(), flush=True)

    return T


def initialize_T1(K, L, signature_dict, t_start):
    """
    Helper function to calculate the matrix for the regualrization for 
    transaction cost (for portfolios of type I)

    Parameters
    ----------
    K : string
        Feature key.
    L : string
        Feature key.
    signature_dict : dictionary 
        Dictionary of feature maps.
    t_start : integer
        Index where to start investing.

    Returns
    -------
    integer

    """
    elem_K= signature_dict[K][(t_start+1):]- signature_dict[K][t_start:-1]
    elem_L= signature_dict[L][(t_start+1):]- signature_dict[L][t_start:-1]
    
    return np.sum(elem_K*elem_L)

def initialize_T2(i,j, K, L, signature_dict, weights, t_start):
    """
    Helper function to calculate the matrix for the regualrization for 
    transaction cost (for portfolios of type I)

    Parameters
    ----------
    K : string
        Feature key.
    L : string
        Feature key.
    signature_dict : dictionary 
        Dictionary of feature maps.
    t_start : integer
        Index where to start investing.

    Returns
    -------
    integer

    """
    elem_L_i= (weights[i, (t_start+1):]*signature_dict[L][(t_start+1):]- 
               weights[i, (t_start):-1]*signature_dict[L][t_start:-1])
    elem_K= signature_dict[K][(t_start+1):]- signature_dict[K][t_start:-1]
    
    elem_L= signature_dict[L][(t_start+1):]- signature_dict[L][t_start:-1]
    elem_K_j= (weights[j, (t_start+1):]*signature_dict[K][(t_start+1):]- 
               weights[j, (t_start):-1]*signature_dict[K][t_start:-1])
    
    return -np.sum(elem_L_i*elem_K)- np.sum(elem_L*elem_K_j)

def initialize_T3(i,j, K, L, signature_dict, weights, t_start):
    """
    Helper function to calculate the matrix for the regualrization for 
    transaction cost (for portfolios of type I)

    Parameters
    ----------
    K : string
        Feature key.
    L : string
        Feature key.
    signature_dict : dictionary 
        Dictionary of feature maps.
    t_start : integer
        Index where to start investing.

    Returns
    -------
    integer

    """
    elem_L_i= (weights[i, (t_start+1):]*signature_dict[L][(t_start+1):]- 
               weights[i, (t_start):-1]*signature_dict[L][t_start:-1])
    elem_K_j= (weights[j, (t_start+1):]*signature_dict[K][(t_start+1):]- 
               weights[j, (t_start):-1]*signature_dict[K][t_start:-1])
    
    return  np.sum(elem_L_i*elem_K_j)

def calc_sig_portfolio_weights(
        l, mkt_weights, order_sig, add_time=True, timestep=None, 
        total_cap=None, incl_MCAP=False, normalize=False, 
        mkt_weights_norm=None, randomsig=False, n_jobs=64, proj_dim=None, 
        rand_mat_list=[], reuse_sig_mu_hat=None, port_type=1):
    
    """
    Calculate the weigths of the signature portfolios. 

    Parameters
    ----------
    l : np.array
        Linear parameters of signature portfolios.
    order_sig : integer
        Order of the signature.
    add_time : boolean, optional 
        Whether process has a time-augmentation. Default is true.
    timestep : float or None, optional 
        Size of the timesteps. Only needed if add_time==True. Default is None.
    total_cap : np.array or None, optional
        Process of the total capitalization. Only needed if incl_MCAP==True. 
        Default is None.
    incl_MCAP : boolean, optional 
        Whether to include the total capitalization. Default is False.
    normalize: boolean, optional
        Whether the normalized market weights are used. Default is False. 
    mkt_weights_norm: np.array or None,
        The process of normalized market weights. Only needed if 
        normalize==True. Default is None. 
    randomsig: False, "JL" or "RANDOMIZED", optional
        Whether to use a randomization of the signature and if yes which one. 
        Default is False. 
    n_jobs: integer, optional
        Number of paralell workers to be used. Default is 64. 
    proj_dim: integer or None, optional 
        If a randomization of the signature is used, the dimension of the 
        projection. Default is None. 
    rand_mat_list: list, optional 
        If a randomization of the signature is used, the matrices/vectors 
        associated with the randomization. Default is [].
    reuse_sig_mu_hat: np.array or None, optional
        np.array of the signature has already been calculated and can be 
        reused. This is usefull when optimizing the regularization parameter 
        for transaction-costs. Default is None.
    port_type: 1 or 2, optional 
        Which portfolio type to be used. Default is 1. 

    Returns
    -------
    sig: list of np.arrays
        The siganture weights, each elemnt of the list the process of weights 
        for the corresponding stock.
    F: list
        List of processes of values of the portfolio controlling function. 

    """

    if not add_time:
        raise NotImplementedError("add_time_comp=False is currently not implemented") 
    
    if reuse_sig_mu_hat==None:
        if normalize:
            #augment mkt-weights process by time
            mu_hat= get_integrator_and_or_helper_path(
                mkt_weights=mkt_weights_norm,timestep=timestep, 
                result="helper", total_cap=total_cap, incl_MCAP=incl_MCAP)
        else:
            #augment mkt-weights process by time
            mu_hat= get_integrator_and_or_helper_path(
                mkt_weights=mkt_weights,timestep=timestep, result="helper", 
                total_cap=total_cap, incl_MCAP=incl_MCAP)
    
        if randomsig==False:
            sig_mu_hat, sig_mu_hat_keys, sig_mu_hat_keys_str= get_signature_full(
                mu_hat, order_sig)#calculate signature of augmented process
        elif randomsig=="JL":
            sig_mu_hat, sig_mu_hat_keys_str= get_JL_signature(
                order_sig, proj_dim, n_jobs, mu_hat, rand_mat=rand_mat_list)
        elif randomsig=="RANDOMIZED":
            sig_mu_hat, sig_mu_hat_keys_str= get_R_sig(
                proj_dim,  mu_hat, rand_mat_list=rand_mat_list)


    else:
        print("reusing here!")
        sig_mu_hat, sig_mu_hat_keys_str= reuse_sig_mu_hat
    
    
    len_sig= len(sig_mu_hat.keys())#nr of words in signature
    
    F=[]
    curr_ls=[]
    
    for i in range(len(l)):
        curr_ls.append(l[i])#store l as long as we are in range of one stock
        # before we reach the next stock, calculate F
        if (i+1)%(len_sig)==0 and i!=0: 
            curr_F= sum([curr_ls[k]*sig_mu_hat[str(key)] 
                         for k, key in enumerate(sig_mu_hat_keys_str)])
            F.append(curr_F) #store the F
            curr_ls=[]
            
    if port_type==1:
        sum_term= sum([mkt_weights[f,:]*Fi for f, Fi in enumerate(F)]) 
        sig_weights=[]
        for f, Fi in enumerate(F):
            curr_term= mkt_weights[f,:]*(Fi + [1]*len(Fi)- sum_term) 
            sig_weights.append(curr_term) #store the current signature-weights
        
        return sig_weights, F
    
    elif port_type==2:
        sum_term= sum([Fi for Fi in F]) 

        sig_weights=[]
        for f, Fi in enumerate(F):
            curr_term= Fi + mkt_weights[f,:]*([1]*len(Fi)- sum_term) 
            sig_weights.append(curr_term) 
        
        return sig_weights, F



def optimization(x,m,Q,c, TC_mat, tc, l2_gamma,n_param):
    """
    Do convex quadratic optimization for the log-value maximization using the 
    gurobipy framework. 

    Parameters
    ----------
    x : gurobipy variable 
        Gurobipy optimization variable.
    m : gurobipy model
        Gurobipy model within which to perform optimization.
    Q : np.array 2-dim
        Quadratic matrix associated to the optimization problem.
    c : np.array 1-dim
        Vector associated to the optimization problem.
    TC_mat : np.array 2-dim
        Matrix associated to regularization for transaction costs.
    tc : float
        Proportional transaction costs.
    l2_gamma : float
        Parameter for L2-regurlarization.
    n_param : integer
        Number of parameters to be optimized.

    Returns
    -------
    val : list of floats
        List of optimized parameters.
    ins_perf : float
        In-sample performance.

    """
    
    if l2_gamma!=False:
        ident=np.diag(2*l2_gamma*np.ones((n_param)))
        Qexpr = x@Q@x + x@ident@x
    else:
        Qexpr = x@Q@x

    Lexpr= x@c.T
    
    if tc!=0:
        TC_mat2= tc*TC_mat
        TC_expr=  x@TC_mat2@x
        print("here", tc)
        m.setObjective( 0.5*Qexpr + TC_expr + Lexpr, GRB.MINIMIZE)
    else:
        m.setObjective( 0.5*Qexpr + Lexpr, GRB.MINIMIZE)

    m._vars = m.getVars()
    m.optimize()
    
    
    ins_perf = -1*(m.getObjective().getValue())
   
    l_list= x.tolist()

    val = [l.getAttr(GRB.Attr.X) for l in l_list]
    
    return val, ins_perf

def optimization_MV(x,m,Q,c,TC_mat, tc,l2_gamma,rf,n_param):
    """
    Do convex quadratic optimization for the mean-variance task using the 
    gurobipy framework. 

    Parameters
    ----------
    x : gurobipy variable 
        Gurobipy optimization variable.
    m : gurobipy model
        Gurobipy model within which to perform optimization.
    Q : np.array 2-dim
        Quadratic matrix associated to the optimization problem.
    c : np.array 1-dim
        Vector associated to the optimization problem.
    TC_mat : np.array 2-dim
        Matrix associated to regularization for transaction costs.
    tc : float
        Proportional transaction costs.
    l2_gamma : float
        Parameter for L2-regurlarization.
    rf: float
        Risk-factor of the mean-variance task. 
    n_param : integer
        Number of parameters to be optimized.

    Returns
    -------
    val : list of floats
        List of optimized parameters.
    ins_perf : float
        In-sample performance.

    """
    if l2_gamma!=False:
        ident=np.diag(l2_gamma*np.ones((n_param)))
        Qexpr = x@Q@x + x@ident@x
    else:
        Qexpr = x@Q@x

    Lexpr= x@c.T
    
    if tc!=0:
        TC_mat2= tc*TC_mat
        TC_expr=  x@TC_mat2@x
        print("here", tc)
        m.setObjective( Qexpr + TC_expr + rf*Lexpr, GRB.MINIMIZE)
    else:
        m.setObjective( Qexpr  + rf*Lexpr, GRB.MINIMIZE)
        
    m._vars = m.getVars()
    m.optimize()
    
    ins_perf=-1*(m.getObjective().getValue()) 
    
    l_list= x.tolist()
    #print(l_list)
    val = [l.getAttr(GRB.Attr.X) for l in l_list]
    
    return val, ins_perf




def relative_log_return(
        weights_denom, weights_numerator, full_hist=False, t_start=0):
    """
    Calculate the relative log-value. Once by discretization of a continuous 
    strategy and once the buy-and-hold strategy. 

    Parameters
    ----------
    weights_denom : np.array shape (T, D)
        Weights used as a benchmark.
    weights_numerator : np.array shape (T, D)
        Weights whose log-relative value shall be calculated.
    full_hist : boolean, optional
        Whether to return the entire value-process. The default is False.
    t_start : integer, optional
        The index where to start investing. The default is 0.

    Returns
    -------
    rel_log : float 
        The discretized log-relative-value of a continous strategy.
    rel_log_alt : float or np.array
        The log-relative-value(-process) of a buy-and-hold strategy.

    """
    
    
    weights_denom_quad_var, quad_var_labels, quad_var_index= get_quadratic_variation(
                            weights_denom, add_time_comp=False)
    
    rel_log=0
    for i in range(len(weights_denom[:,0])): 
        integrand_1= weights_numerator[i,:]/weights_denom[i,:]
        rel_log+= Ito_integral(integrand_1, weights_denom[i,:], 
                               full_hist, t_start=t_start)
        
        for j in range(len(weights_denom[:,0])):
            integrand_2= weights_numerator[j,:]/weights_denom[j,:]
            integrand= integrand_1*integrand_2
            integrator= weights_denom_quad_var[quad_var_dict_label(
                                                    i,j,add_time_comp=False)]
            rel_log= rel_log-0.5*Ito_integral(integrand, integrator, 
                                              full_hist,t_start=t_start)
    
    prod=[1]
    for t in range(t_start,(len(weights_denom[0,:])-1)):
        tmp=0
        for i in range(len(weights_denom[:,t])):
            tmp+= weights_numerator[i,t]/weights_denom[i,t]*weights_denom[i,t+1]
        prod.append(prod[-1]*tmp)
    
    if full_hist:
        rel_log_alt= np.log(np.array(prod))
    else:
        rel_log_alt= np.log(np.array(prod))[-1]
    
    return rel_log, rel_log_alt


def relative_log_value(
        weights_denom, weights_numerator, full_hist=False, t_start=0):
    
    """
    Calculate the relative log-value of a buy-and-hold strategy. 

    Parameters
    ----------
    weights_denom : np.array shape (D,T)
        Weights used as a benchmark. (E.g. market weights.)
    weights_numerator : np.array shape (D,T)
        Weights whose log-relative value shall be calculated.
    full_hist : boolean, optional
        Whether to return the entire value-process. The default is False.
    t_start : integer, optional
        The index where to start investing. The default is 0.

    Returns
    -------
   
    rel_log_value : float or np.array
        The log-relative-value(-process) of a buy-and-hold strategy.

    """

    
    prod=[1]
    for t in range(t_start,(len(weights_denom[0,:])-1)):
        tmp=0
        for i in range(len(weights_denom[:,t])):
            tmp+= weights_numerator[i,t]/weights_denom[i,t]*weights_denom[i,t+1]
        prod.append(prod[-1]*tmp)
    
    if full_hist:
        rel_log_value= np.log(np.array(prod))
    else:
        rel_log_value= np.log(np.array(prod))[-1]
    
    return rel_log_value


    
    
def relative_log_value_advanced(
        weights_denom, weights_numerator, caps, full_caps,  t_start=0):
    
    """
    Calculate the log-value as well as the log-realtive value relative to the
    universe and the entire market, at the final time. 

    Parameters
    ----------
    weights_denom : np.array shape (T, D)
        Weights used as a benchmark. (E.g. market weights.)
    weights_numerator : np.array shape (T, D)
        Weights whose log-relative value shall be calculated.
    caps : np.array
        Capitalization process of the universe.
    full_caps : np.array
        Capitalization process of the entire market. 
    t_start : integer, optional
        The index where to start investing. The default is 0.

    Returns
    -------
    rel_log_value : float
        Log-relative value relative to univerese at final time.
    log_value : float
        Log-value at final time.
    rel_log_value_vs_full_caps : float
        Log-relative value relative to the entire market at final time.

    """
   
    
    prod=[1]
    for t in range(t_start,(len(weights_denom[0,:])-1)):
        tmp=0
        for i in range(len(weights_denom[:,t])):
            tmp+= weights_numerator[i,t]/caps[i,t]*caps[i,t+1]
        prod.append(prod[-1]*tmp)
    
    return_caps= np.sum(caps[:,-1])/np.sum(caps[:,t_start])
    return_full_caps= np.sum(full_caps[:,-1])/np.sum(full_caps[:,t_start])
    
    log_value= np.log(np.array(prod))[-1]
    rel_log_value= np.log(np.array(prod))[-1] - np.log(return_caps)
    rel_log_value_vs_full_caps= np.log(np.array(prod))[-1] - np.log(return_full_caps)
    
    
    return rel_log_value, log_value, rel_log_value_vs_full_caps




def relative_log_value_transactioncosts(
        caps, full_caps, weights_denom, weights_numerator, prop_cost, 
        t_start=0):
    
    """
    Calculate the (relative) value of a portfolio with and without proportional 
    transaction costs. 
    
    Parameters
    ----------
    caps : np.array
        Capitalization process of the universe.
    full_caps : np.array
        Capitalization process of the entire market. 
    weights_denom : np.array shape (T, D)
        Weights used as a benchmark. (E.g. market weights.)
    weights_numerator : np.array shape (T, D)
        Weights whose log-relative value shall be calculated.
    prop_cost : float
        Proportional transaction costs. 
    t_start : integer, optional
        The index where to start investing. The default is 0.

    Returns
    -------
    values : list of floats
        Value-process under transaction costs. 
    val_rel_caps_withTC : float
        Relative value to universe with transaction costs at final time.
    val_rel_full_caps_withTC : float
        Relative value to entire market with transaction costs at final time.
    rel_log_value : float
        Relative log-value to universe without transaction costs at final time.
    log_value : float
        Log-value without transaction costs at the final time.
    rel_log_value_vs_full_caps : float
        Relative log-value to market without transaction costs at final time.

    """
    
    
    values=[1]
    for t in range(t_start,(len(weights_denom[0,:])-1)):
        
        phi_minus=np.zeros(len(weights_denom[:,t]))
        for i in range(len(weights_denom[:,t])):
            phi_minus[i]= weights_numerator[i,t]*values[-1]/caps[i,t]*caps[i,t+1]
        V_minus= np.sum(phi_minus)
        pi_minus= phi_minus/V_minus
        
        if V_minus >=0:
            def func(alpha):
                return 1-alpha - prop_cost*np.sum(np.abs(
                    alpha*weights_numerator[:,t+1]- pi_minus))
            
            root = fsolve(func, 1)[-1] #scipy.optimize.fsolve
            found_zero = np.isclose(func(root), 0.0)
            if not found_zero: #check whether this is really a zero
                print("NOT REAL ZERO!! ",func(root), flush=True)
                values.append(0)
                break
            else:
                if root < 0:
                    values.append(0)
                    break
                else:
                    values.append(root*V_minus)
                
        else: 
            values.append(0)
            break
        
    
    return_caps= np.sum(caps[:,-1])/np.sum(caps[:,t_start])
    return_full_caps= np.sum(full_caps[:,-1])/np.sum(full_caps[:,t_start])
    
    
    val_rel_caps_withTC= values[-1]/return_caps
    val_rel_full_caps_withTC= values[-1]/return_full_caps
   
    rel_log_value, log_value, rel_log_value_vs_full_caps = relative_log_value_advanced(
        weights_denom, weights_numerator, caps, full_caps,  t_start=t_start)
    
    return (values, val_rel_caps_withTC, val_rel_full_caps_withTC, 
            rel_log_value, log_value, rel_log_value_vs_full_caps)








 
