# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import os
import copy
from scipy.linalg import sqrtm
from scipy.special import binom
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

from Signature_portfolios_functions import *
from Randomize_signatures import *


def learn_sig_weights_MonteCarlo_fullyIto(
        n_stocks, order_sig, T, n_samples, n_workers, bounds, l2_gamma, 
        market_model, port_type=1, load_model=False, reuse_c_Q=False, 
        init_Cov=None, init_mu=None, beta_init=None):
    """
    Monte-Carlo log-uitlity optimization of signature portfolios of type 1 or 
    2 in simulated market. The optimized parameters are stored as well as 
    several plots produced.

    Parameters
    ----------
    n_stocks : integer
        Number of stocks in the market.
    order_sig : integer
        Order of the siganture.
    T : integer
        Number of time-steps.
    n_samples : integer
        Number of Monte-Carlo samples.
    n_workers : integer
        Number of workers for paralellization.
    bounds : integer
        Bounds on the absolute value for each optimization parameter 
        (same bound for all).
    l2_gamma : float or None
        L2-regularization parameter.
    market_model : "BS" or "VolStab" or "Sig"
        Which model to simulate.
    port_type : 1 or 2, optional
        Type of the signature portfolio. The default is 1.
    load_model : string or False, optional
        Whether to use a certain market model. If a string is provided it must 
        be the file name of the pickle file where the market is stored. 
        The default is False.
    reuse_c_Q : boolean, optional
        Whether to reuse Q and c from a given precomputed market model. 
        The default is False.
    init_Cov : np.array shape (n_stocks, n_stocks) or None, optional
        If market_model=="BS" and load_model==False a covariance matrix must 
        be provided. The default is None.
    init_mu : np.array shape (n_stocks) or None, optional
        If market_model=="BS" and load_model==False a drift vector must 
        be provided. The default is None.
    beta_init : positive float or None, optional
        If market_model=="VolStab" and load_model==False the model parameter
        beta must be provided. The default is None.

    Raises
    ------
    NotImplementedError
        If market_model is not in ["BS", "VolStab", "Sig"].
    NotImplementedError
        If market_model=="BS" and load_model==False and (np.any(init_Cov)==None 
        or np.any(init_mu)==None).
    NotImplementedError
        If market_model=="VolStab" and load_model==False and beta_init==None.
    NotImplementedError
        If load_model==False and reuse_c_Q.
    Returns
    -------
    mkt_dir : stirng
        Directory of the market model.
    weights_name : string
        Pickle-file name where the optimized parameters are stored.

    """
    
    if market_model not in ["BS", "VolStab", "Sig"]:
        raise NotImplementedError(
            "market_model has to be in ['BS', 'VolStab', 'Sig'] ")
    
    if market_model=="BS" and load_model==False:
        if np.any(init_Cov)==None or np.any(init_mu)==None:
            raise NotImplementedError(
                "If no model is loaded Cov an mu must be provided.")
            
    if market_model=="VolStab" and load_model==False:
        if beta_init==None:
            raise NotImplementedError(
                "If no model is loaded beta must be provided.")

    
    if market_model=="BS":
        if not load_model:
            if reuse_c_Q:
                raise NotImplementedError(
                    "c and Q cannot be reused without loading a market!")
            mkt = classes.GenerateMarket(
                n_stocks,n_stocks,T,Cov= init_Cov, mu=init_mu)
            mkt.generate_prices()
            mkt_price= mkt.price

        
        if load_model!=False:
            with open(load_model+'.pickle', 'rb') as f:
                mkt, mkt_price = pickle.load(f)
                
    elif market_model=="VolStab":
        def get_prices(beta=None):
            tmp_mkt = classes.GenerateVolStabMarket(n_stocks,n_stocks,T, beta=beta)
            tmp_mkt.generate_prices() 
            tmp_mkt_price2= tmp_mkt.price
            if np.any(tmp_mkt_price2==0) or np.any(np.isnan(tmp_mkt_price2)):
                tmp_mkt, tmp_mkt_price2= get_prices()
            return tmp_mkt, tmp_mkt_price2
    
        if not load_model:
            if reuse_c_Q:
                raise NotImplementedError(
                    "c and Q cannot be reused without loading a market!!")
            mkt, mkt_price= get_prices(beta_init)

        
        if load_model!=False:
            with open(load_model+'.pickle', 'rb') as f:
                mkt, mkt_price = pickle.load(f)

                
    elif market_model=="Sig":
        if not load_model:
            if reuse_c_Q:
                raise NotImplementedError("c and Q cannot be reused without loading a market!!")
            mkt = classes.GenerateSigMarket(n_stocks,n_stocks,T, order=order_sig)
            mkt.generate_prices()
            mkt_price= mkt.price
  
    
        if load_model!=False:
            with open(load_model+'.pickle', 'rb') as f:
                mkt, mkt_price = pickle.load(f)
    
            
    if load_model!=False:
        mkt_dir=load_model
    else:
        base_mkt_dir= market_model+"_Mkt_{}stocks_vers{}".format(n_stocks,str(date.today()))
        mkt_dir= version_my_dir(base_mkt_dir)
            
        os.makedirs(mkt_dir)
        with open(mkt_dir+'.pickle', 'wb') as f:
            pickle.dump((mkt, mkt_price), f)
        
    os.chdir(mkt_dir)
    

    if market_model=="BS":
        Cov= copy.deepcopy(mkt.MktCov)
        mu= copy.deepcopy(mkt.mu)


        mkt = classes.GenerateMarket(n_stocks,n_stocks,T, Cov=Cov, mu=mu)
        mkt.generate_prices()
        mkt_price= mkt.price
    elif market_model=="VolStab":
        beta= copy.deepcopy(mkt.beta)
        mkt, mkt_price= get_prices(beta)
        
    elif market_model=="Sig":
        Cov= copy.deepcopy(mkt.MktCov)
        a_coeff_list= copy.deepcopy(mkt.a_coeff_list)
        mkt = classes.GenerateSigMarket(n_stocks,n_stocks,T, Cov=Cov, 
                                        a_coeff_list=a_coeff_list, 
                                        order=order_sig)
        mkt.generate_prices()
        mkt_price= mkt.price
        

    mkt_weights= get_market_weights(mkt_price)
    mkt_weights_ous= copy.deepcopy(mkt_weights)
    time_step_ous=1/(len(mkt_weights_ous[0,:]))

    signature_keys_tolearn= create_word_list(n_stocks+1,order_sig)
    
    n_param= len(signature_keys_tolearn)*n_stocks
        
    
    if not reuse_c_Q:
        
        itr_per_worker=int(n_samples/n_workers)
        
        n_samples= itr_per_worker*n_workers
        
        
        def compute_per_worker(): 
            Q_w= np.zeros((n_param,n_param))
            c_w= np.zeros((n_param))
            for n in range(itr_per_worker):
                
                if market_model=="BS":
                    tmp_mkt = classes.GenerateMarket(n_stocks,n_stocks,T, 
                                                      Cov=Cov, mu=mu)
                    tmp_mkt.generate_prices()
                    tmp_mkt_price= tmp_mkt.price
                elif market_model=="VolStab":
                    tmp_mkt, tmp_mkt_price= get_prices(beta)
                elif market_model=="Sig":
                    tmp_mkt = classes.GenerateSigMarket(
                        n_stocks,n_stocks,T, Cov=Cov, 
                        a_coeff_list=a_coeff_list, order=order_sig)
                    tmp_mkt.generate_prices()
                    tmp_mkt_price= tmp_mkt.price
                    

                mkt_weights_ins= get_market_weights(tmp_mkt_price)
            
            
                time_step_ins=1/(len(mkt_weights_ins[0,:]))
                
                mkt_weights_quad_var, quad_var_labels, quad_var_index= get_quadratic_variation(
                    mkt_weights_ins)
            
                integrator_path, helper_path= get_integrator_and_or_helper_path(
                    mkt_weights_ins, mkt_weights_quad_var, quad_var_labels,
                    timestep=time_step_ins, result="both")
                

                signature_full_dict, signature_full_words, signature_full_keys_str= get_signature_full(
                    helper_path, order_sig)
                        

                Q_tmp= initialize_Q_mem_optim(
                    n_stocks, signature_full_keys_str, integrator_path, 
                    signature_full_dict, quad_var_index, t_start=0, 
                    port_type=port_type)
                c_tmp= initialize_c_Ito_mem_optim(
                    n_stocks, signature_full_keys_str, signature_full_dict, 
                    integrator_path, quad_var_index, t_start=0, 
                    port_type=port_type)

                            
                Q_w+=Q_tmp
                c_w+=c_tmp
            return Q_w,c_w
    
        res= Parallel(n_jobs=n_workers)(delayed(compute_per_worker)() for _ in range(n_workers))
           
        Q= np.zeros((n_param,n_param))
        c= np.zeros((n_param))
        
        for r in res:
            Q+=r[0]
            c+=r[1]
        
        with open('Q_c'+'.pickle', 'wb') as f:
            pickle.dump((Q, c, n_samples), f)
        
    else:
        with open('Q_c'+'.pickle', 'rb') as f:
            Q, c, n_samples = pickle.load(f)
    
    m= gurobipy.Model("mymodel")


    
    x= m.addMVar(n_param,lb=-bounds, ub= bounds, vtype=GRB.CONTINUOUS, name="x")
    
    val, _=optimization(x, m, Q, c, 0, 0, l2_gamma, n_param)
    sig_weights_ous, F_ous= calc_sig_portfolio_weights(
        val, mkt_weights_ous, order_sig, add_time=True, timestep=time_step_ous, 
        port_type=port_type)
    
    weights_name= "Sig_weights_order{}_length{}_nsamples{}".format(
        order_sig, T,n_samples)
    model_specs= "order{}_length{}_nsamples{}".format(order_sig, T,n_samples)

    wk_dir= weights_name
    os.makedirs(wk_dir, exist_ok=True)
    os.chdir(wk_dir)
    
    plt.figure()
    for i in range(n_stocks):
        plt.plot(mkt_price[i,:], label="Stock{}".format(i+1))
    plt.title("Market Prices: "+model_specs)
    plt.legend()
    plt.savefig("Plot_MktPrice_"+model_specs)
    plt.show()
    
    plt.figure()
    for i in range(n_stocks):
        plt.plot(mkt_weights[i,:], label="Stock{}".format(i+1))
    plt.title("Market Weights: "+model_specs)
    plt.legend()
    plt.savefig("Plot_MktWeights_"+model_specs)
    plt.show()

    
    plt.figure()
    for i in range(n_stocks):
        plt.plot(sig_weights_ous[i], label="Stock{}".format(i+1))
    plt.title("Sig Weights Out-Of-Sample: "+model_specs)
    plt.legend()
    plt.savefig("Plot_SigWeightsOus_"+model_specs)
    plt.show()
    
    if market_model=="BS":
        to_be_saved= {'l_s': val, 'outs_sig_weights': sig_weights_ous,
                      "F_ous":F_ous, "Cov": Cov, "mu":mu} 
    elif market_model=="VolStab":
        to_be_saved= {'l_s': val, 'outs_sig_weights': sig_weights_ous,
                      "F_ous":F_ous, "beta": beta}
    elif market_model=="Sig":
        to_be_saved= {'l_s': val, 'outs_sig_weights': sig_weights_ous,
                      "F_ous":F_ous, "Cov": Cov, "a_coeff_list":a_coeff_list} 

        
    with open(weights_name+'.pickle', 'wb') as f:
        pickle.dump(to_be_saved, f)
    
    return mkt_dir, weights_name







def benchmark_montecarlo(
        mkt_model, order_sig, n_stocks, T, n_samples, test_weights, 
        market_model, port_type=1):
    """
    Evaluate out-of-sample performance of optimized signature portfolios and 
    compare them to the theoretically growth-optimal portfolio. Stores the 
    respective performance info in a text file and produces several plots. 

    Parameters
    ----------
    mkt_model : "BS" or "VolStab" or "Sig"
        Which model to simulate.
    order_sig : integer
        Order of the signature.
    n_stocks : integer
        Number of stocks.
    T : integer
        Number of time steps.
    n_samples : integer
        Number of test-samples.
    test_weights : string
        Name of the file where the parameters of the signature portfolios are 
        stored.
    market_model : string
        Name of the (version of the) market model, in which the portfolios 
        were trained.
    port_type : 1 or 2, optional
        Type of siganture portfolio. The default is 1.

    Returns
    -------
    None

    """
    
    if market_model=="Sig":
        with open(mkt_model+'.pickle', 'rb') as f:
            mkt, mkt_price = pickle.load(f)
            Cov= copy.deepcopy(mkt.MktCov)
            a_coeff_list= copy.deepcopy(mkt.a_coeff_list)
            
    elif market_model=="VolStab":
        with open(mkt_model+'.pickle', 'rb') as f:
            mkt, mkt_price = pickle.load(f)
            beta= copy.deepcopy(mkt.beta)
            
    elif market_model=="BS":
        with open(mkt_model+'.pickle', 'rb') as f:
            mkt, mkt_price = pickle.load(f)
            Cov= copy.deepcopy(mkt.MktCov)
            mu= copy.deepcopy(mkt.mu)
    

    Gopt_ret=[]
    Gopt_ret_disc=[]
    Gopt_min=[]
    Gopt_max=[]
    
    sig_ret=[]
    sig_ret_disc=[]
    sig_min=[]
    sig_max=[]
    
    
    wk_dir= mkt_model+'/'+test_weights
    os.chdir(wk_dir)
    with open(test_weights+'.pickle', 'rb') as f:
        test_weights_dict = pickle.load(f)
        val = test_weights_dict['l_s']

    time_step_ins=1/(T)

    if market_model=="VolStab":
        def get_prices():
            tmp_mkt = classes.GenerateVolStabMarket(
                n_stocks,n_stocks,T, order=order_sig, beta=beta)
            tmp_mkt.generate_prices() 
            tmp_mkt_price2= tmp_mkt.price
            if np.any(tmp_mkt_price2==0) or np.any(np.isnan(tmp_mkt_price2)):
                tmp_mkt, tmp_mkt_price2= get_prices() 
            return tmp_mkt, tmp_mkt_price2


    def one_sample():
        if market_model=="Sig":
            tmp_mkt = classes.GenerateSigMarket(
                n_stocks,n_stocks,T, order=order_sig, Cov=Cov, 
                a_coeff_list=a_coeff_list)
            tmp_mkt.generate_prices()
            tmp_mkt_price= tmp_mkt.price
            
        elif market_model=="VolStab":
            tmp_mkt, tmp_mkt_price= get_prices()
            
        elif market_model=="BS":
            tmp_mkt = classes.GenerateMarket(
                n_stocks,n_stocks,T, Cov=Cov, mu=mu)
            tmp_mkt.generate_prices()
            tmp_mkt_price= tmp_mkt.price
            
        mkt_weights_ins= get_market_weights(tmp_mkt_price)
        
        helper_path= get_integrator_and_or_helper_path(
            mkt_weights_ins, timestep=time_step_ins, result="helper")

        signature_full_dict, signature_full_words, signature_full_keys_str= get_signature_full(
            helper_path, order_sig)
        
        sig_weights, F= calc_sig_portfolio_weights(
            val, mkt_weights_ins, order_sig, add_time=True, 
            timestep=time_step_ins, port_type=port_type)
        sig_weights=[ list(sig_weights[i]) for i in range(n_stocks)]
        
        if market_model=="Sig":
            g_optimal= classes.GrowthOptimal(
                n_stocks,n_stocks, T, a_list=tmp_mkt.sig_term_list,
                Sigma_list=[tmp_mkt.L])
            g_optimal.get_GO_weights()
            tmp_go_weights= np.concatenate(
                (g_optimal.go_weights.T, np.zeros((1,n_stocks)).T), axis=1)
        elif market_model=="VolStab":
            g_optimal= classes.GrowthOptimal(
                n_stocks,n_stocks, T, a_list=tmp_mkt.t_term_list,
                Sigma_list=tmp_mkt.W_term_list)
            g_optimal.get_GO_weights()
            tmp_go_weights= np.concatenate(
                (g_optimal.go_weights.T, np.zeros((1,n_stocks)).T), axis=1)
        elif market_model=="BS":
            g_optimal= classes.GrowthOptimal(
                n_stocks,n_stocks, T, a_list=[tmp_mkt.mu],
                Sigma_list=[tmp_mkt.L])
            g_optimal.get_GO_weights()
            tmp_go_weights=g_optimal.go_weights.T
            
        tmp_sig_ret, tmp_sig_ret_disc= relative_log_return(
            np.array(mkt_weights_ins), np.array(sig_weights), full_hist=True)
        tmp_go_ret, tmp_go_ret_disc= relative_log_return(
            np.array(mkt_weights_ins), tmp_go_weights, full_hist=True)

        return {"go_ret": tmp_go_ret[-1], "go_ret_disc": tmp_go_ret_disc[-1], 
                "go_min": np.min(tmp_go_ret), "go_max":np.max(tmp_go_ret),
                "sig_ret":tmp_sig_ret[-1],"sig_ret_disc":tmp_sig_ret_disc[-1], 
                "sig_min":np.min(tmp_sig_ret), "sig_max": np.max(tmp_sig_ret)}
        
    
    res= Parallel(n_jobs=-1)(delayed(one_sample)() for _ in range(n_samples))

    for r in res:
        Gopt_ret.append(r["go_ret"])
        Gopt_ret_disc.append(r["go_ret_disc"])
        Gopt_min.append(r["go_min"])
        Gopt_max.append(r["go_max"])
    
        sig_ret.append(r["sig_ret"])
        sig_ret_disc.append(r["sig_ret_disc"])
        sig_min.append(r["sig_min"])
        sig_max.append(r["sig_max"])
        
        
    ############## To compare weights ######################
    if market_model=="Sig":
        tmp_mkt = classes.GenerateSigMarket(
            n_stocks,n_stocks,T, order=order_sig, Cov=Cov, 
            a_coeff_list=a_coeff_list)
        tmp_mkt.generate_prices()
        tmp_mkt_price= tmp_mkt.price
    elif market_model=="VolStab":
        tmp_mkt, tmp_mkt_price= get_prices()
    elif market_model=="BS":
        tmp_mkt = classes.GenerateMarket(n_stocks,n_stocks,T, Cov=Cov, mu=mu)
        tmp_mkt.generate_prices()
        tmp_mkt_price= tmp_mkt.price
        
    mkt_weights_ous= get_market_weights(tmp_mkt_price)
    
    helper_path= get_integrator_and_or_helper_path(
        mkt_weights_ous, timestep=time_step_ins, result="helper")

    signature_full_dict, signature_full_words, signature_full_keys_str= get_signature_full(
        helper_path, order_sig)
    
    sig_weights, F= calc_sig_portfolio_weights(
        val, mkt_weights_ous, order_sig, add_time=True, timestep=time_step_ins, 
        port_type=port_type)
    sig_weights=[ list(sig_weights[i]) for i in range(n_stocks)]
    
    if market_model=="Sig":
        g_optimal= classes.GrowthOptimal(
            n_stocks,n_stocks, T, a_list=tmp_mkt.sig_term_list,
            Sigma_list=[tmp_mkt.L])
    elif market_model=="VolStab":
        g_optimal= classes.GrowthOptimal(
            n_stocks,n_stocks, T, a_list=tmp_mkt.t_term_list,
            Sigma_list=tmp_mkt.W_term_list)
    elif market_model=="BS":
        g_optimal= classes.GrowthOptimal(
            n_stocks,n_stocks, T, a_list=[tmp_mkt.mu],Sigma_list=[tmp_mkt.L])

    g_optimal.get_GO_weights()
    print("shape GO weights:", np.shape((g_optimal.go_weights)))
    
    plt.figure()
    for i in range(n_stocks):
        plt.plot(sig_weights[i], label="Stock{}".format(i+1))
    plt.title("Sig Weights Out-Of-Sample: ")
    plt.legend()
    plt.savefig("Plot_SigWeightsOus")
    plt.show()
    
    plt.figure()
    for i in range(n_stocks):
        plt.plot(g_optimal.go_weights[:,i], label="Stock{}".format(i+1))
    plt.title("Growth-Optimal Weights Out-Of-Sample: ")
    plt.legend()
    plt.savefig("Plot_GOWeightsOus")
    plt.show()
    
    x= [i for i in range(len(Gopt_ret))]
    plt.figure()
    plt.scatter(x,Gopt_ret, s=1,  label="log-opt")
    plt.scatter(x,sig_ret, s=1, label="sig")
    plt.title("Final return per sample")
    plt.legend()
    plt.savefig("returns_per_samples.png")
    plt.show()
    
    plt.figure()
    plt.scatter(x,Gopt_ret_disc, s=1,  label="log-opt")
    plt.scatter(x,sig_ret_disc, s=1, label="sig")
    plt.title("Final return per sample")
    plt.legend()
    plt.savefig("DISC_returns_per_samples.png")
    plt.show()
    
    plt.figure()
    plt.scatter(x,Gopt_min, s=1, label="log-opt")
    plt.scatter(x,sig_min, s=1, label="sig")
    plt.title("Lowest relative return per sample")
    plt.legend()
    plt.savefig("min_per_samples.png")
    plt.show()
    
    plt.figure()
    plt.scatter(x,Gopt_max, s=1,  label="log-opt")
    plt.scatter(x,sig_max, s=1,  label="sig")
    plt.title("Highest relative return per sample")
    plt.legend()
    plt.savefig("max_per_samples.png")
    plt.show()
    
    with open('statistics.txt', 'a') as f:
        f.write("Mean return of log-opt={}".format(
            np.mean(Gopt_ret))+ os.linesep)
        f.write("Mean return of sig={}".format(np.mean(sig_ret))+ os.linesep)
        f.write("Mean DISC return of log-opt={}".format(
            np.mean(Gopt_ret_disc))+ os.linesep)
        f.write("Mean DISC return of sig={}".format(
            np.mean(sig_ret_disc))+ os.linesep)
        f.write("Std dev of returns of log-opt={}".format(
            np.std(Gopt_ret))+ os.linesep)
        f.write("Std dev of returns of sig={}".format(
            np.std(sig_ret))+ os.linesep)
        f.write("Std dev of DISC returns of log-opt={}".format(
            np.std(Gopt_ret_disc))+ os.linesep)
        f.write("Std dev of DISC returns of sig={}".format(
            np.std(sig_ret_disc))+ os.linesep)
