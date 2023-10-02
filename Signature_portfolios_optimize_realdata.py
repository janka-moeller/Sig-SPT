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



def learn_sig_weights_REALDATA_TCgradient(
        data_name, order_sig, t_insample, t_outsample, bounds, prop_cost, 
        rank_based, objective, risk_factor=0.5,l2_gamma=False, ranks=0, 
        t_start=0, lambda_TC=None, end_time=0, base_model_dir=None, 
        randomsig=False, reuse_TC=None, proj_dim= None, n_jobs=64, 
        rand_mat_name=None):
    
    """
    Function to optimize signature or JL- or randmoized-signature portfolios
    in the rank-based or name-based setting and for either optimizing
    log-untility or mean-variance. In the namebased setting the optimization
    is performed with or without a regularization for a given level of 
    transactions costs. If such a regularization is chosen, the regularization
    parameter is also optimized. Several performance values are stored in a 
    pickle and text file, as well as several plots are produced. 

    Parameters
    ----------
    data_name : string
        The name of the dataset/market under which the data is stored.
    order_sig : integer
        Order of the signature.
    t_insample : integer
        Number of days in the insample investment period.
    t_outsample : integer
        Number of days in the out-of-sample investment period.
    bounds : float
        Bound for the optimization parameters (will be the same bound for all).
    prop_cost : float
        Proportional transaction costs.
    rank_based : boolean
        Whether a ranked market is considered.
    objective : "MV" or "Log-Utility"
        Whether to optimize the mean-variance or the log-utility.
    risk_factor : float, optional
        Risk-factor of the mean-variance optimization. Only needed if 
        objective=="MV". The default is 0.5.
    l2_gamma : False or float, optional
        L2-regularization parameter. The default is False.
    ranks : list of ints, optional
        Ranks of the stocks to include in the universe, which in the 
        namebased-setting refers to the ranks of the stocks on the first 
        day of the dataset.  The default is 0.
    t_start : integer, optional
        Integer where to start investing. The default is 0.
    lambda_TC : float or None, optional
        If None no regularization for transaction costs is applied, otherwise 
        this is the starting value for the search of the optimial 
        regularization parameter. The default is None.
    end_time : integer, optional
        How many days to cut from the end of the dataset. The default is 0.
    base_model_dir : string, optional
        Directionary to store outputs. The default is None.
    randomsig : False, "JL" or "RANDOMIZED", optional
        Whether to use a randomization of the signature and if so which one.
        The default is False.
    reuse_TC : None or np.array, optional
        If None the matrix for the regularization for transaction costs is 
        calculated otherwise a precomputed matrix can be provided. 
        The default is None.
    proj_dim : integer, optional
        Dimension of a randomization of the signature. Only needed if 
        randomsig=="JL" or randomsig=="RANDOMIZED". The default is None.
    n_jobs : integer, optional
        Number of workers to be used in paralleization. The default is 64.
    rand_mat_name : string or None, optional
        If None new random coefficients for the JL- or Randomized-signature 
        are initialized and stored, otherwise the (pickle-)file name of 
        precomputed coefficients can be passed. Only needed if 
        randomsig!=False. The default is None.

    Raises
    ------
    NotImplementedError
        If randomsig is not in [False, "JL", "RANDOMIZED"].
        
    NotImplementedError
        If objective is not in ["MV", "Log-Utility"].
    
    NotImplementedError
        If rank_based and lambda_TC!=None. In the rank-based setting, no 
        regularization for transaction costs can be used.

    Returns
    -------
    
    TC_mat : np.array or None
        If a regularization for transaction costs is used the associated 
        quadratic matrix is returned.
    
    rand_mat_name : string or None
        If randomsig!=None name of the file where the coefficients of the 
        randomization are stored. 

    """ 


    if randomsig not in [False, "JL", "RANDOMIZED"]:
        raise NotImplementedError(
            "randomsig needs to be in [False, 'JL', 'RANDOMIZED']")
        return 0
    
    if objective not in ["MV", "Log-Utility"]:
        raise NotImplementedError(
            "objective needs to be in ['MV','Log-Utility']")
        return 0
    
    if rank_based and lambda_TC!=None:
        raise NotImplementedError(
            "Optimization under transaction cost is not possible in rank-based setting")
        return 0
    
    if randomsig!=False:
        if rand_mat_name==None:
            rand_mat_was_none =True
        else:
            rand_mat_was_none = False
    
    normalize= False
    incl_MCAP=False
    
    if rank_based:    
        with open(data_name+'.pickle', 'rb') as f:
            	df_argsort, df_sorted= pickle.load(f)
    
        full_caps= np.array(df_sorted).T
        rank_idx= [-1*r for r in ranks]
        caps= full_caps[rank_idx,:]
        dates= df_sorted.index.to_list()
               
        company_names= [df_sorted.columns.to_list()[r] for r in rank_idx]

    else:
        with open(data_name+'.pickle', 'rb') as f:
            df, start_argsort= pickle.load(f)
        
        full_caps= np.array(df).T
        rank_idx= [-1*r for r in ranks]
        start_idx= start_argsort[rank_idx]
        caps= full_caps[start_idx,:]
        
        
        dates= df.index.to_list()
        company_names= [df.columns.to_list()[r] for r in start_idx]
            
    if len(ranks)<=10:
        data_name=data_name + str(ranks)
    else:
        data_name=data_name+"_from"+str(ranks[0])+"until"+str(ranks[-1])
    
    total_time= (t_insample+t_outsample)+t_start
    full_caps= full_caps[:,:-end_time]
    caps= caps[:,:-end_time]
    dates= dates[:-end_time]
    full_caps= full_caps[:,-total_time:]
    caps= caps[:,-total_time:]
    dates=dates[-total_time:]
    n_stocks= len(ranks)
                

    mkt_weights= get_market_weights(caps)
    if incl_MCAP:
        total_cap=np.sum(caps,axis=0)/10**(8)
    else:
        total_cap=None
    
    if incl_MCAP:
        signature_keys_tolearn= create_word_list(n_stocks+2,order_sig)
    else:
        signature_keys_tolearn= create_word_list(n_stocks+1,order_sig)

    time_step_ins= 1/(t_insample+t_start)
    time_step_ous= 1/(t_outsample+t_start)
    

    os.makedirs(data_name, exist_ok=True)
    
    os.chdir(data_name)
    
    if l2_gamma==False:
        l2_gamma_str="False"
    else:
        l2_gamma_str="{:e}".format(l2_gamma).replace('.','').replace('0','')

    prop_cost_str="{:e}".format(prop_cost).replace('.','').replace('0','')
    
    if lambda_TC!=None:
        lambda_TC_str="{:e}".format(lambda_TC).replace('.','').replace('0','')
    else:
        lambda_TC_str="None"
    
    if incl_MCAP:
        model_name=objective+"_MCAP_Sig_order{}_start-ins{}_ti{}_to{}_b{}_\
            l2{}_tstart{}_propcost{}_lambdaTC{}_projdim{}".format(
            order_sig,  dates[0], t_insample,t_outsample, bounds, 
            l2_gamma_str, t_start, prop_cost_str, lambda_TC_str, proj_dim)
    else:
        model_name=objective+"_Ito_Sig_order{}_start-ins{}_ti{}_to{}_b{}_\
            l2{}_tstart{}_propcost{}_lambdaTC{}_projdim{}".format(order_sig,  
            dates[0], t_insample,t_outsample, bounds, l2_gamma_str, t_start, 
            prop_cost_str, lambda_TC_str, proj_dim)

    if normalize:
        model_name= "NORM_"+model_name

    if objective=="MV":
        rf_str="{:e}".format(risk_factor).replace('.','').replace('0','')
        model_name= model_name+ "_riskfactor"+rf_str
    
    if randomsig=="JL":
        model_name= "JLSIG_"+model_name
    elif randomsig=="RANDOMIZED":
        model_name= "RSIG_"+model_name
    
   
   
    if randomsig!=False:
        if rand_mat_name==None:
            base_rand_mat_name= randomsig+"_Matrix_sigorder{}_projdim{}_\
                vers{}".format(
                order_sig, proj_dim,str(date.today()))
            rand_mat_name= version_my_dir(base_rand_mat_name)
            os.makedirs(rand_mat_name)
        else: 
            with open(rand_mat_name+'.pickle', 'rb') as f:
                rand_mat_list= pickle.load(f)
    
    
    ous_dates=[]
    ins_perf=[]
    
    mkt_weights_ins= mkt_weights[:,0:(t_insample+t_start)]
    caps_ins= caps[:,0:(t_insample+t_start)]
    full_caps_ins= full_caps[:,0:(t_insample+t_start)]
    
    if normalize:
        norm_caps_ins= np.array([caps_ins[i,:]/caps_ins[i,0] 
                                 for i in range(len(caps_ins[:,0]))])
        norm_mkt_weights_ins= get_market_weights(norm_caps_ins)
    else:
        norm_mkt_weights_ins=None
    
    if incl_MCAP:
        total_cap_ins= total_cap[:,0:(t_insample+t_start)]
    else:
        total_cap_ins=None
    
    mkt_weights_quad_var, quad_var_labels, quad_var_index= \
        get_quadratic_variation(mkt_weights_ins, incl_MCAP=incl_MCAP)

    if not normalize:
        integrator_path, helper_path= get_integrator_and_or_helper_path(
            mkt_weights_ins, mkt_weights_quad_var, quad_var_labels,
            timestep=time_step_ins, result="both", total_cap=total_cap_ins, 
            incl_MCAP=incl_MCAP)
    else:
        helper_path= get_integrator_and_or_helper_path(
            norm_mkt_weights_ins, timestep=time_step_ins, 
            result="helper", total_cap=total_cap_ins, incl_MCAP=incl_MCAP)
    
    
    
    if randomsig==False:
        signature_full_dict, signature_full_words, signature_full_keys_str= \
            get_signature_full(helper_path, order_sig)
        rand_mat_list=None
    elif randomsig=="JL":
        if rand_mat_was_none:
            signature_full_dict, signature_keys_tolearn, rand_mat_list= \
                get_JL_signature(
                    order_sig, proj_dim, n_jobs, helper_path, rand_mat=None)
            signature_full_keys_str= copy.deepcopy(signature_keys_tolearn)
            with open(rand_mat_name+'.pickle', 'wb') as f:
                pickle.dump(rand_mat_list, f)
        else:
            signature_full_dict, signature_keys_tolearn= get_JL_signature(
                order_sig, proj_dim, n_jobs, helper_path, 
                rand_mat=rand_mat_list)
            signature_full_keys_str= copy.deepcopy(signature_keys_tolearn)
    elif randomsig=="RANDOMIZED":
        if rand_mat_was_none:
            signature_full_dict, signature_keys_tolearn, rand_mat_list= \
                get_R_sig(proj_dim, helper_path, rand_mat_list=None)
            signature_full_keys_str= copy.deepcopy(signature_keys_tolearn)
            with open(rand_mat_name+'.pickle', 'wb') as f:
                pickle.dump(rand_mat_list, f)
        else:
            signature_full_dict, signature_keys_tolearn= get_R_sig(proj_dim, 
                helper_path, rand_mat_list=rand_mat_list)
            
            signature_full_keys_str= copy.deepcopy(signature_keys_tolearn)
            

    if randomsig==False:
        if incl_MCAP:
            signature_keys_tolearn= create_word_list(n_stocks+2,order_sig)
        else:
            signature_keys_tolearn= create_word_list(n_stocks+1,order_sig)



    if incl_MCAP:
        raise NotImplementedError("MCAP version in currently not implemented")
    else:
        if objective=="Log-Utility":
            Q= initialize_Q_mem_optim(n_stocks, signature_full_keys_str, 
                                      integrator_path, signature_full_dict, 
                                      quad_var_index, t_start=t_start)
            c= initialize_c_Ito_mem_optim(n_stocks, signature_full_keys_str, 
                                          signature_full_dict, integrator_path, 
                                          quad_var_index, t_start=t_start)
        elif objective=="MV":
            Q,c= initialize_Q_c_MV(n_stocks, signature_keys_tolearn, 
                                   signature_full_dict, 
                                   weights=np.array(mkt_weights_ins), 
                                   t_start=t_start)

    
    if lambda_TC!=None:
        if np.all(reuse_TC)==None:
            print("time before TC_mat:", datetime.now(), flush=True)
            TC_mat= initialize_TC_mat(n_stocks, signature_keys_tolearn, 
                                      signature_full_dict, 
                                      weights=mkt_weights_ins, t_start=t_start, 
                                      n_jobs=n_jobs)
            print("time after TC_mat:", datetime.now(), flush=True)
            TC_mat=TC_mat/(t_insample)
        else: 
            TC_mat= copy.deepcopy(reuse_TC)
    else:
        TC_mat= 0
    
    
    if lambda_TC==None:    
        m= gurobipy.Model("mymodel")
        
            
        n_param= len(signature_keys_tolearn)*n_stocks
    
        x= m.addMVar(n_param,lb=-bounds, ub= bounds, vtype=GRB.CONTINUOUS, 
                     name="x")
        
        if objective=="Log-Utility":
            val, tmp_ins_perf=optimization(x, m, Q, c, 0, 0, l2_gamma, n_param)  
        elif objective=="MV":
            val, tmp_ins_perf=optimization_MV(
                x, m, Q, c, 0, 0, l2_gamma, risk_factor, n_param)  
            
        
        print("ins_perf",ins_perf)
        
        tc_result=np.nan
        print("COV_term", val@Q@val)
        print("E_term", val@c.T)

    else: 
        def optim_func(tc):
            m= gurobipy.Model("mymodel")
            n_param= len(signature_keys_tolearn)*n_stocks
            x= m.addMVar(n_param,lb=-bounds, ub= bounds, vtype=GRB.CONTINUOUS, 
                         name="x")
            
            if objective=="Log-Utility":
                val, _=optimization(x, m, Q, c, TC_mat, tc, l2_gamma, n_param)  
            elif objective=="MV":
                val, _=optimization_MV(
                    x, m, Q, c, TC_mat, tc, l2_gamma, risk_factor, n_param)  
            
            sig_weights_ins, F_ins= calc_sig_portfolio_weights(
                val, mkt_weights_ins, order_sig, add_time=True, 
                timestep=time_step_ins, total_cap=total_cap_ins, 
                incl_MCAP=incl_MCAP, normalize=normalize, 
                mkt_weights_norm=norm_mkt_weights_ins,randomsig=randomsig, 
                n_jobs=n_jobs, proj_dim=proj_dim, rand_mat_list=rand_mat_list, 
                reuse_sig_mu_hat=(signature_full_dict, signature_keys_tolearn))
            
            sig_weights_ins=[ list(sig_weights_ins[i]) 
                             for i in range(n_stocks)]
            
            info_ins= relative_log_value_transactioncosts(
                caps=caps_ins, full_caps=full_caps_ins, 
                weights_denom=np.array(mkt_weights_ins), 
                weights_numerator=np.array(sig_weights_ins), 
                prop_cost=prop_cost,  t_start=t_start)

            sig_value_withTC_ins= info_ins[0]
            
            return -1*sig_value_withTC_ins[-1]
        
        #checking that we dont start in ruin area
        while (-1*optim_func(lambda_TC))<10**(-4): 
            print("returned", -1*optim_func(lambda_TC))
            lambda_TC+=0.5
            print("trying", lambda_TC)
                
        tc_result_object= minimize(optim_func, lambda_TC, 
                                   bounds=[(10**(-8), 100000)])
                
        tc_result= tc_result_object.x[0]
        print(tc_result_object.message)
                
        m= gurobipy.Model("mymodel")
        n_param= len(signature_keys_tolearn)*n_stocks
        x= m.addMVar(n_param,lb=-bounds, ub= bounds, vtype=GRB.CONTINUOUS, 
                     name="x")
        
        if objective=="Log-Utility":
            val, _=optimization(x, m, Q, c, TC_mat, tc_result, l2_gamma, 
                                n_param)  
        elif objective=="MV":
            val, _=optimization_MV(
                x, m, Q, c, TC_mat, tc_result, l2_gamma, risk_factor, n_param)  
        
        
        tc_result_str="{:e}".format(tc_result).replace('.','').replace('0','')
        model_name=model_name+"_tcfound"+tc_result_str
        
    
    sig_weights_ins, F_ins= calc_sig_portfolio_weights(val, mkt_weights_ins, 
                                order_sig, add_time=True, 
                                timestep=time_step_ins, 
                                total_cap=total_cap_ins, incl_MCAP=incl_MCAP, 
                                normalize=normalize, 
                                mkt_weights_norm=norm_mkt_weights_ins,
                                randomsig=randomsig, n_jobs=n_jobs, 
                                proj_dim=proj_dim, rand_mat_list=rand_mat_list)
    
    sig_weights_ins=[ list(sig_weights_ins[i]) for i in range(n_stocks)]

        
    info_ins= relative_log_value_transactioncosts(
        caps=caps_ins, full_caps=full_caps_ins, 
        weights_denom=np.array(mkt_weights_ins), 
        weights_numerator=np.array(sig_weights_ins), 
        prop_cost=prop_cost,  t_start=t_start)
    
    sig_value_withTC_ins= info_ins[0]
    sig_value_withTC_ins_rel_caps= info_ins[1]
    sig_value_withTC_ins_rel_full_caps= info_ins[2]
    sig_ins_rel_log_value= info_ins[3]
    sig_ins_log_value= info_ins[4]
    sig_ins_rel_log_full_caps= info_ins[5]
    
            
    if sig_value_withTC_ins_rel_caps<0:
        sig_value_withTC_ins_rel_caps_log= np.log(0)
    else:
        sig_value_withTC_ins_rel_caps_log= np.log(
            sig_value_withTC_ins_rel_caps)

                 
    mkt_weights_ous= mkt_weights[:,-(t_outsample+t_start):]
        
        
    caps_ous= caps[:,-(t_outsample+t_start):]
    full_caps_ous= full_caps[:,-(t_outsample+t_start):]
    
    if normalize:
        norm_caps_ous= np.array([caps_ous[i,:]/caps_ous[i,0] 
                                 for i in range(len(caps_ous[:,0]))])
        norm_mkt_weights_ous= get_market_weights(norm_caps_ous)
    else:
        norm_mkt_weights_ous= None
        
    if incl_MCAP:
        total_cap_ous= total_cap[:,-(t_outsample+t_start):]
    else:
        total_cap_ins= None
        total_cap_ous= None
  
        
    sig_weights_ous, F_ous= calc_sig_portfolio_weights(val, mkt_weights_ous, 
                                order_sig, add_time=True, 
                                timestep=time_step_ous, 
                                total_cap=total_cap_ous, 
                                incl_MCAP=incl_MCAP, normalize=normalize, 
                                mkt_weights_norm=norm_mkt_weights_ous,
                                randomsig=randomsig, n_jobs=n_jobs, 
                                proj_dim=proj_dim, rand_mat_list=rand_mat_list)
        
    ins_date= [0,0]
    ous_date= [0,0]
        
    ins_date[0]= dates[t_start]
    ins_date[1]= dates[t_start+t_insample-1]
        
    ous_date[0]= dates[-(t_outsample)]
    ous_date[1]= dates[-1]
    
    
    ous_dates= dates[-t_outsample:]
    ous_dates= [str(d)[0:4]+'-'+str(d)[4:6]+'-'+str(d)[6:8] for d in ous_dates]
    
    step_dates=int(len(ous_dates)/3)
    ous_dates_used=[ous_dates[0],ous_dates[step_dates], 
                    ous_dates[2*step_dates], ous_dates[-1]]
    
    ins_dates= dates[t_start:t_start+t_insample]
    ins_dates= [str(d)[0:4]+'-'+str(d)[4:6]+'-'+str(d)[6:8] for d in ins_dates]
    
    step_ins_dates=int(len(ins_dates)/3)
    ins_dates_used=[ins_dates[0],ins_dates[step_ins_dates], 
                    ins_dates[2*step_ins_dates], ins_dates[-1]]
    
    
    if base_model_dir==None:
        if randomsig!=False:
            model_dir=os.path.join(rand_mat_name,model_name)
        else:
            model_dir=copy.copy(model_name)
    else: 
        model_dir=os.path.join(base_model_dir,model_name)
    os.makedirs(model_dir, exist_ok=True)
    os.chdir(model_dir)
        
    curr_model_dir= os.getcwd()
        
    ous_dir= "OUT_of_sample_{}-{}".format(ous_date[0],ous_date[1])
        

    os.makedirs(ous_dir, exist_ok=True)
    os.chdir(ous_dir)

    
    plt.figure()
    for i in range(n_stocks):
        plt.plot(caps[i,:], label=company_names[i])
    plt.title("Market Caps: "+data_name)
    if n_stocks<10:
        plt.legend()
    plt.savefig("Plot_CAPS_"+data_name)
    plt.show()
    
    plt.figure()
    for i in range(n_stocks):
        plt.plot(mkt_weights[i,t_start:], label=company_names[i])
    plt.title("Market Weights: "+data_name)
    if n_stocks<10:
        plt.legend()
    plt.savefig("Plot_MktWeights_"+data_name)
    plt.show()
    
    avg_mkt_weights_ous= np.mean(mkt_weights_ous[:,t_start:], axis=1)
    #finds the indices of the companies with maximal average out-of-sample 
    #market weights
    ind_max_avg_mkt_weights_ous=avg_mkt_weights_ous.argsort()[-min(n_stocks,
                                                                  5):].tolist()
        
    plt.figure()
    for i in range(n_stocks):
        if i in ind_max_avg_mkt_weights_ous:
            plt.plot(mkt_weights_ous[i,t_start:], label=company_names[i])
        else:
            plt.plot(mkt_weights_ous[i,t_start:])
    plt.title("Market Weights OUS: "+data_name)
    plt.xticks([0, step_dates, 2*step_dates,t_outsample], ous_dates_used)
    plt.xlabel("Date (YYYY-MM-DD)")
    plt.legend()
    plt.savefig("Plot_MktWeights_OUS_"+data_name)
    plt.show()
    
    if normalize:
        plt.figure()
        for i in range(n_stocks):
            plt.plot(norm_mkt_weights_ous[i,t_start:], label=company_names[i])
        plt.title("NORM_Market Weights OUS: "+data_name)
        plt.xticks([0, step_dates, 2*step_dates,t_outsample], ous_dates_used)
        plt.xlabel("Date (YYYY-MM-DD)")
        #plt.legend()
        plt.savefig("Plot_NORM_MktWeights_OUS_"+data_name)
        plt.show()
    
    
    
    abs_avg_sig_weights_ins=np.abs(
        np.mean(np.array(sig_weights_ins)[:,t_start:], axis=1))
    #finds the indices of the companies with maximal average in-sample 
    #sig weights
    ind_max_abs_avg_sig_weights_ins=abs_avg_sig_weights_ins.argsort()[-min(n_stocks,5):].tolist()
    plt.figure()
    for i in range(n_stocks):
        if i in ind_max_abs_avg_sig_weights_ins:
            plt.plot(sig_weights_ins[i][t_start:], label=company_names[i])
        else:
            plt.plot(sig_weights_ins[i][t_start:])
    plt.title("Sig Weights In-Sample {}-{}".format(ins_date[0], ins_date[1]))
    plt.xticks([0, step_ins_dates, 2*step_ins_dates,t_insample], ins_dates_used)
    plt.xlabel("Date (YYYY-MM-DD)")
    plt.legend()
    plt.savefig("Plot_SigWeightsIns_{}-{}".format(ins_date[0], ins_date[1]))
    plt.show()
    
    
    abs_avg_sig_weights_ous=np.abs(
        np.mean(np.array(sig_weights_ous)[:,t_start:], axis=1))
    #finds the indices of the companies with maximal average in-sample 
    #sig weights
    ind_max_abs_avg_sig_weights_ous=abs_avg_sig_weights_ous.argsort()[-min(n_stocks,5):].tolist()
    plt.figure()
    for i in range(n_stocks):
        if i in ind_max_abs_avg_sig_weights_ous:
            plt.plot(sig_weights_ous[i][t_start:], label=company_names[i]) 
        else:
            plt.plot(sig_weights_ous[i][t_start:])
    plt.title("Sig Weights Out-Of-Sample {}-{}".format(ous_date[0], 
                                                       ous_date[1]))
    plt.xticks([0, step_dates, 2*step_dates,t_outsample], ous_dates_used)
    plt.xlabel("Date (YYYY-MM-DD)")
    plt.legend()
    plt.savefig("Plot_SigWeightsOus_{}-{}".format(ous_date[0], ous_date[1]))
    plt.show()
        
        

    sig_weights_ous=[ list(sig_weights_ous[i]) for i in range(n_stocks)]   

     

    sig_ins_rel_log_value_full_hist= relative_log_value(
        np.array(mkt_weights_ins), np.array(sig_weights_ins), full_hist=True, 
        t_start=t_start)
    sig_ous_rel_log_value_full_hist= relative_log_value(
        np.array(mkt_weights_ous), np.array(sig_weights_ous), full_hist=True, 
        t_start=t_start)
        
    info_ous = relative_log_value_transactioncosts(
        caps=caps_ous, full_caps=full_caps_ous,
        weights_denom=np.array(mkt_weights_ous), 
        weights_numerator=np.array(sig_weights_ous), 
        prop_cost=prop_cost,  t_start=t_start)
    
    
    sig_value_withTC_ous= info_ous[0]
    sig_value_withTC_ous_rel_caps= info_ous[1]
    sig_value_withTC_ous_rel_full_caps=info_ous[2]
    sig_ous_rel_log_value= info_ous[3]
    sig_ous_log_value= info_ous[4]
    sig_ous_rel_log_full_caps= info_ous[5]
            

    plt.figure()
    plt.plot(sig_ous_rel_log_value_full_hist)
    plt.title('Relative Log-Value to Market (at T: {:.4f})'.format(
        sig_ous_rel_log_value_full_hist[-1]))
    plt.xticks([0, step_dates, 2*step_dates,t_outsample], ous_dates_used)
    plt.xlabel("Date (YYYY-MM-DD)")
    plt.savefig("OUS_Rel_Log_Ret_alt")#+test_weights)
    plt.show()
    

    plt.figure()
    plt.plot(sig_ins_rel_log_value_full_hist)
    plt.title('Relative Log-Value to Market (at T: {:.4f})'.format(
        sig_ins_rel_log_value_full_hist[-1]))
    plt.xticks([0, step_ins_dates, 2*step_ins_dates,t_insample], 
               ins_dates_used)
    plt.xlabel("Date (YYYY-MM-DD)")
    plt.savefig("INS_Rel_Log_Ret_alt")
    plt.show()
        

    

    if sig_value_withTC_ous_rel_caps<0:
        sig_value_withTC_outsample_rel_log=(np.log(0))
    else:
        sig_value_withTC_outsample_rel_log=(np.log(
            sig_value_withTC_ous_rel_caps))
               
    
    to_be_saved= {'insample_sig_weights': sig_weights_ins, "F_ins": F_ins, 
                  'mkt_weights_ins': mkt_weights_ins, 
                  'mkt_weights_ous': mkt_weights_ous,
                  'caps_ins': caps_ins, 'caps_ous': caps_ous,
                  'l_s': val, 'outs_sig_weights': sig_weights_ous,
                  "F_ous":F_ous, "sig_ous_ret":np.nan, 
                  "sig_ous_ret_alt":sig_ous_rel_log_value_full_hist,
                  "sig_ins_ret":np.nan, 
                  "sig_ins_ret_alt":sig_ins_rel_log_value_full_hist, 
                  "sig_log_value":sig_ous_log_value,
                  "sig_rel_log_full_caps":sig_ous_rel_log_full_caps, 
                  "sig_ret_with_TC_ous":sig_value_withTC_ous, 
                  "sig_ret_with_TC_ins":sig_value_withTC_ins,
                  "TC_found": tc_result}

        
        
    with open(ous_dir+'.pickle', 'wb') as f:
        pickle.dump(to_be_saved, f)
    
    os.chdir(curr_model_dir)
    
    
    with open('INFO.txt', 'w') as f:

        f.write(r" Dates & Ins Log-Relative Value & Ous Log-Relative Value & Ous Log Value & Ous Log Rel Value vs Full Mkt& Insample Value with TC &  Insample Log-Rel Value with TC & Out-of-sample Value with TC & Out-of-sample Log-Rel Value with TC & TC_found\\"+ os.linesep)
        
        f.write(r"{}-{} & {:.4f} & {:.4f} & {:.4f}& {:.4f}& {:.4f} &{:.4f} & {:.4f} &{:.4f}  &{} ".format(ous_date[0], ous_date[1], sig_ins_rel_log_value,sig_ous_rel_log_value_full_hist[-1], sig_ous_log_value, sig_ous_rel_log_full_caps, sig_value_withTC_ins[-1], sig_value_withTC_ins_rel_caps_log,sig_value_withTC_ous[-1],sig_value_withTC_outsample_rel_log, tc_result )+ os.linesep)

    return TC_mat, rand_mat_name






def CV_sig_weights_timeaverage_REALDATA(
        data_name, order_sig, t_insample, t_outsample, bounds, 
        rank_based,l2_gamma_list=False, ranks=0, t_start=0, 
        end_time=250, base_model_dir=None, randomsig=False, 
        rand_mat_name=None, proj_dim= None, n_jobs=64):
    """
    Performs cross-validation (grid-)search for the L2-regularization 
    parameter used in the log-utility optimization without transaction cost in 
    either rank-based or namebased market.

    Parameters
    ----------
    data_name : string
        Name of the dataset/market.
    order_sig : integer
        Order of the signature.
    t_insample : integer
        Number of days in the insample investment period.
    t_outsample : integer
        Number of days in the out-of-sample investment period.
    bounds : float
        Bound for the optimization parameters (will be the same bound for all).
    rank_based : boolean
        Whether a ranked market is considered.
    l2_gamma_list : list of floats, optional
        Grid for the regularization parameter to be optimized during cross-
        validation. The default is False.
    ranks : list of ints, optional
        Ranks of the stocks to include in the universe, which in the 
        namebased-setting refers to the ranks of the stocks on the first 
        day of the dataset.  The default is 0.
    t_start : integer, optional
        Integer where to start investing. The default is 0.
    end_time : integer, optional
        How many days to cut from the end of the dataset. The default is 0.
    base_model_dir : string, optional
        Directionary to store outputs. The default is None.
    randomsig : False, "JL" or "RANDOMIZED", optional
        Whether to use a randomization of the signature and if so which one.
        The default is False.
    rand_mat_name : string or None, optional
        If None new random coefficients for the JL- or Randomized-signature 
        are initialized and stored, otherwise the (pickle-)file name of 
        precomputed coefficients can be passed. Only needed if 
        randomsig!=False. The default is None.
    proj_dim : integer, optional
        Dimension of a randomization of the signature. Only needed if 
        randomsig=="JL" or randomsig=="RANDOMIZED". The default is None.
    n_jobs : integer, optional
        Number of workers to be used in paralleization. The default is 64.
        

    Raises
    ------
    NotImplementedError
        If randomsig not in [False, "JL", "RANDOMIZED"]

    Returns
    -------
    l2_gamma_opt : float
        The best L2-regularization parameter found in cross-validation. 
    rand_mat_name : string or None
        If randomsig!=None name of the file where the coefficients of the 
        randomization are stored. 
    """
  
    if randomsig not in [False, "JL", "RANDOMIZED"]:
        raise NotImplementedError("randomsig needs to be in [False, 'JL', 'RANDOMIZED']")
        return 0
    
    if randomsig!=False:
       if rand_mat_name==None:
           rand_mat_was_none =True
       else:
           rand_mat_was_none = False
    
    normalize= False
    incl_MCAP=False
    if rank_based:    
        with open(data_name+'.pickle', 'rb') as f:
            	df_argsort, df_sorted= pickle.load(f)
    
        full_caps= np.array(df_sorted).T
        rank_idx= [-1*r for r in ranks]
        caps= full_caps[rank_idx,:]
        dates= df_sorted.index.to_list()
        company_names= [df_sorted.columns.to_list()[r] for r in rank_idx]

    else:
        with open(data_name+'.pickle', 'rb') as f:
            df, start_argsort= pickle.load(f)
        
        full_caps= np.array(df).T
        rank_idx= [-1*r for r in ranks]
        start_idx= start_argsort[rank_idx]
        caps= full_caps[start_idx,:]
        
        dates= df.index.to_list()
        company_names= [df.columns.to_list()[r] for r in start_idx]
    
            
    total_time= (t_insample+2*t_outsample)+t_start
    full_caps= full_caps[:,:-end_time]
    caps= caps[:,:-end_time]
    dates= dates[:-end_time]
    full_caps= full_caps[:,-total_time:]
    caps= caps[:,-total_time:]
    dates=dates[-total_time:]
    n_stocks= len(ranks)
                

    mkt_weights= get_market_weights(caps)
    if incl_MCAP:
        total_cap=np.sum(caps,axis=0)/10**(8)
    else:
        total_cap=None
    
    if incl_MCAP:
        signature_keys_tolearn= create_word_list(n_stocks+2,order_sig)
    else:
        signature_keys_tolearn= create_word_list(n_stocks+1,order_sig)

    time_step_ins= 1/(t_insample+t_start)
    
    
    if randomsig!=False:
        base_dir= os.getcwd()
        if len(ranks)<=10:
            store_data_name=data_name + str(ranks)
        else:
            store_data_name=data_name+"_from"+str(ranks[0])+"until"+str(ranks[-1])
        
        os.makedirs(store_data_name, exist_ok=True)
    
        os.chdir(store_data_name)
        if rand_mat_name==None:
            base_rand_mat_name= randomsig+"_Matrix_sigorder{}_projdim{}_vers{}".format(
                order_sig, proj_dim,str(date.today()))
            rand_mat_name= version_my_dir(base_rand_mat_name)
        else: 
            with open(rand_mat_name+'.pickle', 'rb') as f:
                rand_mat_list= pickle.load(f)
    

    mkt_weights_ins= mkt_weights[:,0:(t_insample+t_start)]
    caps_ins= caps[:,0:(t_insample+t_start)]
    
    if normalize:
        norm_caps_ins= np.array([caps_ins[i,:]/caps_ins[i,0] 
                                 for i in range(len(caps_ins[:,0]))])
        norm_mkt_weights_ins= get_market_weights(norm_caps_ins)
    else:
        norm_mkt_weights_ins=None
    
    if incl_MCAP:
        total_cap_ins= total_cap[:,0:(t_insample+t_start)]
    else:
        total_cap_ins=None
    
    mkt_weights_quad_var, quad_var_labels, quad_var_index= get_quadratic_variation(
        mkt_weights_ins, incl_MCAP=incl_MCAP)

    if not normalize:
        integrator_path, helper_path= get_integrator_and_or_helper_path(
            mkt_weights_ins, mkt_weights_quad_var, quad_var_labels,
            timestep=time_step_ins, result="both", total_cap=total_cap_ins, 
            
            incl_MCAP=incl_MCAP)
    if normalize:
        helper_path= get_integrator_and_or_helper_path(
            norm_mkt_weights_ins, timestep=time_step_ins, result="helper", 
            total_cap=total_cap_ins, incl_MCAP=incl_MCAP)
    
    
    
    if randomsig==False:
        signature_full_dict, signature_full_words, signature_full_keys_str= get_signature_full(
            helper_path, order_sig)
        rand_mat_list=None
    elif randomsig=="JL":
        if rand_mat_was_none:
            signature_full_dict, signature_keys_tolearn, rand_mat_list= get_JL_signature(
                order_sig, proj_dim, n_jobs, helper_path, rand_mat=None)
            signature_full_keys_str= copy.deepcopy(signature_keys_tolearn)
            with open(rand_mat_name+'.pickle', 'wb') as f:
                pickle.dump(rand_mat_list, f)
        else:
            signature_full_dict, signature_keys_tolearn= get_JL_signature(
                order_sig, proj_dim, n_jobs, helper_path, 
                rand_mat=rand_mat_list)
            signature_full_keys_str= copy.deepcopy(signature_keys_tolearn)
            
        os.chdir(base_dir)
    elif randomsig=="RANDOMIZED":
        if rand_mat_was_none:
            signature_full_dict, signature_keys_tolearn, rand_mat_list= get_R_sig(
                proj_dim, helper_path, rand_mat_list=None)
            signature_full_keys_str= copy.deepcopy(signature_keys_tolearn)
            with open(rand_mat_name+'.pickle', 'wb') as f:
                pickle.dump(rand_mat_list, f)
        else:
            signature_full_dict, signature_keys_tolearn= get_R_sig(
                proj_dim, helper_path, rand_mat_list=rand_mat_list)
            signature_full_keys_str= copy.deepcopy(signature_keys_tolearn)
            
        os.chdir(base_dir)
    if randomsig==False:
        if incl_MCAP:
            signature_keys_tolearn= create_word_list(n_stocks+2,order_sig)
        else:
            signature_keys_tolearn= create_word_list(n_stocks+1,order_sig)



    if incl_MCAP:
        raise NotImplementedError("MCAP version in currently not implemented")
    else:
        Q= initialize_Q_mem_optim(n_stocks, signature_full_keys_str, 
                                  integrator_path, signature_full_dict, 
                                  quad_var_index, t_start=t_start)
        
        c= initialize_c_Ito_mem_optim(n_stocks, signature_full_keys_str, 
                                      signature_full_dict, integrator_path, 
                                      quad_var_index, t_start=t_start)

   
    m= gurobipy.Model("mymodel")    
    n_param= len(signature_keys_tolearn)*n_stocks
    x= m.addMVar(n_param,lb=-bounds, ub= bounds, vtype=GRB.CONTINUOUS, 
                 name="x")
    
    perf_cv_l2=[]
    for l2_gamma in l2_gamma_list:
        val, ins_perf =optimization(x,m,Q,c,0,0, l2_gamma, n_param)  
        
        
        mkt_weights_cv= mkt_weights[:,t_insample:-(t_outsample)]
        caps_cv= caps[:,t_insample:-(t_outsample)]
        
        if normalize:
            norm_caps_cv= np.array([caps_cv[i,:]/caps_cv[i,0] 
                                    for i in range(len(caps_cv[:,0]))])
            norm_mkt_weights_cv= get_market_weights(norm_caps_cv)
        else:
            norm_mkt_weights_cv=None
        
        if incl_MCAP:
            total_cap_cv= total_cap[:,t_insample:-(t_outsample)]
        else:
            total_cap_cv= None
        sig_weights_cv, F_cv= calc_sig_portfolio_weights(
            val, mkt_weights_cv,order_sig, add_time=True, 
            timestep=1/(t_outsample+t_start), total_cap=total_cap_cv, 
            incl_MCAP=incl_MCAP, normalize=normalize, 
            mkt_weights_norm=norm_mkt_weights_cv,randomsig=randomsig, 
            n_jobs=n_jobs, proj_dim=proj_dim, rand_mat_list=rand_mat_list)
        
        sig_weights_cv=[ list(sig_weights_cv[i]) for i in range(n_stocks)]
        
        sig_cv_value_rel_log= relative_log_value(
            np.array(mkt_weights_cv), np.array(sig_weights_cv), 
            full_hist=False, t_start=t_start)

        perf_cv_l2.append(sig_cv_value_rel_log)


    ind_max = np.nanargmax(perf_cv_l2)
    l2_gamma_opt= l2_gamma_list[ind_max]

    _ = learn_sig_weights_REALDATA_TCgradient(
        data_name=data_name, order_sig=order_sig, t_insample=t_insample, 
        t_outsample=t_outsample, bounds=bounds, prop_cost=0, 
        rank_based=rank_based, objective= "Log-Utility", risk_factor=0, 
        l2_gamma=l2_gamma_opt, ranks=ranks, t_start=t_start, 
        incl_MCAP=incl_MCAP, lambda_TC=None, end_time=end_time, 
        base_model_dir=base_model_dir, normalize=normalize, 
        randomsig=randomsig, reuse_TC=None, proj_dim=proj_dim, n_jobs=n_jobs,
        rand_mat_name=rand_mat_name)

    return l2_gamma_opt, rand_mat_name
 


