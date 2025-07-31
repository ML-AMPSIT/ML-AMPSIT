#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
from scipy.stats import spearmanr


import json
from ampsit.config import load_config
config = load_config()

totalhours = config['totalhours']
variables = config['variables']
regions = config['regions']
verticalmax = config['verticalmax']
totalsim = config['totalsim']
parameter_names = config['parameter_names']
output_path = config['output_pathname']
tun_iter = config['tun_iter']

def sa_baesyanreg(X_train, X_test, y_train, y_test,problem,N, importance_list,score_list,pvalue_list,mse_list,mae_list,ytest,ypred,first_ord,f,tun):
  from sklearn.model_selection import train_test_split

  from sklearn.linear_model import BayesianRidge
  from joblib import dump, load


  if tun==1:
    
    from skopt import BayesSearchCV
    from sklearn.model_selection import cross_val_score
    
    br_model = BayesianRidge()

    params = {
        'max_iter': (100, 500),  
        'tol': (1e-9, 1e-3, 'log-uniform'),
        'alpha_1': (1e-10, 1e-4, 'log-uniform'),  
        'alpha_2': (1e-10, 1e-4, 'log-uniform'),  
        'lambda_1': (1e-10, 1e-4, 'log-uniform'),  
        'lambda_2': (1e-10, 1e-4, 'log-uniform'), 
        'fit_intercept': [True, False], 
    }


    br = BayesSearchCV(br_model, params, n_iter=tun_iter, cv=5, n_jobs=-1)

    br.fit(X_train, y_train)

    score = cross_val_score(br, X_train, y_train, cv=5).mean()
    
    if os.path.exists(f'{output_path}tuning_results_br_{f[:-4]}.txt'):
        os.remove(f'{output_path}tuning_results_br_{f[:-4]}.txt')
    with open(f'{output_path}tuning_results_br_{f[:-4]}.txt', 'a') as file:
        file.write("Best parameters: {}\n".format(br.best_params_))
        file.write("Cross-validation score: {}\n".format(score))
    
    dump(br, output_path+'br_model_'+f[:-4]+'.joblib')
    y_pred = br.predict(X_test)
    
  else:    
    if tun==2:
      br = load(output_path+'br_model_'+f[:-4]+'.joblib')
      y_pred = br.predict(X_test)
    else:
      br = BayesianRidge()
      br.fit(X_train, y_train)
      
      y_pred = br.predict(X_test)


  from SALib.sample import sobol
  X_sobol = sobol.sample(problem, N, calc_second_order=True)
  from numpy import zeros
  Y_sobol = zeros((len(X_sobol), 1))
  for i in range(len(X_sobol)):
      Y_sobol[i] = br.predict(X_sobol[i].reshape(1, -1))
  Y_sobol = Y_sobol.reshape(-1)
  from SALib.analyze import sobol
  Si = sobol.analyze(problem, Y_sobol, calc_second_order=True, print_to_console=False)

  importances = Si['ST']
  first_order = Si['S1']
  interactions=Si['S2']
  np.savetxt(output_path+f'interactions_matrix_{len(importance_list)}.txt', interactions, delimiter='\t', fmt='%f')
  
  
  if len(importance_list) <= totalhours-1:
      #importance_list.append(importances)
      sum_importances = np.sum(importances, axis=0)
      normalized_importances = importances / sum_importances       
      importance_list.append(normalized_importances)
      first_ord.append(first_order)
  else:
      #importance_list = []   
      #first_ord=[]            
      importance_list.clear()
      importance_list.append(importances)    
      first_ord.clear()
      first_ord.append(first_order)  


  from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

  spearman_corr, p_value = spearmanr(y_test, y_pred)
  score=spearman_corr
  pvalue= p_value  
  
  mse = mean_squared_error(y_test, y_pred)
  mae = mean_absolute_error(y_test, y_pred)
  
  if len(score_list) <= totalhours-1:
    score_list.append(score)
    pvalue_list.append(pvalue)
    mse_list.append(mse)
    mae_list.append(mae)
  else:
    #score_list=[]
    #pvalue_list=[]
    #mse_list=[]
    #mae_list=[]
    score_list.clear()
    pvalue_list.clear()
    mse_list.clear()
    mae_list.clear()
    score_list.append(score)
    pvalue_list.append(pvalue)
    mse_list.append(mse)
    mae_list.append(mae)

  if len(ytest)<=totalhours-1:
    ytest.append(y_test)
    ypred.append(y_pred)
  else:
    #ytest=[]
    #ypred=[]
    ytest.clear()
    ypred.clear()
    ytest.append(y_test)
    ypred.append(y_pred)

