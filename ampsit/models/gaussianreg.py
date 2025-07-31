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

def sa_gaussianreg(X_train, X_test, y_train, y_test,problem,N, importance_list,score_list,pvalue_list,mse_list,mae_list,ytest,ypred,first_ord,f,tun):

  from sklearn.gaussian_process import GaussianProcessRegressor
  from sklearn.gaussian_process.kernels import RBF
  from sklearn.gaussian_process.kernels import Matern
  from joblib import dump, load
  
  
  if tun==1:
    
    from skopt import BayesSearchCV
    from sklearn.model_selection import cross_val_score  
    from skopt.space import Real, Categorical, Integer
    from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, Sum, Product
        
    #kernel = RBF()    
    #kernel = Matern()
    kernel = RationalQuadratic()
    #kernel = Sum(Matern(), RationalQuadratic())
    #kernel = Product(Matern(), RationalQuadratic())
    gp_model = GaussianProcessRegressor(kernel=kernel)
    
    '''
    ##RBF
    #params = {
    #'alpha': (1e-10, 1e+3, 'log-uniform'),  
    #'n_restarts_optimizer': (0, 20),
    #'kernel__length_scale': (1e-2, 1e+2, 'log-uniform'),
    #}
    params = {
        'alpha': Real(1e-8, 1e+3, prior='log-uniform'),
        'n_restarts_optimizer': Integer(0, 20),
        'kernel__length_scale': Real(1e-2, 1e+2, prior='log-uniform'),
    }    
    '''
    ''
    ##rational quadratic
    params = {
    'alpha': (1e-10, 1e+3, 'log-uniform'),  
    'n_restarts_optimizer': (0, 20),
    'kernel__length_scale': (1e-2, 1e+2, 'log-uniform'),
    'kernel__alpha': Real(1e-2, 1e+2, prior='log-uniform'),
    }
    '''
    ''
    ##matern
    params = {
    'alpha': (1e-10, 1e+3, 'log-uniform'),  
    'n_restarts_optimizer': (0, 20),
    'kernel__length_scale': (1e-2, 1e+2, 'log-uniform'),
    'kernel__nu': Real(1e-2, 1e+2, prior='log-uniform'),
    }    
    '''

    ##product/sum
    #params = {
    #'alpha': Real(1e-10, 1e+3, prior='log-uniform'),
    #'n_restarts_optimizer': Integer(0, 20),
    #'kernel__k1__length_scale': Real(1e-2, 1e+2, prior='log-uniform'),  # Per Matern nel kernel Sum/Product
    #'kernel__k2__length_scale': Real(1e-2, 1e+2, prior='log-uniform'),  # Per RationalQuadratic nel kernel Sum/Product
    #'kernel__k1__nu': Real(0.5, 2.5),  # Parametro aggiuntivo per Matern
    #'kernel__k2__alpha': Real(0.1, 10),  # Parametro aggiuntivo per RationalQuadratic
    #}
    
    gp = BayesSearchCV(gp_model, params, n_iter=tun_iter, cv=5,n_jobs=-1)
    gp.fit(X_train, y_train)

    score = cross_val_score(gp, X_train, y_train, cv=5).mean()

    if os.path.exists(f'{output_path}tuning_results_gp_{f[:-4]}.txt'):
        os.remove(f'{output_path}tuning_results_gp_{f[:-4]}.txt')
    with open(f'{output_path}tuning_results_gp_{f[:-4]}.txt', 'a') as file:
        file.write("Best parameters: {}\n".format(gp.best_params_))
        file.write("Cross-validation score: {}\n".format(score))
    
    dump(gp, output_path+'gp_model_'+f[:-4]+'.joblib')
    
    y_pred = gp.predict(X_test)
        
  else:  
    
    if tun==2:
      gp = load(output_path+'gp_model_'+f[:-4]+'.joblib')
      y_pred = gp.predict(X_test)
    else:
      kernel = RBF(length_scale=1.0) ##########################################################
      gp = GaussianProcessRegressor(kernel=kernel)
      gp.fit(X_train, y_train)
      
      y_pred = gp.predict(X_test)
  
  
  from SALib.sample import sobol
  X_sobol = sobol.sample(problem, N, calc_second_order=True)
  from numpy import zeros
  Y_sobol = zeros((len(X_sobol), 1))
  for i in range(len(X_sobol)):
      Y_sobol[i] = gp.predict(X_sobol[i].reshape(1, -1))
  Y_sobol = Y_sobol.reshape(-1)
  from SALib.analyze import sobol
  Si = sobol.analyze(problem, Y_sobol, calc_second_order=True, print_to_console=False)

  importances = np.where(Si['ST'] < 1, Si['ST'], np.nan)
  first_order=Si['S1']
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
      

  from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
  
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

