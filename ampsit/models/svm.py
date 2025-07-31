#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

def sa_svm(X_train, X_test, y_train, y_test, problem, N, importance_list,score_list,pvalue_list,mse_list,mae_list,ytest,ypred,f,tun):
  from sklearn import svm
  from joblib import dump, load
  from sklearn.model_selection import train_test_split, cross_val_score 
  
  
  if tun==1:
    
    from skopt import BayesSearchCV
    from sklearn.model_selection import cross_val_score
    
    svm_model = svm.SVR()
    
    #params = {
    #    'C': (1e-3, 1e+1, 'log-uniform'),
    #    'kernel': ['linear'],
    #    'gamma': (1e-3, 1e+1, 'log-uniform'),
    #    'epsilon': (1e-4, 1e-1, 'log-uniform'), 
    #}
    
    from skopt.space import Real, Categorical, Integer
    params = {
        'C': Real(1e-4, 1e+1, prior='log-uniform'), 
        'kernel': Categorical(['poly']),
        'gamma': Real(1e-3, 1e+1, prior='log-uniform'),
        'epsilon': Real(1e-4, 1e-1, prior='log-uniform'),
        'degree': Integer(2, 6),
        'coef0': Real(0, 10),
    }


    svr = BayesSearchCV(svm_model, params, n_iter=tun_iter, cv=5, n_jobs=-1)

    svr.fit(X_train, y_train)

    score = cross_val_score(svr, X_train, y_train, cv=5).mean()

    if os.path.exists(f'{output_path}tuning_results_svm_{f[:-4]}.txt'):
        os.remove(f'{output_path}tuning_results_svm_{f[:-4]}.txt')
    with open(f'{output_path}tuning_results_svm_{f[:-4]}.txt', 'a') as file:
        file.write("Best parameters: {}\n".format(svr.best_params_))
        file.write("Cross-validation score: {}\n".format(score))

    y_pred = svr.predict(X_test)
    #coef = svr.best_estimator_.coef_[0]  #classical linear
    
    #import shap
    #best_svm_model = svr.best_estimator_
    #explainer = shap.Explainer(best_svm_model.predict, X_train)
    #shap_values = explainer(X_test)
    #coef = np.abs(shap_values.values).mean(axis=0) #shap
    
    dump(svr, output_path+'svm_model_'+f[:-4]+'.joblib')
  
  else:
 
    if tun==2:
      svr = load(output_path+'svm_model_'+f[:-4]+'.joblib')
      
      y_pred = svr.predict(X_test)
      
      #coef=loaded_model.best_estimator_.coef_[0] #classical linear
      
    else:
      
      svr = svm.SVR(kernel='linear', C=1, epsilon=0.1)
      svr.fit(X_train, y_train)
      
      y_pred = svr.predict(X_test)
      
      #coef=clf.coef_[0] #classical linear
  
  
  from SALib.sample import sobol
  X_sobol = sobol.sample(problem, N, calc_second_order=True)
  from numpy import zeros
  Y_sobol = zeros((len(X_sobol), 1))
  for i in range(len(X_sobol)):
      Y_sobol[i] =svr.predict(X_sobol[i].reshape(1, -1))
  Y_sobol = Y_sobol.reshape(-1)
  from SALib.analyze import sobol
  Si = sobol.analyze(problem, Y_sobol, calc_second_order=True, print_to_console=False)
  importances = np.where(Si['ST'] < 1, Si['ST'], np.nan) #sobol
  
  
  if len(importance_list) <= totalhours-1:
      #importance_list.append(coef) #classical linear / shap
      #importance_list.append(importances) #sobol
      sum_importances = np.sum(importances, axis=0)
      normalized_importances = importances / sum_importances       
      importance_list.append(normalized_importances)
  else:
      #importance_list = []               
      importance_list.clear()
      importance_list.append(importances)       

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

