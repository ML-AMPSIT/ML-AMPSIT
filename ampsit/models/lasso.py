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


def sa_lassoregression(X_train, X_test, y_train, y_test, importance_list,score_list,pvalue_list,mse_list,mae_list,ytest,ypred,f,tun):
  import numpy as np
  from sklearn.linear_model import LassoCV
  from joblib import dump, load
  
  if tun==1:
    
    from skopt import BayesSearchCV
    from sklearn.model_selection import cross_val_score
    
    lasso_cv = LassoCV()

    params = {
        'eps': (1e-8, 1e-1, 'log-uniform'),
        'n_alphas': (50, 300),
        'tol': (1e-8, 1e-3, 'log-uniform'),
        'cv': [5, 7],
    }

    opt = BayesSearchCV(lasso_cv, params, n_iter=tun_iter, cv=5, n_jobs=-1)

    opt.fit(X_train, y_train)

    score = cross_val_score(opt, X_train, y_train, cv=5).mean()

    if os.path.exists(f'{output_path}tuning_results_lasso_{f[:-4]}.txt'):
        os.remove(f'{output_path}tuning_results_lasso_{f[:-4]}.txt')
    with open(f'{output_path}tuning_results_lasso_{f[:-4]}.txt', 'a') as file:
        file.write("Best parameters: {}\n".format(opt.best_params_))
        file.write("Cross-validation score: {}\n".format(score))

    y_pred = opt.best_estimator_.predict(X_test)

    importances = np.abs(opt.best_estimator_.coef_)

    dump(opt, output_path+'lasso_model_'+f[:-4]+'.joblib')

  else:
  
    if tun==2:
      loaded_model = load(output_path+'lasso_model_'+f[:-4]+'.joblib')
      
      y_pred = loaded_model.predict(X_test)
      
      importances = np.abs(loaded_model.best_estimator_.coef_)
      
    else:
  
  
      lasso_cv = LassoCV(cv=5)
      
      lasso_cv.fit(X_train, y_train)                        

      y_pred = lasso_cv.predict(X_test)
      
      importances = np.abs(lasso_cv.coef_)
                      
  
       
  if len(importance_list) <= totalhours-1:
      #importance_list.append(importances)
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

