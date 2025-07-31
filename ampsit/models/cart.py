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


def sa_cart(X_train, X_test, y_train, y_test, importance_list,score_list,pvalue_list,mse_list,mae_list,ytest,ypred,f,tun):
  from sklearn.tree import DecisionTreeRegressor
  
  from joblib import dump, load
  
  if tun==1:

    from skopt import BayesSearchCV
    from sklearn.model_selection import cross_val_score

    dt = DecisionTreeRegressor()

    params = {
        'max_depth': (2, 6),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 5)
    }

    opt = BayesSearchCV(dt, params, n_iter=tun_iter, cv=5, n_jobs=-1)

    opt.fit(X_train, y_train)
    score = cross_val_score(opt, X_train, y_train, cv=5).mean()

    if os.path.exists(f'{output_path}tuning_results_cart_{f[:-4]}.txt'):
        os.remove(f'{output_path}tuning_results_cart_{f[:-4]}.txt')
    with open(f'{output_path}tuning_results_cart_{f[:-4]}.txt', 'a') as file:
        file.write("Best parameters: {}\n".format(opt.best_params_))
        file.write("Cross-validation score: {}\n".format(score))

    best_params = opt.get_params()
    
    dt_params = {key: value for key, value in best_params.items() if key in ['max_depth', 'min_samples_split', 'min_samples_leaf']}

    dt = DecisionTreeRegressor(**dt_params)

    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)

    importances = dt.feature_importances_

    dump(dt, output_path+'cart_model_'+f[:-4]+'.joblib')
    
  else:
    
    if tun==2:
      loaded_model = load(output_path+'cart_model_'+f[:-4]+'.joblib')
      
      y_pred = loaded_model.predict(X_test)
      
      importances = loaded_model.feature_importances_
      
    else:    
    
      dt = DecisionTreeRegressor(max_depth=5,min_samples_leaf= 1, min_samples_split= 2)
    
      dt.fit(X_train, y_train)

      y_pred = dt.predict(X_test)

      importances = dt.feature_importances_


  if len(importance_list) <= totalhours-1:
      importance_list.append(importances)
  else:
      #importance_list = []               
      importance_list.clear()
      importance_list.append(importances)           
  
  spearman_corr, p_value = spearmanr(y_test, y_pred)
  score=spearman_corr
  pvalue= p_value
  
  from sklearn.metrics import mean_squared_error
  mse = mean_squared_error(y_test, y_pred)
  from sklearn.metrics import mean_absolute_error
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



