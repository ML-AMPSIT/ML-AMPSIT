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


def sa_xgboost(X_train, X_test, y_train, y_test, importance_list,score_list,pvalue_list,mse_list,mae_list,ytest,ypred,f,tun):
  from xgboost import XGBRFRegressor
  from joblib import dump, load

  
  if tun==1:

    from skopt import BayesSearchCV
    from sklearn.model_selection import cross_val_score

    gb = XGBRFRegressor()

    params = {
        'n_estimators': (10, 30),
        'max_depth': (2, 6),
    }
    '''
    from skopt.space import Real, Integer, Categorical
    params = {
        'n_estimators': Integer(50, 100),  # Numero di alberi
        'max_depth': Integer(3, 10),  # Profondità massima di ciascun albero
        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),  # Tasso di apprendimento
        'min_child_weight': Integer(1, 10),  # Minimo peso richiesto per un nodo
        'gamma': Real(0, 0.5),  # Minima riduzione della perdita necessaria per fare ulteriori partizioni
        'subsample': Real(0.5, 1.0),  # Frazione di esempi da usare per addestrare ciascun albero
        'colsample_bytree': Real(0.5, 1.0),  # Frazione di feature da usare per costruire ciascun albero
        'reg_lambda': Real(1e-2, 1e2, prior='log-uniform'),  # Regolarizzazione L2 sui pesi
        'reg_alpha': Real(1e-2, 1e2, prior='log-uniform'),  # Regolarizzazione L1 sui pesi
    }
    '''
    opt = BayesSearchCV(gb, params, n_iter=tun_iter, cv=5, n_jobs=-1)

    opt.fit(X_train, y_train)
    score = cross_val_score(opt, X_train, y_train, cv=5).mean()

    if os.path.exists(f'{output_path}tuning_results_xgb_{f[:-4]}.txt'):
        os.remove(f'{output_path}tuning_results_xgb_{f[:-4]}.txt')
    with open(f'{output_path}tuning_results_xgb_{f[:-4]}.txt', 'a') as file:
        file.write("Best parameters: {}\n".format(opt.best_params_))
        file.write("Cross-validation score: {}\n".format(score))

    best_params = opt.get_params()
    gb = XGBRFRegressor(**best_params)

    gb.fit(X_train, y_train)
  
    y_pred = gb.predict(X_test)
    
    importances = gb.feature_importances_

    dump(gb, output_path+'xgb_model_'+f[:-4]+'.joblib')
    
  else:

    if tun==2:
      loaded_model = load(output_path+'xgb_model_'+f[:-4]+'.joblib')
      
      y_pred = loaded_model.predict(X_test)
      
      importances = loaded_model.feature_importances_
      
    else:        
    
      gb = XGBRFRegressor(n_estimators=20, max_depth=5, max_features='log2',min_samples_leaf= 1, min_samples_split= 2)
    
      gb.fit(X_train, y_train)
    
      y_pred = gb.predict(X_test)

      importances = gb.feature_importances_

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
