#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
import numpy as np
import warnings
import numpy as np
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

from ampsit.models import (
    sa_randomforest, sa_lassoregression, sa_svm,
    sa_baesyanreg, sa_gaussianreg, sa_xgboost, sa_cart
)

from ampsit.utils import get_distinct_colors
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

importance_list = []
first_ord = []
score_list = []
pvalue_list = []
mse_list = []
mae_list = []
ypred = []
ytest = []

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

from ampsit.config import load_loop_config
config_data = load_loop_config()

hour=config_data['hour'] 
n_sample=config_data['Nsobol'] #mod nparam+2 == 0
        
tun = config_data['tun']      


def run_analysis_job(meth, N, var, vpoint, hpoint):

    importance_list=[]
    first_ord=[]
    score_list = []
    pvalue_list =[]
    mse_list = []
    mae_list = []
    ypred=[]
    ytest=[]

  
    try:
  
      if var >= 1 and var <= len(variables):
          nam1 = variables[var - 1]
      else:
          nam1 = 'Invalid var value'

      if hpoint >= 1 and hpoint <= len(regions):
          nam2 = regions[hpoint - 1]
      else:
          nam2 = 'Invalid hpoint value'        

      name=nam1+'_'+nam2+'_lev'+str(vpoint)
      file_list=[nam1+'_'+nam2+'_lev'+str(vpoint)+'_'+str(i)+'.txt' for i in range(1,totalhours+1)]
      
      Xnonscaled = np.loadtxt(output_path+'X.txt') 


      for file in file_list:

        f = file
        ynonscaled = np.loadtxt(output_path+file, delimiter=',')
        
        # Taglia i dati a N simulazioni
        Xnonscaled = Xnonscaled[:N, :]
        ynonscaled = ynonscaled[:N]

        # Verifica che abbiano la stessa dimensione
        min_len = min(len(Xnonscaled), len(ynonscaled))
        Xnonscaled = Xnonscaled[:min_len, :]
        ynonscaled = ynonscaled[:min_len]
        

        y=ynonscaled
        X=Xnonscaled            
        
        
        from sklearn.model_selection import train_test_split
        
        partition=0.3
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=partition, random_state=42)

        from sklearn.preprocessing import StandardScaler
        from joblib import dump

        ##########################
        ########

        scalerX = StandardScaler()
        X_train = scalerX.fit_transform(X_train)
        X_test = scalerX.transform(X_test)

        scalery = StandardScaler()
        y_train = scalery.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test = scalery.transform(y_test.reshape(-1, 1)).ravel()

        
        ''
        Xlow = np.min(X_train, axis=0)
        Xup = np.max(X_train, axis=0)
        Nn = n_sample 
        bounds = [(Xlow[i], Xup[i]) for i in range(Xlow.shape[0])]
        problem = {'num_vars': X.shape[1], 'names': parameter_names, 'bounds': bounds}            
        ''
        
        
        if meth==1:
          Si = sa_randomforest(X_train, X_test, y_train, y_test, importance_list,score_list,pvalue_list,mse_list,mae_list,ytest,ypred,f,tun)
          method='randomforest'
        elif meth==2:
          Si = sa_lassoregression(X_train, X_test, y_train, y_test, importance_list,score_list,pvalue_list,mse_list,mae_list,ytest,ypred,f,tun)
          method='lasso'
        elif meth==3:
          Si = sa_svm(X_train, X_test, y_train, y_test,problem,Nn, importance_list,score_list,pvalue_list,mse_list,mae_list,ytest,ypred,f,tun)
          method='svm'
        elif meth==4:        
          Si = sa_baesyanreg(X_train, X_test, y_train, y_test,problem,Nn, importance_list,score_list,pvalue_list,mse_list,mae_list,ytest,ypred,first_ord,f,tun)
          method='br'
        elif meth==5:
          Si = sa_gaussianreg(X_train, X_test, y_train, y_test,problem,Nn, importance_list,score_list,pvalue_list,mse_list,mae_list,ytest,ypred,first_ord,f,tun)
          method='gp'
        elif meth==6:
          Si = sa_xgboost(X_train, X_test, y_train, y_test, importance_list,score_list,pvalue_list,mse_list,mae_list,ytest,ypred,f,tun)
          method='xgboost'              
        elif meth==7:
          Si = sa_cart(X_train, X_test, y_train, y_test, importance_list,score_list,pvalue_list,mse_list,mae_list,ytest,ypred,f,tun)
          method='cart'
          
                      
        data = np.array([score_list, pvalue_list, mse_list, mae_list]).T
        import pandas as pd
        df = pd.DataFrame(data, columns=['score', 'pvalue' , 'mse', 'mae'])
        importance_df = pd.DataFrame(importance_list)
        if hour==totalhours:
          importance_df.to_csv(output_path+'importance'+method+str(N)+file[:-7]+'.txt', header=False, index=False, sep=' ')
          df.to_csv(output_path+'df'+method+str(N)+file[:-7]+'.txt', header=False, index=False, sep=' ')
          yt=np.array(ytest)
          yp=np.array(ypred)
          
      ''
      
      #print(f"[PID {os.getpid()}] Start: meth={meth}, N={N}, var={var}, vpoint={vpoint}, hpoint={hpoint}")
      
      #############################################################################################
      import matplotlib
      matplotlib.use('Agg') 
      import matplotlib.pyplot as plt

      matplotlib.rc('xtick',labelsize=10)
      matplotlib.rc('ytick',labelsize=10)
      
      num_colors = len(parameter_names)
      cmap = plt.get_cmap('tab20', num_colors)
      x = np.arange(totalhours - 1)

      pvalue = df.loc[hour-1, 'pvalue']
      if pvalue < 0.0001:
          pvalue_str = "<0.0001"
      else:
          pvalue_str = "{:.3g}".format(pvalue)

      #print("Len importance_list:", len(importance_list))
      #print("Len importance_list[0]:", len(importance_list[0]) if importance_list else "vuota")
      
      if meth in [4,5]:
      
          fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))

          x = np.arange(totalhours)
          from matplotlib.cm import get_cmap
          cmap = get_cmap('viridis')
          #colors = [cmap(i / (len(parameter_names)-1)) for i in range(len(parameter_names))]
          colors = get_distinct_colors(len(parameter_names))
          metric_colors = {
          'score': '#1f77b4',   # blu
          'mse':   '#ff7f0e',   # arancio
          'mae': '#2ca02c',   # verde
          'pvalue': '#d62728'    # rosso
          }
          #colors = ['b','g','r','c','m','y']
          #colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#999999']
          for i in range(len(parameter_names)):     
            axs[0,0].plot(x,[abs(importance_list[j][i]) for j in range(totalhours)],label=parameter_names[i],color=colors[i])    
          
          axs[0,0].set_xlabel("Time",fontsize=10)
          axs[0,0].set_ylabel("Importance",fontsize=10)
          axs[0,0].set_title('Importance in time',fontsize=10)
          axs[0,0].legend(prop={'size':4,'weight':'normal'})

          ln1 = axs[0,1].plot(df.index, df['score'], color=metric_colors['score'], label='Score')[0]
          ln2 = axs[0,1].plot(df.index, df['mse'],  color=metric_colors['mse'],   label='MSE')[0]
          ln3 = axs[0,1].plot(df.index, df['mae'],  color=metric_colors['mae'],   label='MAE')[0]
          twin=axs[0,1].twinx()
          ln4 = twin.plot(df.index, df['pvalue'],   color=metric_colors['pvalue'],label='p-value')[0]
          twin.set_ylabel('p-value',fontsize=12)
          twin.set_ylim(0, 1e-15)
          axs[0,1].set_xlabel('Time',fontsize=12)
          axs[0,1].set_ylabel('Value',fontsize=12)
          axs[0,1].set_title('Metrics time evolution',fontsize=12)
          axs[0,1].set_ylim(0, 1)
          lns = [ln1, ln2, ln3, ln4]
          labels = [l.get_label() for l in lns]
          axs[0,1].legend(lns, labels, prop={'size':10, 'weight':'normal'}, loc='upper center', bbox_to_anchor=(0.15, 0.8))
          axs[0,1].grid(True)

          axs[1,0].scatter(ytest[hour-1], ypred[hour-1])
          ideal = [min(ytest[hour-1]), max(ytest[hour-1])]
          axs[1,0].plot(ideal, ideal, 'r--')
          axs[1,0].set_xlabel('True Values')
          axs[1,0].set_ylabel('Predictions')
          axs[1,0].set_title('Hour: {}; Performance (score: {:.2f} [p: {}], mse: {:.2f}, mae: {:.2f})'.format(hour, df.loc[hour-1, 'score'], pvalue_str, df.loc[hour-1, 'mse'], df.loc[hour-1, 'mae']))

          axs[1,1].bar(parameter_names,importance_list[hour-1])
          axs[1,1].set_xticklabels(parameter_names, rotation=90, ha='right')
          axs[1,1].set_xlabel("Features")
          axs[1,1].set_ylabel("Coefficient")
          axs[1,1].set_title('Hour: '+str(hour)+'; Sobol total index')

          import seaborn as sns
          file_path = os.path.join(output_path, f'interactions_matrix_{hour-1}.txt')
          inter_mat = np.loadtxt(file_path, delimiter='\t')
          sns.heatmap(inter_mat, annot=True, cmap='coolwarm', fmt=".3f", cbar_kws={'label': 'Sobol second order index'},
                      xticklabels=parameter_names, yticklabels=parameter_names, ax=axs[2, 0])
          axs[2, 0].set_title(f'Interactions Matrix {hour}')
          
          
          axs[2,1].bar(parameter_names,first_ord[hour-1])
          
          axs[2,1].set_xlabel("Features")
          axs[2,1].set_ylabel("Coefficient")
          axs[2,1].set_title('Hour: '+str(hour)+'; Sobol first order index')    
          axs[2,1].set_xticklabels(parameter_names, rotation=90, ha='right')

      else:

          fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

          x = np.arange(totalhours)
          #colors = [cmap(i / (len(parameter_names)-1)) for i in range(len(parameter_names))]
          colors = get_distinct_colors(len(parameter_names))
          metric_colors = {
          'score': '#1f77b4',   # blu
          'mse':   '#ff7f0e',   # arancio
          'mae': '#2ca02c',   # verde
          'pvalue': '#d62728'    # rosso
          }
          #colors 
          for i in range(len(parameter_names)):    
            axs[0,0].plot(x,[abs(importance_list[j][i]) for j in range(totalhours)],label=parameter_names[i],color=colors[i])  


          axs[0,0].set_xlabel("Time",fontsize=10)
          axs[0,0].set_ylabel("Importance",fontsize=10)
          axs[0,0].set_title('Importance in time',fontsize=10)
          axs[0,0].legend(prop={'size':4,'weight':'normal'})

          ln1 = axs[0,1].plot(df.index, df['score'], color=metric_colors['score'], label='Score')[0]
          ln2 = axs[0,1].plot(df.index, df['mse'],  color=metric_colors['mse'],   label='MSE')[0]
          ln3 = axs[0,1].plot(df.index, df['mae'],  color=metric_colors['mae'],   label='MAE')[0]
          twin=axs[0,1].twinx()
          ln4 = twin.plot(df.index, df['pvalue'],   color=metric_colors['pvalue'],label='p-value')[0]
          twin.set_ylabel('p-value',fontsize=12)
          twin.set_ylim(0, 1e-15)
          axs[0,1].set_xlabel('Time',fontsize=12)
          axs[0,1].set_ylabel('Value',fontsize=12)
          axs[0,1].set_title('Metrics time evolution',fontsize=12)
          axs[0,1].set_ylim(0, 1)
          lns = [ln1, ln2, ln3, ln4]
          labels = [l.get_label() for l in lns]
          axs[0,1].legend(lns, labels, prop={'size':10, 'weight':'normal'}, loc='upper center', bbox_to_anchor=(0.15, 0.8))
          axs[0,1].grid(True)

          axs[1,0].scatter(ytest[hour-1], ypred[hour-1])
          ideal = [min(ytest[hour-1]), max(ytest[hour-1])]
          axs[1,0].plot(ideal, ideal, 'r--')
          axs[1,0].set_xlabel('True Values')
          axs[1,0].set_ylabel('Predictions')
          axs[1,0].set_title('Hour: {}; Performance (score: {:.2f} [p: {}], mse: {:.2f}, mae: {:.2f})'.format(hour-1, df.loc[hour-1, 'score'], pvalue_str, df.loc[hour-1, 'mse'], df.loc[hour-1, 'mae']))

          axs[1,1].bar(parameter_names,abs(importance_list[hour-1]))
          axs[1,1].set_xticklabels(parameter_names, rotation=90, ha='right')
          axs[1,1].set_xlabel("Features")
          axs[1,1].set_ylabel("Coefficient")
          axs[1,1].set_title('Hour: '+str(hour-1)+'; Feature importance')

      fig.suptitle('Risultati di ' + method + ' su ' + name[:-9] + ' N:'+ str(N) + ' v:' + str(vpoint), fontsize=16)

      plt.tight_layout()
      fig.canvas.draw()
      output_file = output_path + method + '_' + name[:-9] + '_hour' + str(hour-1) + '_v' + str(vpoint) + 'N' + str(N) + '.png'
      #output_file = f"{output_path}_{method}_{name[:-9]}_hour{hour-1}_v{vpoint}_N{N}_pid{os.getpid()}.png"
      plt.savefig(output_file, dpi=300)
      #print("Salvato plot in:", output_file) 
      ''
      plt.close()
      
      importance_list=[]
      first_ord=[]
      score_list = []
      pvalue_list =[]
      mse_list = []
      mae_list = []
      ypred=[]
      ytest=[]

      print(f"Eseguito: meth={meth}, N={N}, var={var}, vpoint={vpoint}, hpoint={hpoint}")
      
    except Exception as e:
        import traceback
        print(f"Errore nel processo (meth={meth}, N={N}, var={var}, vpoint={vpoint}, hpoint={hpoint}):")
        traceback.print_exc()

    return f"{method}-{name}"

