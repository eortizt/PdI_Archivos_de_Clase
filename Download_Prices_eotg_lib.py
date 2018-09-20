# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 14:50:32 2018

@author: Esteban Ortiz Tirado
"""
import pandas as pd
import numpy as np
from eotg import eotg

names = ['ALFAA.MX', 'MEXCHEM.MX', 'GFNORTEO.MX', 'CEMEXCPO.MX', 'NAFTRACISHRS.MX','WALMEX.MX', 'LABB.MX', 'GMEXICOB.MX','GRUMAB.MX','LIVEPOLC-1.MX']
start, end= '01/01/2015', '09/18/2018'
#%%
closes = eotg.get_closes(tickers=names,start_date=start,end_date=end, freq='d')
#%%
import matplotlib.pyplot as plt
closes.plot(figsize=(12,8)).legend()
#plt.plot(closes)
#closes.plot().legend()
plt.xlabel('Date')
plt.ylabel('MXN')
plt.show()
#%%
daily_ret = closes.pct_change().dropna()
#%%
annual_ret = eotg.calc_annual_ret(daily_ret)
media_diaria = daily_ret.mean()
desv_est = daily_ret.std()
annual_ret_summary = pd.DataFrame(index=['Mean', 'Volatility'],columns=names)*252
annual_ret_summary.loc['Mean']=media_diaria*252
annual_ret_summary.loc['Volatility']=desv_est*np.sqrt(252)
annual_ret_summary
#%%
# Gráfico rendimiento esperado vs. volatilidad
x_points = annual_ret_summary.loc['Volatility'].values
y_points = annual_ret_summary.loc['Mean'].values
plt.figure(figsize=(8,6))
plt.plot(x_points,y_points,'ro',ms=5)
plt.xlabel('Volatility $\sigma$')
plt.ylabel('Expected Return $E[r]$')
#Etiqueta de cada instrumento
for i in np.arange(10):
    plt.text(x_points[i],y_points[i],names[i])



#%% Matriz de correlacion y de covarianza
cov_mat = daily_ret.cov()
corr_mat = daily_ret.corr()

#%%
W = [0.2,0.2,0.2,0.2,0.2]
p_names = ['GMEXICOB.MX','MEXCHEM.MX','GRUMAB.MX','WALMEX.MX','GFNORTEO.MX']
#daily_ret['Port'] = np.sum(daily_ret[p_names]*W)
#Port = np.sum(np.sum(daily_ret[p_names]*W))

Port = daily_ret['GMEXICOB.MX']*0.2+daily_ret['MEXCHEM.MX']*0.2+daily_ret['GRUMAB.MX']*0.2+daily_ret['WALMEX.MX']*0.2+daily_ret['GFNORTEO.MX']*0.2
port_ret_summary = pd.DataFrame(index=['Mean', 'Volatility'],columns=['Port'])
port_ret_summary.loc['Mean'] = Port.mean()*252
port_ret_summary.loc['Volatility'] = Port.std()*np.sqrt(252)

#%%
# Gráfico rendimiento esperado vs. volatilidad
x_points = annual_ret_summary.join(port_ret_summary).loc['Volatility'].values
y_points = annual_ret_summary.join(port_ret_summary).loc['Mean'].values
plt.figure(figsize=(8,6))
plt.plot(x_points,y_points,'ro',ms=5)
plt.xlabel('Volatility $\sigma$')
plt.ylabel('Expected Return $E[r]$')
#Etiqueta de cada instrumento
for i in np.arange(10):
    plt.text(x_points[i],y_points[i],names[i])
