# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 14:50:32 2018

@author: Esteban Ortiz Tirado
"""
import pandas as pd
import numpy as np
from eotg import eotg
#%%
quotes = pd.read_csv('quotes.csv')
quotes = quotes.Symbol.dropna()
#%%
#names = ['ALFAA.MX', 'MEXCHEM.MX', 'GFNORTEO.MX', 'CEMEXCPO.MX', 'NAFTRACISHRS.MX','WALMEX.MX', 'LABB.MX', 'GMEXICOB.MX','GRUMAB.MX','LIVEPOLC-1.MX','BBAJIOO.MX']
names = list(quotes)
start, end= '01/01/2013', '09/20/2018'
#%%
closes = eotg.get_closes(tickers=names,start_date=start,end_date=end, freq='d')
#%%
import matplotlib.pyplot as plt
closes.plot(figsize=(12,8)).legend()
plt.xlabel('Date')
plt.ylabel('Price $')
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
for i in np.arange(len(names)):
    plt.text(x_points[i],y_points[i],names[i])



#%% Matriz de correlacion y de covarianza
cov_mat = daily_ret.cov()
corr_mat = daily_ret.corr()

#%% Creación del portafolio
W = np.array([0.3,0.3,0.2,0.2,0.0,0.0])
p_names = ['CSCO','FTNT','PANW','CYBR','HACK','CHKP']


#%%
Er_activos = annual_ret_summary[p_names].loc['Mean'].transpose()
Er_Port = Er_activos.dot(W)
Std_Port = np.sqrt(W.transpose().dot(cov_mat[p_names].loc[p_names]).dot(W))*np.sqrt(252)
#%%
port_ret_summary = pd.DataFrame(index=['Mean', 'Volatility'],columns=['Port'])
port_ret_summary.loc['Mean'] = Er_Port
port_ret_summary.loc['Volatility'] = Std_Port
port_ret_summary
#%%
# Gráfico rendimiento esperado vs. volatilidad
#x_points = annual_ret_summary.join(port_ret_summary).loc['Volatility'].values
#y_points = annual_ret_summary.join(port_ret_summary).loc['Mean'].values
x_points = annual_ret_summary.loc['Volatility'].values
y_points = annual_ret_summary.loc['Mean'].values
plt.figure(figsize=(8,6))
plt.plot(x_points,y_points,'ro',ms=5)
plt.plot(Std_Port,Er_Port,'bo',ms=8)
plt.xlabel('Volatility $\sigma$')
plt.ylabel('Expected Return $E[r]$')
#Etiqueta de cada instrumento
for i in np.arange(10):
    plt.text(x_points[i],y_points[i],names[i])
plt.text(Std_Port,Er_Port,'Port')