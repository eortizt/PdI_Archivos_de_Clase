# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 16:49:32 2018

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
start= '10/01/2016'
#%%
closes = eotg.get_closes(tickers=names,start_date=start, freq='d')
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
W = np.array([1/7,1/7,1/7,1/7,1/7,1/7,1/7])
p_names = ['AC.MX','MEXCHEM.MX','ALFAA.MX','WALMEX.MX','GFNORTEO.MX','BBAJIOO.MX','CEMEXCPO.MX']


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




#%% para tres activos, obtengamos la frontera de mínima varianza
## Construcción de parámetros
import scipy.optimize as opt
# 1. Sigma: matriz de varianza-covarianza
#       cov_mat
Sigma = cov_mat[p_names].loc[p_names]

#%%
# 2. Eind: rendimientos esperados activos individuales
#       es un vector con los rendimientos esperados individuales   Eind = np.array([E1, E2, E3])
E1 = annual_ret_summary[p_names[0]].loc['Mean']
E2 = annual_ret_summary[p_names[1]].loc['Mean']
E3 = annual_ret_summary[p_names[2]].loc['Mean']
E4 = annual_ret_summary[p_names[3]].loc['Mean']
E5 = annual_ret_summary[p_names[4]].loc['Mean']
E6 = annual_ret_summary[p_names[5]].loc['Mean']
E7 = annual_ret_summary[p_names[6]].loc['Mean']
    
Eind = np.array([E1, E2, E3, E4, E5, E6, E7])
#%%
# 3. Ereq: rendimientos requeridos para el portafolio
#       # Número de portafolios
N = 200
Ereq = np.linspace(Eind.min(), Eind.max(), N)

def varianza(w, Sigma):
    return w.dot(Sigma).dot(w)
def rendimiento_req(w, Eind, Ereq):
    return Eind.dot(w)-Ereq


# Dato inicial
w0 = np.zeros(7,)
# Cotas de las variables
bnds = ((0,None), (0,None), (0,None), (0,None), (0,None), (0,None), (0,None)) # todas tienen que ser positivas



# DataFrame de portafolios de la frontera
portfolios = pd.DataFrame(index=range(N), columns=['w1', 'w2', 'w3','w4', 'w5', 'w6','w7', 'Ret', 'Vol'])

# Construcción de los N portafolios de la frontera
for i in range(N):
    # Restricciones
    cons = ({'type': 'eq', 'fun': rendimiento_req, 'args': (Eind,Ereq[i])},
            {'type': 'eq', 'fun': lambda w: np.sum(w)-1})
    # Portafolio de mínima varianza para nivel de rendimiento esperado Ereq[i]
    min_var = opt.minimize(varianza, w0, args=(Sigma,), bounds=bnds, constraints=cons)
    # Pesos, rendimientos y volatilidades de los portafolio
    portfolios.loc[i,['w1','w2','w3','w4', 'w5', 'w6','w7']] = min_var.x
    portfolios['Ret'][i] = rendimiento_req(min_var.x, Eind, Ereq[i])+Ereq[i]
    portfolios['Vol'][i] = np.sqrt(varianza(min_var.x, Sigma))

portfolios = portfolios.drop([199])

#%%
plt.figure(figsize=(8,6))
#plt.scatter(annual_ret_summary[p_names].loc['Volatility'],annual_ret_summary[p_names].loc['Mean'])
plt.plot(portfolios.Vol, portfolios.Ret, 'k-', lw=4, label='Portafolios')
plt.xlabel('Volatilidad ($\sigma$)')
plt.ylabel('Rendimiento esperado ($E[r]$)')
plt.grid()
plt.legend(loc='best')

#%% Portafolio de mínima varianza 
cons = ({'type': 'eq', 'fun': lambda w: np.sum(w)-1})
min_var = opt.minimize(varianza, w0, args=(Sigma,), bounds=bnds, constraints=cons)

#%%
plt.figure(figsize=(8,6))
#plt.scatter(annual_ret_summary[p_names].loc['Volatility'],annual_ret_summary[p_names].loc['Mean'])
plt.plot(portfolios.Vol, portfolios.Ret, 'k-', lw=4, label='Portafolios')
plt.plot(np.sqrt(min_var.fun),min_var.x.dot(Eind),'mo',ms=5,label='Port Min Var 3')
plt.xlabel('Volatilidad ($\sigma$)')
plt.ylabel('Rendimiento esperado ($E[r]$)')
plt.grid()
plt.legend(loc='best')

