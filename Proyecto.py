# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 15:54:32 2018

@author: Esteban Ortiz Tirado
"""

import pandas as pd
import numpy as np
from eotg import eotg
import matplotlib.pyplot as plt

#%% Descarga de datos

#quotes = pd.read_csv('quotes.csv')
#quotes = quotes.Symbol.dropna()

names = ['LULU', 'AAPL', 'AMZN','TWTR','JPM','HACK','V']
#names = list(quotes)
start= '10/01/2015'

closes = eotg.get_closes(tickers=names,start_date=start, freq='d')

daily_ret = closes.pct_change().dropna()

annual_ret = eotg.calc_annual_ret(daily_ret)
media_diaria = daily_ret.mean()
desv_est = daily_ret.std()
annual_ret_summ = pd.DataFrame(index=['Media', 'Volatilidad'],columns=names)*252
annual_ret_summ.loc['Media']=media_diaria*252
annual_ret_summ.loc['Volatilidad']=desv_est*np.sqrt(252)
annual_ret_summ
#%%
# Tasa libre de riesgo
rf = 0.07

#%%
import random
nube = pd.DataFrame(index=np.arange(5001),columns=['Media', 'Volatilidad'])
Eind = np.array(annual_ret_summ.loc['Media'])
Sigma = daily_ret.cov()
for c in np.arange(5001):
    r = [random.random() for i in range(1,len(names)+1)]
    s = sum(r)
    r = [ i/s for i in r ]
    nube.Media.loc[c] = Eind.dot(np.array(r))
    nube.Volatilidad.loc[c] = np.array(r).dot(Sigma).dot(np.array(r))

plt.scatter(nube.Volatilidad,nube.Media)


#%%
# Encontrar portafolio de mínima varianza
# Importamos minimize de optimize
from scipy.optimize import minimize

## Construcción de parámetros
# 1. Sigma: matriz de varianza-covarianza
#D = np.diag(annual_ret_summ.loc['Volatilidad'])
#Sigma = D.dot(corr).dot(D)

Sigma = daily_ret.cov()

# 2. Eind: rendimientos esperados activos individuales
Eind = np.array(annual_ret_summ.loc['Media'])

# Función objetivo
def varianza(w, Sigma):
    return w.dot(Sigma).dot(w)

# Cantidad de activos
#n = 5
n = len(names)

# Dato inicial
w0 = np.ones((n,1))/n
# Cotas de las variables
bnds = ((0,1),)*n
# Restricciones
cons = ({'type': 'eq','fun': lambda w: np.sum(w)-1},)

# Portafolio de mínima varianza
minvar = minimize(varianza, w0, args = (Sigma,), bounds=bnds, constraints=cons)
minvar

# Pesos, rendimiento y riesgo del portafolio de mínima varianza
w_minvar = minvar.x
Er_minvar = Eind.dot(w_minvar)
Std_minvar = np.sqrt(minvar.fun)


#%% Encontrar portafolio EMV
# Función objetivo
def Sharpe_ratio(w, Sigma, rf, Eind):
    Erp = Eind.dot(w)
    varp = w.dot(Sigma).dot(w)
    return -(Erp-rf)/np.sqrt(varp)

# Dato inicial
w0 = np.ones((n,1))/n
# Cotas de las variables
bnds = ((0,1),)*n
# Restricciones
cons = ({'type': 'eq','fun': lambda w: np.sum(w)-1},)

# Portafolio EMV
EMV = minimize(Sharpe_ratio, w0, args=(Sigma,rf,Eind), bounds=bnds, constraints= cons)
EMV

# Pesos, rendimiento y riesgo del portafolio EMV
w_EMV = EMV.x
Er_EMV = Eind.dot(w_EMV)
Std_EMV = np.sqrt(w_EMV.dot(Sigma).dot(w_EMV))

#%% Construir frontera de mínima varianza
# Covarianza entre los portafolios
cov_p = w_minvar.dot(Sigma).dot(w_EMV)

# Correlación entre los portafolios
corr_p = cov_p/(Std_minvar*Std_EMV)

# Vector de w
w = np.linspace(-5,10,500)

# DataFrame de portafolios: 
# 1. Índice: i
# 2. Columnas 1-2: w, 1-w
# 3. Columnas 3-4: E[r], sigma
# 4. Columna 5: Sharpe ratio
frontera = pd.DataFrame(columns=['w_EMV','w_Minvar','Rend','Vol','SR'])
frontera.w_EMV = w
frontera.w_Minvar = 1-w
frontera.Rend = w*Er_EMV+(1-w)*Er_minvar
frontera.Vol = np.sqrt((w*Std_EMV)**2+((1-w)*Std_minvar)**2+2*w*(1-w)*cov_p)
frontera.SR = (frontera.Rend-rf)/frontera.Vol

# Importar librerías de gráficos

# Gráfica de dispersión de puntos coloreando 
# de acuerdo a SR
x_points = annual_ret_summ.loc['Volatilidad'].values
y_points = annual_ret_summ.loc['Media'].values
plt.figure(figsize=(10,7))
plt.scatter(frontera.Vol,frontera.Rend,c=frontera.SR,cmap='RdYlBu')
#plt.plot(annual_ret_summ.loc['Volatilidad'],annual_ret_summ.loc['Media'],'og')
plt.plot(x_points,y_points,'og')
#Etiqueta de cada instrumento
for i in np.arange(len(names)):
    plt.text(x_points[i],y_points[i],names[i])
plt.plot(Std_EMV,Er_EMV,'*r',ms=10,label='Port EMV')
plt.plot(Std_minvar,Er_minvar,'*m',ms=10,label='Port Min Var')
plt.colorbar()
plt.legend(loc='best')
plt.show()