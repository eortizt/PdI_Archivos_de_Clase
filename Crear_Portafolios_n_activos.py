# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""

# Importamos pandas y numpy
import pandas as pd
import numpy as np

# Resumen en base anual de rendimientos esperados y volatilidades
annual_ret_summ = pd.DataFrame(columns=['EU', 'RU', 'Francia', 'Alemania', 'Japon'], index=['Media', 'Volatilidad'])
annual_ret_summ.loc['Media'] = np.array([0.1355, 0.1589, 0.1519, 0.1435, 0.1497])
annual_ret_summ.loc['Volatilidad'] = np.array([0.1535, 0.2430, 0.2324, 0.2038, 0.2298])

annual_ret_summ.round(4)

# Matriz de correlación
corr = pd.DataFrame(data= np.array([[1.0000, 0.5003, 0.4398, 0.3681, 0.2663],
                                    [0.5003, 1.0000, 0.5420, 0.4265, 0.3581],
                                    [0.4398, 0.5420, 1.0000, 0.6032, 0.3923],
                                    [0.3681, 0.4265, 0.6032, 1.0000, 0.3663],
                                    [0.2663, 0.3581, 0.3923, 0.3663, 1.0000]]),
                    columns=annual_ret_summ.columns, index=annual_ret_summ.columns)
corr.round(4)

# Tasa libre de riesgo
rf = 0.05
#%%
# Encontrar portafolio de mínima varianza
# Importamos minimize de optimize
from scipy.optimize import minimize

## Construcción de parámetros
# 1. Sigma: matriz de varianza-covarianza
D = np.diag(annual_ret_summ.loc['Volatilidad'])
Sigma = D.dot(corr).dot(D)
# 2. Eind: rendimientos esperados activos individuales
Eind = np.array(annual_ret_summ.loc['Media'])

# Función objetivo
def varianza(w, Sigma):
    return w.dot(Sigma).dot(w)

# Cantidad de activos
n = 5
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
bnds = ((0,1),)*5
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
w = np.linspace(-2,4,100)

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
import matplotlib.pyplot as plt

# Gráfica de dispersión de puntos coloreando 
# de acuerdo a SR
plt.figure(figsize=(10,7))
plt.scatter(frontera.Vol,frontera.Rend,c=frontera.SR,cmap='RdYlBu')
plt.plot(annual_ret_summ.loc['Volatilidad'],annual_ret_summ.loc['Media'],'og')
plt.plot(Std_EMV,Er_EMV,'*r',ms=10,label='Port EMV')
plt.plot(Std_minvar,Er_minvar,'*m',ms=10,label='Port Min Var')
plt.colorbar()
plt.legend(loc='best')
plt.show()


# A partir de lo anterior, solo restaría construir la LAC y elegir la distribución de capital de acuerdo a las preferencias (aversión al riesgo).