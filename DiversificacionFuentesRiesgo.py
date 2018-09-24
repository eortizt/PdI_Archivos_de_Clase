# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 12:09:46 2018

@author: if708924
"""

#  Diversificación y fuentes de riesgo en un portafolio
#  Una ilustración con mercados internacionales

import pandas as pd
import numpy as np
#%%
# Resumen en base anual de rendimientos esperados y volatilidades
annual_ret_summ = pd.DataFrame(columns=['EU', 'RU', 'Francia', 'Alemania', 'Japon'], index=['Media', 'Volatilidad'])
annual_ret_summ.loc['Media'] = np.array([0.1355, 0.1589, 0.1519, 0.1435, 0.1497])
annual_ret_summ.loc['Volatilidad'] = np.array([0.1535, 0.2430, 0.2324, 0.2038, 0.2298])

annual_ret_summ.round(4)

#%%
# Matriz de correlación
corr = pd.DataFrame(data= np.array([[1.0000, 0.5003, 0.4398, 0.3681, 0.2663],
                                    [0.5003, 1.0000, 0.5420, 0.4265, 0.3581],
                                    [0.4398, 0.5420, 1.0000, 0.6032, 0.3923],
                                    [0.3681, 0.4265, 0.6032, 1.0000, 0.3663],
                                    [0.2663, 0.3581, 0.3923, 0.3663, 1.0000]]),
                    columns=annual_ret_summ.columns, index=annual_ret_summ.columns)
corr.round(4)

#%% Nos enfocaremos entonces únicamente en dos mercados: EU y Japón
#  Supongamos que w es la participación del mercado de EU en nuestro portafolio.

# Vector de w variando entre 0 y 1 con n pasos
w = np.linspace(0,1,30)
# Rendimientos esperados individuales
# Activo 1: EU, Activo 2: Japon
E1 = annual_ret_summ.EU.loc['Media']
E2 = annual_ret_summ.Japon.loc['Media']
# Volatilidades individuales
S1 = annual_ret_summ.EU.loc['Volatilidad']
S2 = annual_ret_summ.Japon.loc['Volatilidad']
#Correlación
r12 = corr['EU']['Japon']

#%%
# Crear un DataFrame cuyas columnas sean rendimiento
# y volatilidad del portafolio para cada una de las w
# generadas
# se llama portafolios2 porque son 2 activos
portafolios2 = pd.DataFrame(index=w,columns=['Rendimiento','Volatilidad'])
portafolios2.index.name = 'W'
portafolios2.Rendimiento = w*E1+(1-w)*E2
portafolios2.Volatilidad = np.sqrt((w*S1)**2+((1-w)*S2)**2+2*w*(1-w)*r12*S1*S2)
portafolios2

#%%
# Importar matplotlib
import matplotlib.pyplot as plt

# Graficar el lugar geométrico del portafolio en el
# espacio rendimiento esperado vs. volatilidad.
# Especificar también los puntos relativos a los casos
# extremos.
plt.figure(figsize=(8,6))
plt.plot(S1,E1,'ro',ms=10, label='EU')
plt.plot(S2,E2,'bo',ms=10, label='Japon')
plt.plot(portafolios2.Volatilidad,portafolios2.Rendimiento,'k-',lw=4,label='Portafolios')
plt.xlabel('Volatilidad $sigma$')
plt.ylabel('Rendimiento $E[r]$')
plt.legend(loc='best')
plt.grid()
plt.show()

# Comentario: estrictamente, el portafolio que está más a la izquierda en la curva de arriba es el de volatilidad mínima.
# Sin embargo, como tanto la volatilidad es una medida siempre positiva, minimizar la volatilidad equivale a minimizar la varianza.
# Por lo anterior, llamamos a dicho portafolio, el portafolio de varianza mínima.

#%%  ¿Cómo hallar el portafolio de varianza mínima?
# El minimo de la función de volatilidad vs rendimiento, la derivada evaluada en 0
# var_p = w1**2*s1**2+(1-w1)**2*s2**2+2*w1(1-w)*s12
# se deriva, se iguala a cero y se despeja w, esa w es w_minvar
w_minvar = (S2**2-r12*S1*S2)/(S1**2+S2**2-2*r12*S1*S2)
w_minvar


#%% Con scipy.optimize
import scipy.optimize as opt
# Función objetivo
def var2(w,s1,s2,s12):
    return (w*s1)**2+(1-w)**2*s2**2+2*w*(1-w)*s12
# Dato inicial
w0 = 0
# Volatilidades individuales y covarianza
s1 = annual_ret_summ.EU['Volatilidad']
s2 = annual_ret_summ.Japon['Volatilidad']
s12 = corr['EU']['Japon']*s1*s2

# Cota de w
bnd = (0,1)

# Solución
min_var_2 = opt.minimize(var2, w0, args=(s1,s2,s12),bounds=(bnd,))
min_var_2

#%%
# Graficar el portafolio de varianza mínima
# sobre el mismo gráfico realizado anteriormente
plt.figure(figsize=(8,6))
plt.plot(S1,E1,'ro',ms=10, label='EU')
plt.plot(S2,E2,'bo',ms=10, label='Japon')
plt.plot(portafolios2.Volatilidad,portafolios2.Rendimiento,'k-',lw=4,label='Portafolios')
plt.plot(np.sqrt(min_var_2.fun),min_var_2.x*E1+(1-min_var_2.x)*E2,'mo',ms=10,label='Por_Min_var')
plt.xlabel('Volatilidad $sigma$')
plt.ylabel('Rendimiento $E[r]$')
plt.legend(loc='best')
plt.grid()
plt.show()


#%% para tres activos, obtengamos la frontera de mínima varianza
## Construcción de parámetros

# 1. Sigma: matriz de varianza-covarianza
#       cov_mat
s1 = annual_ret_summ['EU']['Volatilidad']
s2 = annual_ret_summ['Japon']['Volatilidad']
s3 = annual_ret_summ['RU']['Volatilidad']
s12 = corr['EU']['Japon']*s1*s2
s13 = corr['EU']['RU']*s1*s3
s23 = corr['Japon']['RU']*s2*s3
Sigma = np.array([[s1**2, s12, s13],
                  [s12, s2**2, s23],
                  [s13, s23, s3**2]])
    
    
# 2. Eind: rendimientos esperados activos individuales
#       es un vector con los rendimientos esperados individuales   Eind = np.array([E1, E2, E3])
E1 = annual_ret_summ['EU']['Media']
E2 = annual_ret_summ['Japon']['Media']
E3 = annual_ret_summ['RU']['Media']
Eind = np.array([E1, E2, E3])


# 3. Ereq: rendimientos requeridos para el portafolio
#       # Número de portafolios
N = 100
Ereq = np.linspace(Eind.min(), Eind.max(), N)

def varianza(w, Sigma):
    return w.dot(Sigma).dot(w)
def rendimiento_req(w, Eind, Ereq):
    return Eind.dot(w)-Ereq


# Dato inicial
w0 = np.zeros(3,)
# Cotas de las variables
bnds = ((0,None), (0,None), (0,None)) # todas tienen que ser positivas



# DataFrame de portafolios de la frontera
portfolios3 = pd.DataFrame(index=range(N), columns=['w1', 'w2', 'w3', 'Ret', 'Vol'])

# Construcción de los N portafolios de la frontera
for i in range(N):
    # Restricciones
    cons = ({'type': 'eq', 'fun': rendimiento_req, 'args': (Eind,Ereq[i])},
            {'type': 'eq', 'fun': lambda w: np.sum(w)-1})
    # Portafolio de mínima varianza para nivel de rendimiento esperado Ereq[i]
    min_var = opt.minimize(varianza, w0, args=(Sigma,), bounds=bnds, constraints=cons)
    # Pesos, rendimientos y volatilidades de los portafolio
    portfolios3.loc[i,['w1','w2','w3']] = min_var.x
    portfolios3['Ret'][i] = rendimiento_req(min_var.x, Eind, Ereq[i])+Ereq[i]
    portfolios3['Vol'][i] = np.sqrt(varianza(min_var.x, Sigma))


plt.figure(figsize=(8,6))
plt.plot(s1,E1,'ro',ms=5,label='EU')
plt.plot(s2,E2,'bo',ms=5,label='Japón')
plt.plot(s3,E3,'co',ms=5,label='RU')
plt.plot(portfolios3.Vol, portfolios3.Ret, 'k-', lw=4, label='Portafolios 3 act')
plt.plot(portafolios2.Volatilidad,portafolios2.Rendimiento,'g-',lw=2,label='Portafolios2')
plt.xlabel('Volatilidad ($\sigma$)')
plt.ylabel('Rendimiento esperado ($E[r]$)')
plt.grid()
#plt.axis([0.14,0.15,0.135,0.142])
plt.legend(loc='best')


# Portafolio de mínima varianza para 3 activos
cons = ({'type': 'eq', 'fun': lambda w: np.sum(w)-1})
min_var3 = opt.minimize(varianza, w0, args=(Sigma,), bounds=bnds, constraints=cons)

plt.figure(figsize=(8,6))
plt.plot(s1,E1,'ro',ms=5,label='EU')
plt.plot(s2,E2,'bo',ms=5,label='Japón')
plt.plot(s3,E3,'co',ms=5,label='RU')
plt.plot(portfolios3.Vol, portfolios3.Ret, 'k-', lw=4, label='Portafolios 3 act')
plt.plot(portafolios2.Volatilidad,portafolios2.Rendimiento,'g-',lw=2,label='Portafolios2')
plt.plot(np.sqrt(min_var_2.fun),min_var_2.x*E1+(1-min_var_2.x)*E2,'yo',ms=5,label='Por_Min_var2')
plt.plot(np.sqrt(min_var3.fun),min_var3.x.dot(Eind),'mo',ms=5,label='Port Min Var 3')
plt.xlabel('Volatilidad ($\sigma$)')
plt.ylabel('Rendimiento esperado ($E[r]$)')
plt.grid()
#plt.axis([0.14,0.15,0.135,0.142])
plt.legend(loc='best')
