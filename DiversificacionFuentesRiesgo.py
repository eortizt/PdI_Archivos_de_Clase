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

w_minvar = (S2**2-r12*S1*S2)/(S1**2+S2**2-2*r12*S1*S2)
w_minvar