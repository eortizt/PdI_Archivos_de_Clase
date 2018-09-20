# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 11:42:01 2018

@author: if708924
"""

# Medir rendimiento esperado en un portafolio a partir de los rendimientos esperados de los activos que lo conforman.
# Medir el riesgo en un portafolio.

import numpy as np
import pandas as pd
#%% Recordando para un solo activo
# Creamos tabla
tabla = pd.DataFrame(columns=['Prob', 'Toyota', 'Walmart', 'Pfizer'], index=['Expansion', 'Normal', 'Recesion', 'Depresion'])
tabla.index.name = 'State'
tabla['Prob']=np.array([0.1, 0.4, 0.3, 0.2])
tabla['Toyota']=np.array([0.06, 0.075, 0.02, -0.03])
tabla['Walmart']=np.array([0.045, 0.055, 0.04, -0.01])
tabla['Pfizer']=np.array([0.025, -0.005, 0.01, 0.13])

tabla.round(6)

#%%
# Toyota
Er_To = np.sum(tabla.Prob*tabla.Toyota)
Er_To

# Walmart
Er_Wmt = np.sum(tabla.Prob*tabla.Walmart)
Er_Wmt

# Toyota
Er_Pf = np.sum(tabla.Prob*tabla.Pfizer)
Er_Pf
#%%Consideramos un portafolio conformado por solamente dos activos de la tabla anterior. La ponderación relativa a cada activo es
# 0.5 Toyota y 0.5 Pfizer.
tabla['Port_TP'] = 0.5*tabla.Toyota+0.5*tabla.Pfizer
# rendimiento esperado del portafolio
Er_porTP = np.sum(tabla.Prob*tabla.Port_TP)
Er_porTP.round(4)

#%%
tabla = pd.DataFrame(columns=['Prob', 'Toyota', 'Walmart', 'Pfizer'], index=['Expansion', 'Normal', 'Recesion', 'Depresion'])
tabla.index.name = 'State'
tabla['Prob']=np.array([0.1, 0.4, 0.3, 0.2])
tabla['Toyota']=np.array([0.06, 0.075, 0.02, -0.03])
tabla['Walmart']=np.array([0.045, 0.055, 0.04, -0.01])
tabla['Pfizer']=np.array([0.025, -0.005, 0.01, 0.13])

tabla.round(6)
# Incluimos una fila para rendimientos esperados y otra para volatilidad
tabla.loc['Expected_ret']=np.array([None, np.sum(tabla.Prob*tabla.Toyota), 
                                    np.sum(tabla.Prob*tabla.Walmart), 
                                    np.sum(tabla.Prob*tabla.Pfizer)])
tabla.loc['Volatility']=np.array([None, None, None, None])

#%% Calcular la volatilidad para cada compania
# Toyota
Vol_To = np.sqrt(np.sum(tabla.Prob[0:4]*(tabla.Toyota[:4]-Er_To)**2))
Vol_To
# Walmart
Vol_Wmt = np.sqrt(np.sum(tabla.Prob[0:4]*(tabla.Walmart[:4]-Er_Wmt)**2))
Vol_Wmt
# Pfizer
Vol_Pf = np.sqrt(np.sum(tabla.Prob[0:4]*(tabla.Pfizer[:4]-Er_Pf)**2))
Vol_Pf

#%%
tabla['Port_TP'] = 0.5*tabla['Toyota']+0.5*tabla['Pfizer'] #incluir el portaflio en la tabla
# Encontrar la volatilidad del portafolio Toyota-Pfizer
Vol_PorTP = np.sqrt(np.sum(tabla.Prob[0:4]*(tabla.Port_TP[:4]-Er_porTP)**2))
Vol_PorTP.round(4)

tabla.Port_TP['Volatility'] = Vol_PorTP



#%% Covarianza entre toyota y pfizer
covTP = np.sum(tabla.Prob[:4]*(tabla.Toyota[:4]-tabla.Toyota[4])*(tabla.Pfizer[:4]-tabla.Pfizer[4]))

#%% CORRELACION ENTRE TOYOTA Y PFIZER
corrTP = covTP/(Vol_To*Vol_Pf)

