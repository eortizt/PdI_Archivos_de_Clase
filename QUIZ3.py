# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 11:18:17 2018

@author: if708924
"""

import numpy as np
import pandas as pd

tabla = pd.DataFrame(columns=['Prob', 'A', 'B', 'C'], index=['Expansion', 'Normal', 'Recesion'])
tabla.index.name = 'State'
tabla['Prob']=np.array([0.3, 0.4, 0.3])
tabla['A']=np.array([-.2, 0.05, 0.4])
tabla['B']=np.array([-.05, 0.1, 0.15])
tabla['C']=np.array([0.05, 0.03, 0.02])
tabla.loc['Expected_ret']=np.array([None, np.sum(tabla.Prob*tabla.A), 
                                    np.sum(tabla.Prob*tabla.B), 
                                    np.sum(tabla.Prob*tabla.C)])

tabla.round(6)

covAB = np.sum(tabla.Prob[:3]*(tabla.A[:3]-tabla.A[3])*(tabla.B[:3]-tabla.B[3]))
covAC = np.sum(tabla.Prob[:3]*(tabla.A[:3]-tabla.A[3])*(tabla.C[:3]-tabla.C[3]))
covBC = np.sum(tabla.Prob[:3]*(tabla.B[:3]-tabla.B[3])*(tabla.C[:3]-tabla.C[3]))


#%%
Vol_A = np.sqrt(np.sum(tabla.Prob[0:3]*(tabla.A[:3]-(np.sum(tabla.Prob*tabla.A)))**2))
Vol_B = np.sqrt(np.sum(tabla.Prob[0:3]*(tabla.B[:3]-(np.sum(tabla.Prob*tabla.B)))**2))
Vol_C = np.sqrt(np.sum(tabla.Prob[0:3]*(tabla.C[:3]-(np.sum(tabla.Prob*tabla.C)))**2))

corrAC = covAC/(Vol_A*Vol_C)
corrCB = covBC/(Vol_C*Vol_B)
corrAB = covAB/(Vol_A*Vol_B)

#%%
tabla['Port_AC'] = 0.5*tabla['A']+0.5*tabla['C'] #incluir el portaflio en la tabla
# Encontrar la volatilidad del portafolio
Er_porAC = 0.5*np.sum(tabla.Prob*tabla.A)+0.5*np.sum(tabla.Prob*tabla.C)
Vol_PorAC = np.sqrt(np.sum(tabla.Prob[0:3]*(tabla.Port_AC[:3]-Er_porAC)**2))
Vol_PorAC.round(4)

