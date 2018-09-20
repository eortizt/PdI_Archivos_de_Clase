# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 16:32:57 2018

@author: Esteban Ortiz Tirado
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from eotg import eotg
#%% Importar datos desde csv descargado de banxico
usdmxn_csv = pd.read_csv('Precios/tipoCambio.csv', index_col='Fecha').dropna()

#usdmxn = usdmxn[~usdmxn['Determinación'].isin(['N/E'])]
usdmxn_csv.columns=['USD/MXN']
usdmxn_csv.plot(figsize=(12,6))

#%% Calcular rendimientos anuales usando eotg.calc_annual_ret
usdmxn = usdmxn_csv
usdmxn.index = pd.to_datetime(usdmxn.index, yearfirst=True, box=True ,infer_datetime_format=True)
day_ret = usdmxn.pct_change().dropna()
annual_ret = eotg.calc_annual_ret(day_ret)
#plt.plot(annual_ret)
#%%  Graficar rendimientos anuales
annual_ret.plot.bar(figsize=(12,6))
plt.title('Rendimiento USD/MXN')

#%% Graficar por año
##usdmxn.groupby(lambda date: date.year)

usdmxn_csv.iloc[9743-365:9743].plot(figsize=(12,6))
usdmxn['2018'].plot(figsize=(12,6))

start = 2016
end = 2019
usdmxn_csv.iloc[9743-246-365*(2018-start):9743-246-365*(2018-end)].plot(figsize=(12,6))

# Por que el archivo usdmxn_csv no reconoce como fecha el indice? y por que la grafica de usdmxn, una vez cambiado el indice a fecha se distorsiona?

#%%Mostrar rendimientos de ciertos años, 27 es 2018, ver tamaño de annual_ret
annual_ret.iloc[18:27]