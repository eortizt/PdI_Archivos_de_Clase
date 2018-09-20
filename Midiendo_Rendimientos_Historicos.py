# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 12:09:41 2018

@author: if708924
"""

# Importar paquetes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web
import scipy.optimize as opt
#%%
# Función para descargar precios de cierre ajustados de varios activos a la vez:
def get_closes(tickers, start_date=None, end_date=None, freq=None):
    # Fecha inicio por defecto (start_date='2010-01-01') y fecha fin por defecto (end_date=today)
    # Frecuencia de muestreo por defecto (freq='d')
    # Importamos paquetes necesarios
    import pandas as pd
    pd.core.common.is_list_like = pd.api.types.is_list_like
    import pandas_datareader.data as web  
    # Creamos DataFrame vacío de precios, con el índice de las fechas
    closes = pd.DataFrame(columns = tickers, index=web.YahooDailyReader(symbols=tickers[0], start=start_date, end=end_date, interval=freq).read().index)
    # Agregamos cada uno de los precios con YahooDailyReader
    for ticker in tickers:
        df = web.YahooDailyReader(symbols=ticker, start=start_date, end=end_date, interval=freq).read()
        closes[ticker]=df['Adj Close']
    closes.index_name = 'Date'
    closes = closes.sort_index()
    return closes
#%%Informacion
names = ['AAPL', 'WMT', 'ibm', 'NKE']
start, end= '01/01/2010', '08/29/2018'

#%%
# Precios diarios
closes = get_closes(tickers=names,start_date=start,end_date=end, freq='d')
closes
# Gráfico de histórico de precios diarios
closes.plot(figsize=(8,6))
#%% Calcular rendimientos

# SE UTILIZAN LOS RENDIMIENTOS EN LAS FINANZAS PORQUE TIENEN UNA TENDENCIA ALREDEDOR DEL 0, A DIFERENCIA DE LOS PRECIOS

#Hasta aqui todo igual a descarga de precios

#%%
# Método shift() de un DataFrame...
Lcloses = closes.shift() #shift es funcion retraso de una serie
# Calcular los rendimientos
daily_ret = ((closes-Lcloses)/Lcloses).dropna()

#%% Metodo facil
daily_ret = closes.pct_change().dropna()
plt.plot(daily_ret)
plt.show()

#%% Reportar rendimientos en base anual
def calc_annual_ret(ret):
    return (1+ret).groupby(lambda date: date.year).prod()-1
#%%
annual_ret = calc_annual_ret(daily_ret)

#%% Rendimientos Logaritmicos, rendimientos compuestos continuos
daily_logret = np.log(closes/closes.shift()).dropna()

#%% Midiendo el error de los rendimientos log contra porcentuales
err = np.abs(daily_ret-daily_logret)



#%% Caracterizacion de los rendimientos
media_diaria = daily_ret.mean()
desv_est = daily_ret.std()

#resumen de medias y volatilidad (desv. estandar)
daily_ret_summary = pd.DataFrame(index=['Mean', 'Volatility'],columns=names)
daily_ret_summary.loc['Mean']=media_diaria
daily_ret_summary.loc['Volatility']=desv_est

#%%

# Resumen en base anual
annual_ret_summary = pd.DataFrame(index=['Mean', 'Volatility'],columns=names)*252
annual_ret_summary.loc['Mean']=media_diaria*252
annual_ret_summary.loc['Volatility']=desv_est*np.sqrt(252)
annual_ret_summary

#%%
# Gráfico rendimiento esperado vs. volatilidad
x_points = annual_ret_summary.loc['Volatility'].values
y_points = annual_ret_summary.loc['Mean'].values
plt.figure(figsize=(8,6))
plt.plot(x_points,y_points,'ro',ms=10)
plt.xlabel('Volatility $\sigma$')
plt.ylabel('Expected Return $E[r]$')
#Etiqueta de cada instrumento
plt.text(x_points[0],y_points[0],names[0])
plt.text(x_points[1],y_points[1],names[1])
plt.text(x_points[2],y_points[2],names[2])
plt.text(x_points[3],y_points[3],names[3])




#%% Ajuste lineal
def fun_obj(b,x,y):
    return np.sum((y-b[0]-b[1]*x)**2)
    
b0=[0,0]

#%% Optimizacion de la funcion, el ajuste lineal

res = opt.minimize(fun_obj,b0, args=(x_points,y_points))


# res.x[0] es B0 que significa el rendimiento libre de riesgo
# res.x[] es B1 que significa cuanto mas rendimiento esperado se obtiene por cada unidad de riesgo extra que se asume
#%%Grafico recta ajustada
# Gráfico rendimiento esperado vs. volatilidad
x_points = annual_ret_summary.loc['Volatility'].values
y_points = annual_ret_summary.loc['Mean'].values
plt.figure(figsize=(8,6))
plt.plot(x_points,y_points,'ro',ms=10)
plt.xlabel('Volatility $\sigma$')
plt.ylabel('Expected Return $E[r]$')
#Etiqueta de cada instrumento
plt.text(x_points[0],y_points[0],names[0])
plt.text(x_points[1],y_points[1],names[1])
plt.text(x_points[2],y_points[2],names[2])
plt.text(x_points[3],y_points[3],names[3])
plt.plot(np.sort(x_points),res.x[0]+res.x[1]*np.sort(x_points))





