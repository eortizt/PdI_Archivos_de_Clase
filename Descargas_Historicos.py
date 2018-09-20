import pandas as pd
#descargar archivo excel de yahoo en una carpeta llamada Precios
# Cargamos hoja de calculo en un dataframe
file_name = 'Precios/AAPL.csv'
aapl = pd.read_csv(file_name)
#indizar por fecha
aapl = pd.read_csv(file_name, index_col='Date')


# Graficar precios de cierre y precios de cierre ajustados
import matplotlib.pyplot as plt
aapl[['Close', 'Adj Close']].plot(figsize=(8,6));


# Para solo tener adj close
aapl = pd.read_csv(file_name, index_col='Date', usecols=['Date', 'Adj Close'])
aapl.columns = ['AAPL']
aapl

# Importar el modulo data del paquete pandas_datareader. La comunidad lo importa con el nombre de web
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web

# Ejemplo google finance
ticker = 'AAPL'
source = 'google'
start = '2015-01-01'
end = '2017-12-31'
aapl_goo = web.DataReader(ticker, source, start, end)

# Ejemplo quandl
ticker = 'AAPL'
source = 'quandl'
start = '2015-01-01'
end = '2017-12-31'
aapl_qua = web.DataReader(ticker, source, start, end)
aapl_qua

# Ejemplo iex
ticker = 'AAPL'
source = 'iex'
start = '2015-01-01'
end = '2017-12-31'
aapl_iex = web.DataReader(ticker, source, start, end)
aapl_iex

# YahooDailyReader
ticker = 'AAPL'
start = '2015-01-01'
end = '2017-12-31'
aapl_yah = web.YahooDailyReader(ticker, start, end, intervals='d'.read())
aapl_yah

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

#%% Descargar precios
names = ['AAPL', 'WMT', 'ibm', 'NKE']
start, end= '01/01/2010', '08/29/2018'

closes = get_closes(tickers=names,start_date=start,end_date=end, freq='d')
# Gráfico
closes.plot(figsize=(8,6))