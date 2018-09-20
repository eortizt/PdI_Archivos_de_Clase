class eotg:
    def dqr(data):
        import pandas as pd
        #%% Lista de variables o columnas
        columns = pd.DataFrame(list(data.columns.values))
        
        #%% Lista de tipos de variables
        d_types = pd.DataFrame(data.dtypes,columns=['D_types'])
        
        #%% Lista con los datos faltantes
        missing = pd.DataFrame(data.isnull().sum(),columns=['missing_Values'])
        
        #%% Lista de datos presentes
        present = pd.DataFrame(data.count(),columns=['present_values'])
        
        #%% Tabla de Valores unicos
        unique_values = pd.DataFrame(columns=['Unique_Values'])
        for col in list(data.columns.values):
            unique_values.loc[col] = [data[col].nunique()]
            
        #%% Lista de valores minimos
        min_values = pd.DataFrame(columns=['Min_Values'])
        for col in list(data.columns.values):
            try:
                min_values.loc[col] = [data[col].min()]
            except:
                pass
            
        #%% Lista de valores maximos
        max_values = pd.DataFrame(columns=['Max_Values'])
        for col in list(data.columns.values):
            try:
                max_values.loc[col] = [data[col].max()]
            except:
                pass
            
        #%% Crear reporte de calidad de los datos (data quality report)
        return d_types.join(missing).join(present).join(unique_values).join(min_values).join(max_values)


#%% Get close prices
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
    
#names = ['AAPL', 'WMT', 'ibm', 'NKE']
#start, end= '01/01/2010', '08/29/2018'
#closes = get_closes(tickers=names,start_date=start,end_date=end, freq='d')
    
#%% Rendimientos Anuales de una data frame de rendimientos diarios de ciertos activos
    def calc_annual_ret(ret):
        return (1+ret).groupby(lambda date: date.year).prod()-1
    
    
    #annual_ret = calc_annual_ret(daily_ret) donde daily_ret es un data frame de rendimientos diarios
    #Ver archivo Midiendo_Rendimientos_Historicos
    # daily_ret = closes.pct_change().dropna() donde closes es obenido con la funcion get_closes

#%% Remover signos de puntuacion a una cadena de texto
def remove_punctuation(x):
    import string
    try:
        x = ''.join(c for c in x if c not in string.punctuation)
    except:
        pass
    return x

# ejemplo a un DF data.people = data.people.apply(remove_punctiation)  #del dataframe, columna people, aplicar la funcion que deseas
# es para poder aplicar las funciones que ya existen a un data frame, usando .apply()
#%% Remover numeros de un texto
def remove_digits(x):
    import string
    try:
        x = ''.join(c for c in x if c not in string.digits)
    except:
        pass
    return x

#%% Remover espacios en blanco
def remove_whitespaces(x):
    try:
        x = ''.join(x.split())
    except:
        pass
    return x

#%% Convertir todo a mayusculas
def uppercase_text(x):
    try:
        x = x.upper()
    except:
        pass
    return x

#%% Convertir todo a minusculas
def lowercase_text(x):
    try:
        x = x.lower()
    except:
        pass
    return x
    
#%% Reemplazar texto
def replace_text(x,to_replace,replacement):
    try:
        x = x.replace(to_replace,replacement)
    except:
        pass
    return x

#%% Mantener digitos
def keep_digits(x):
    import string
    try:
        x = ''.join(c for c in x if c in string.digits)
    except:
        pass
    return x