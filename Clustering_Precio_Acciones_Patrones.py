# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 10:35:10 2018

@author: if708924
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from datetime import datetime
from eotg import eotg

#%%
names = ['IVV']
start = '09/27/2016'
closes = eotg.get_closes(tickers=names,start_date=start, freq='d')

#%% Sub series
n_dias = 5
colnames=['P1','P2','P3','P4','P5']
sub = pd.DataFrame(index= np.arange(len(closes.index)-n_dias),columns=colnames)
for k in np.arange(n_dias):
    sub[colnames[k]][:] = closes[names[0]][k:len(closes.index)-n_dias+k]

#%% Normalizar la matriz sub
sub_norm = ((sub.transpose()-sub.mean(axis=1))/sub.std(axis=1)).transpose()

#%% trabajar con rendimientos, en vez de precios
rend_sub = sub.transpose().pct_change().dropna().transpose()*100
#%% Grafica de Codo
n_grupos=20
inercias = np.zeros(n_grupos)  
for k in np.arange(n_grupos)+1:
    model = KMeans(n_clusters=k,init='random')
    model = model.fit(sub_norm)
    inercias[k-1] = model.inertia_

plt.plot(np.arange(1,n_grupos+1),inercias)
plt.xlabel('Numero de grupos')
plt.ylabel('Inercia Global')
plt.show()

#%% Reconocimmiento de patrones, se suponen n grupos
model = KMeans(n_clusters=5,init='random')
model = model.fit(sub_norm)
centroides = model.cluster_centers_
plt.plot(centroides[:,0:5].transpose())
plt.xlabel('Dia')
#plt.ylabel('Cambio %') usar solo con rend_sub
plt.title('Patrones')

#%%
Ypredict = model.predict(sub_norm)
#%% Colorear la serie de closes con el centroide actual
nclust = Ypredict[-1]
pos = np.arange(n_dias,len(closes))[Ypredict==nclust]
plt.figure(figsize=(12,6))
plt.plot(closes,'b-')
for k in pos:
    plt.plot(closes.index[np.arange(k-n_dias,k)],closes[names[0]][k-n_dias:k],'r-')
plt.xlabel('Tiempo')
plt.ylabel('Precio')
plt.grid()
plt.show()

#%% Interpretar la agrupacion en funcion del tiempo
plt.subplot(211) #subplot de 2 filas por una columna, en el grafico 1
plt.plot(closes)
plt.xlabel('Tiempo')
plt.ylabel('Precio')
plt.subplot(212)
plt.bar(np.arange(n_dias,len(closes)),Ypredict)
plt.xlabel('Tiempo')
plt.ylabel('Grupo')
plt.show()

#%% Graficar los grupos separados con su respectiva etiqueta
n_subfig = np.ceil(np.sqrt(len(np.unique(Ypredict))))
for k in np.arange(len(np.unique(Ypredict))):
    plt.subplot(n_subfig,n_subfig,k+1)
    plt.plot(centroides[k,:])
    plt.ylabel('Cluster %d'%(k))


#%%
datos_grupo = sub_norm.loc[Ypredict==9,:] # se extraen los datos de un grupo y se grafican como serie de tiempo donde
                                            # cada columna es un valor en el tiempo. Un patrón representa un comportamiento de una
                                            # variable en todas las dimensiones que existen

datos_grupo.transpose().plot()

