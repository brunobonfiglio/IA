import MetaTrader5
from datetime import datetime
from MetaTrader5 import * 
from pytz import timezone 

import pandas as pd

utc_tz = timezone('UTC') 
 
 
   # connect to MetaTrader 5
MT5Initialize()
   # wait till MetaTrader 5 establishes connection to the trade server and synchronizes the environment
MT5WaitForTerminal()
 
   # request connection status and parameters
print(MT5TerminalInfo())
   # get data on MetaTrader 5 version
print(MT5Version())
 
 # request ticks from EURAUD within 2019.04.01 13:00 - 2019.04.02 13:00 
euraud = MT5CopyRatesRange("IBOV", MT5_TIMEFRAME_M1, datetime(2019,4,17,13), datetime(2019,5,31,13))
df = pd.DataFrame(list(euraud), columns=['datetime', 'open', 'low','high','close', 'tick_volume', 'spread', 'real_volume' ])
df.datetime = pd.to_datetime(df.datetime, format='%d.%m.%Y %H:%M:$S.%f')

#print(euraud)
MT5Shutdown()

ma72 = df.close.rolling(center=False, window=120).mean()
ma72 = pd.DataFrame(list(ma72))

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

ma72_imputer = imputer.fit(ma72[:])
ma72[:] = ma72_imputer.transform(ma72[:])


df = df.merge(
        ma72, how="inner", left_index=True, right_index=True
    )

ma17 = df.close.rolling(center=False, window=20).mean()
ma17 = pd.DataFrame(list(ma17))
ma17_imputer = imputer.fit(ma17[:])
ma17[:] = ma72_imputer.transform(ma17[:])


df = df.merge(
        ma17, how="inner", left_index=True, right_index=True
    )

data = df.iloc[: , [1,2,3,4,7,8,9]].values

dfNovo=[]
janela = 2
for x in range(len(data[:,3])):
    arr = []
    window = 2
    if len(data[:,3]) > x + window:
        arr.append((data[x + 2,0])-data[x,0])
        arr.append(data[x,0]-(data[x - window,0]))
        arr.append(data[x,4])  
        arr.append((data[x,5]-data[x,0]))
        arr.append((data[x,6]-data[x,0])) 
    #print((data[x + janela,3])-data[x,3])
    #print(x + janela)
    '''
    for z in range(1,janela):
        arr.append((data[x - z,0])-data[x,0])
        arr.append((data[x - z,1])-data[x,0])
        arr.append((data[x - z,2])-data[x,0])

    '''
       
    dfNovo.append(arr) 
    
    
base = pd.DataFrame(dfNovo)  
#exclui as primeiras e ultimas linhas
base.drop(list(range(janela +100)), axis=0,inplace=True)
base.drop(list(range(len(data)-(janela + 10),len(data))), axis=0,inplace=True)


base[0].loc[base[0] < 0] = 0
base[0].loc[base[0] > 0] = 1

#-------------------------

               
previsores = base.iloc[:, 1:5].values
classe = base.iloc[:, 0].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)


imputer = imputer.fit(previsores[:, 1:5])
previsores[:, 1:5] = imputer.transform(previsores[:, 1:5])




from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

from sklearn.neural_network import MLPClassifier
classificador = MLPClassifier(verbose=True,
                              max_iter=10000,
                              tol=0.000001,
                              solver='adam',
                              hidden_layer_sizes=(1000,1000),
                              activation='logistic',
                              )
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)