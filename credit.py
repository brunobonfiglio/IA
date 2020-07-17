import pandas as pd
base = pd.read_csv('credit-data.csv')

#print(base.describe())

#print(base.loc[base['age'] < 0])



#apagar todas as colunas
#base.drop('age', 1, inplace=True)




#apagar somente os registro com problema

#print(base[base.age < 0])
#base.drop(base[base.age < 0].index, inplace=True)



#base.mean()
#base['age'][base.age < 0].mean()

#atualizar campo na coluna
base.loc[base.age < 0, 'age'] = 40.92

#print(base.loc[pd.isnull(base['age'])])

previsores = base.iloc[:,1:4].values
classes = base.iloc[:, 4].values

from sklearn.preprocessing import Imputer


imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)