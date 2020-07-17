import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

base = pd.read_csv('./corona.csv', index_col=None)


X = base.iloc[:,0].values
X = X.reshape(-1, 1)
y = base.iloc[:, 1].values
y = y.reshape(-1, 1)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)


coef = model.coef_

previsao = model.predict(X)

def viz_linear():
    plt.plot(X, y, color='red',marker='o', linestyle='none')
    plt.plot(X, previsao, color='blue',linestyle='solid')
    plt.title('Truth or Bluff (Linear Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
    return
viz_linear()

