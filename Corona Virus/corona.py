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


# Fitting Polynomial Regression to the dataset 
from sklearn.preprocessing import PolynomialFeatures 
  
poly = PolynomialFeatures(degree = 4) 
X_poly = poly.fit_transform(X) 
  
poly.fit(X_poly, y) 
lin2 = LinearRegression() 
lin2.fit(X_poly, y) 

poly_previsao = lin2.predict(poly.fit_transform(X))

def viz_linear():
    plt.plot(X, y, color='red',marker='o', linestyle='none')
    #plt.plot(X, previsao, color='blue',linestyle='solid')
    plt.plot(X, poly_previsao, color='black',linestyle='solid')
    plt.title('Corona Virus (regressão polinomial)')
    plt.xlabel('Dias')
    plt.ylabel('Casos')
    plt.show()
    return
viz_linear()

print(lin2.predict(poly.fit_transform([[45]])))



print("Model slope:    ", lin2.coef_[0])
print("Model intercept:", lin2.intercept_)



