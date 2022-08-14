from sklearn.linear_model import LinearRegression, Lars, Ridge, Lasso, ElasticNet, SGDRegressor, BayesianRidge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib import pyplot as plt
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm
import pandas as pd



def ajusta_modelo_lineal(df, X, y, tgt='tgt', modelo=None, flag=False):
    ajuste = modelo.fit(X, y)
    predicted = ajuste.predict(X)
    print('*'*100)
    print(f'{modelo}')
    print('*'*100)
    print('Estadísticos de interés')
    print('*'*100)
    print(f'El modelo tiene un error aboluto medio de: {mean_absolute_error(y, predicted)}')
    print(f'El modelo tiene un error cuadrático medio de: {mean_squared_error(y, predicted)}')
    print('*'*100)
    print('Coeficientes:')
    for i in range(len(list(modelo.coef_))):
        print(f'Beta{i} = {list(modelo.coef_)[i]}, {list(X.columns)[i]}')
    df[f'predicted_by_{modelo}'] = modelo.predict(X)
    df[f'error_{modelo}'] = df[tgt] -  df[f'predicted_by_{modelo}']
    fig = px.histogram(df, x=f'error_{modelo}', marginal='box')
    fig.show()
    if flag:
        for i in list(X.columns):
            sm.qqplot(df[i], line ='45')
    return df