#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 19:16:25 2022

@author: daniel
"""

############ Regressão simples (A partir de aributos tentamos encontrar um valor)
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


base_plano_saude = pd.read_csv("plano_saude.csv")

X_plano_saude = base_plano_saude.iloc[:, 0].values
y_plano_saude = base_plano_saude.iloc[:, 1].values

np.corrcoef(X_plano_saude, y_plano_saude) # Olhar a correlação dos dos dados X e y

# Colocar X no formato de matriz
X_plano_saude = X_plano_saude.reshape(-1, 1) # NO caso de classificacao ou regressao com apenas 1 atributo, é necessário fazer isso

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_plano_saude, y_plano_saude)

# Previsoes
previsoes = regressor.predict(X_plano_saude)

grafico = px.scatter(x=X_plano_saude.ravel(), y=y_plano_saude)

grafico.add_scatter(x=X_plano_saude.ravel(), y=previsoes, name='Regressao')

grafico.write_html('grafico0.html')

# Previsao para idade especifica
regressor.predict([[22]])

##### Base dados casas
base_casas = pd.read_csv('house_prices.csv')

# grafico de correlação
figura = plt.figure(figsize=(40, 40))

sns.heatmap(base_casas.corr(), annot=True) # Visualizar

X_casas = base_casas.iloc[:, 5:6].values

y_casas = base_casas.iloc[:, 2].values

from sklearn.model_selection import train_test_split

X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(X_casas, y_casas, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_casas_treinamento, y_casas_treinamento)

regressor.score(X_casas_treinamento, y_casas_treinamento)

regressor.score(X_casas_teste, y_casas_teste)

previsoes = regressor.predict(X_casas_treinamento)

grafico = px.scatter(x=X_casas_treinamento.ravel(), y=previsoes)

grafico.write_html('grafico1.html')

grafico1 = px.scatter(x=X_casas_treinamento.ravel(), y=y_casas_treinamento)
grafico2 = px.line(x=X_casas_treinamento.ravel(), y=previsoes)
grafico2.data[0].line.color = 'red'
grafico3 = go.Figure(data=grafico1.data+grafico2.data)

grafico3.write_html('grafico2.html')

previsoes_teste = regressor.predict(X_casas_teste)

from sklearn.metrics import mean_absolute_error, mean_squared_error

mean_absolute_error(y_casas_teste, previsoes_teste) # Marge de erro acima e abaixo que pode ser calculada

mean_squared_error(y_casas_teste, previsoes_teste) # usado para comparar com outros algoritmos, tentar entender esse valor não faz muito sentido


# Base casas com multiplas colunas
X_casas = base_casas.iloc[:, 3:19].values

y_casas = base_casas.iloc[:, 2].values

from sklearn.model_selection import train_test_split

X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(X_casas, y_casas, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_casas_treinamento, y_casas_treinamento)

previsoes_teste = regressor.predict(X_casas_teste)

from sklearn.metrics import mean_absolute_error, mean_squared_error

mean_absolute_error(y_casas_teste, previsoes_teste) # Marge de erro acima e abaixo que pode ser calculada


# Base casas com multiplas colunas Polinomial
from sklearn.preprocessing import PolynomialFeatures

X_casas = base_casas.iloc[:, 3:19].values

y_casas = base_casas.iloc[:, 2].values

from sklearn.model_selection import train_test_split

X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(X_casas, y_casas, test_size=0.3, random_state=0)


poly = PolynomialFeatures(degree=2)

X_casas_treinamento_poly = poly.fit_transform(X_casas_treinamento)
X_casas_teste_poly = poly.fit_transform(X_casas_teste)

X_casas_treinamento_poly.shape

from sklearn.linear_model import LinearRegression # Usa o linear, pq o polinomial apenas altera a base de dados

regressor = LinearRegression()

regressor.fit(X_casas_treinamento_poly, y_casas_treinamento)

previsoes_teste = regressor.predict(X_casas_teste_poly)

from sklearn.metrics import mean_absolute_error, mean_squared_error

mean_absolute_error(y_casas_teste, previsoes_teste) # Marge Média de erro acima e abaixo que pode ser calculada


########################## Decision Tree ##########################
from sklearn.tree import DecisionTreeRegressor # Nas folhas da árvore, vai ter a média dos atributos

regressor = DecisionTreeRegressor()

regressor.fit(X_casas_treinamento, y_casas_treinamento)

regressor.score(X_casas_treinamento, y_casas_treinamento)

regressor.score(X_casas_teste, y_casas_teste)

previsoes_teste = regressor.predict(X_casas_teste)

mean_absolute_error(y_casas_teste, previsoes_teste)

####################### Random Forest ########################## 
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=100)

regressor.fit(X_casas_treinamento, y_casas_treinamento)

regressor.score(X_casas_treinamento, y_casas_treinamento)

regressor.score(X_casas_teste, y_casas_teste)

previsoes_teste = regressor.predict(X_casas_teste)

mean_absolute_error(y_casas_teste, previsoes_teste)


##################### SVR ###########################
from sklearn.svm import SVR


regressor = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)

regressor.fit(X_casas_treinamento, y_casas_treinamento)


regressor.score(X_casas_treinamento, y_casas_treinamento)

regressor.score(X_casas_teste, y_casas_teste)

previsoes_teste = regressor.predict(X_casas_teste)

mean_absolute_error(y_casas_teste, previsoes_teste)











