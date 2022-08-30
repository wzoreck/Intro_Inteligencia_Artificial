# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

## Exploração dos dados
base_credit = pd.read_csv('credit_data.csv')

base_credit

base_credit.head(10) # Ver os 10 primeiros registros da base

base_credit.tail(10) # Ver os 10 últimos registros da base

base_credit.describe() # Mostra uma descrição da dos dados da base

base_credit[base_credit['income'] >= 69995] # Buscar na base de dados quem tem a renda igual ou maior que o valor informado

## Visualizacao de dados
np.unique(base_credit['default'], return_counts=True) # Mostra a amostragem de dados nesse caso 0 ou 1, e a quantidade de cada

sns.countplot(x=base_credit['default']) #Gerar um gráfico

plt.hist(x = base_credit['age'])

plt.hist(x = base_credit['income'])

plt.hist(x = base_credit['loan'])

sns.pairplot(base_credit, vars=['age', 'income'], hue='default') # Plotando gráfico comparando idade com renda e mostrando um indice de quem pagou e não pagou

sns.pairplot(base_credit, vars=['age', 'income', 'loan'], hue='default') # Relacionando três campos

## Tratamento de valores inconsistentes
base_credit.loc[base_credit['age'] < 0] # Encontrando idades erradas

    # Alternativas
base_credit2 = base_credit.drop('age', axis=1) # Apagar uma coluna inteira

base_credit[base_credit['age'] < 0].index # Pegar os indices

base_credit_3 = base_credit.drop(base_credit[base_credit['age']< 0].index) # Apagar apenas os índices que saem do desvio padrão

base_credit.mean() # Médias dos campos

base_credit['age'][base_credit['age'] > 0].mean() # Média do campo com idades maiores que 0

base_credit.loc[base_credit['age'] < 0, 'age'] = 40.93 # Alterando o valores de idade dos campos pela média (é uma alternativa)

## Tratamento de valores nulos
base_credit.isnull().sum()

base_credit.loc[pd.isnull(base_credit['age'])] # Mostrando os que possuem idade nula

base_credit.loc[pd.isnull(base_credit['age']), 'age'] = 40.93 # Substituindo valores nulos pela média de idade


## Divisão base de dados (Previsores [atribuos] e Classes[valores a aprender])

X_credit = base_credit.iloc[:, 1:4].values # Gerando um array com os Previsores

Y_credit = base_credit.iloc[:, 4].values # Gerando um array  com as Classes

# Escalonamento dos valores
X_credit[:, 0].min(), X_credit[:, 1].min(), X_credit[:, 2].min()

X_credit[:, 0].max(), X_credit[:, 1].max(), X_credit[:, 2].max()


# Normalização (IA)

from sklearn.preprocessing import MinMaxScaler
scaler_credit = MinMaxScaler()

X_credit = scaler_credit.fit_transform(X_credit) # Normalização dos dados previsores

