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


#### Outra basa de dados -------------------------------------------------------------------------------

# Exploração dos dados
base_census = pd.read_csv('census.csv')

base_census.describe()

base_census.isnull().sum()

# Visualização dos dados
np.unique(base_census['income'], return_counts=True)

sns.countplot(x=base_census['income'])

plt.hist(x=base_census['age'])

plt.hist(base_census['education-num'])

plt.hist(base_census['hour-per-week'])

import plotly.express as px

grafico = px.treemap(base_census, path=['workclass'])

grafico.write_html('grafico0.html')

grafico = px.treemap(base_census, path=['workclass', 'age'])

grafico.write_html('grafico1.html')

grafico = px.treemap(base_census, path=['occupation', 'relationship', 'age'])

grafico.write_html('grafico2.html')

    # Categorias paralelas
grafico = px.parallel_categories(base_census, dimensions=['occupation', 'relationship'])

grafico.write_html('grafico3.html')

grafico = px.parallel_categories(base_census, dimensions=['native-country', 'income'])

grafico.write_html('grafico4.html')

grafico = px.parallel_categories(base_census, dimensions=['workclass', 'occupation', 'income'])

grafico.write_html('grafico5.html')

grafico = px.parallel_categories(base_census, dimensions=['education', 'income'])

grafico.write_html('grafico6.html')

    # Divisao Previsores e Classes
base_census.columns

X_census = base_census.iloc[:, 0:14].values

Y_census = base_census.iloc[:, 14].values

    # Tratamento de atributos categóricos
    ## LABELENCODER - converte string para inteiro ex: P M G -> 0 1 2
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

indices = [1, 3, 5, 6, 7, 8, 9, 13]

for i in indices:
    X_census[:, i] = label_encoder.fit_transform(X_census[:, i])
    
    ## ONEHOTENCODER
    # EX Carro - Gol Pálio Uno
    
    # No LABELENCODER seria
    # Gol Pálio Uno
    # 0   1     2
    
    # No ONEHOTENCODER
    # Gol   1 0 0
    #Pálio  0 1 0
    #Uno    0 0 1
    
    # Não há peso por ser apenas 0 e 1 (há algoritmos que isso interfere)
    
len(np.unique(base_census['workclass']))

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

onehotencoder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), indices)], remainder='passthrough')

X_census = onehotencoder_census.fit_transform(X_census).toarray()

X_census.shape # Qtd registros e colunas

# Escalonamento de valores

from sklearn.preprocessing import StandardScaler

scaler_census = StandardScaler()

X_census = scaler_census.fit_transform(X_census)

# Divisão bases Treinamento e Teste
from sklearn.model_selection import train_test_split

X_census_treinamento, X_census_teste, Y_census_treinamento, Y_census_teste = train_test_split(X_census, Y_census, test_size=0.25, random_state=0)


X_credit_treinamento, X_credit_teste, Y_credit_treinamento, Y_credit_teste = train_test_split(X_credit, Y_credit, test_size=0.25, random_state=0)

# Salvar variáveis
import pickle

with open('census.pkl', mode='wb') as f:
    pickle.dump([X_census_treinamento, X_census_teste, Y_census_treinamento, Y_census_teste], f)
    
with open('credit.pkl', mode='wb') as f:
    pickle.dump([X_credit_treinamento, X_credit_teste, Y_credit_treinamento, Y_credit_teste], f)


    






