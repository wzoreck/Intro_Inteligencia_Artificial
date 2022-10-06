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


#######################################################################
#Aula 04

# Naive Bayes (Algoritmo problabilistico)
from sklearn.naive_bayes import GaussianNB

base_risco_credito = pd.read_csv('risco_credito.csv')
 
X_risco_credito = base_risco_credito.iloc[:, 0:4].values # X indica um vetor, x indica um vetor
y_risco_credito = base_risco_credito.iloc[:, 4].values # Nesse caso um vetor

from sklearn.preprocessing import LabelEncoder
label_encoder_risco = LabelEncoder()

indices_risco = [0, 1, 2, 3]

for i in indices_risco:
    X_risco_credito[:, i] = label_encoder_risco.fit_transform(X_risco_credito[:, i])

import pickle

with open('risco_credito.pkl', 'wb') as f:
    pickle.dump([X_risco_credito, y_risco_credito], f)

naive_risco_credito = GaussianNB()

# ML Aprendendo
naive_risco_credito.fit(X_risco_credito, y_risco_credito)

# história: Boa (0), dívida: alta (0), garantias: nenhuma (1), renda: > 35 (2)

# história: Ruim (2), dívida: alta (0), garantias: adequada (0), renda: < 15 (0)

previsao = naive_risco_credito.predict([[0, 0, 1, 2], [2, 0, 0, 0]]) # Passsando uma matriz

previsao

################################################ Aplicando IA nas bases credit_data
with open('credit.pkl', 'rb') as f:
    X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = pickle.load(f)
    
# Ver se X e y tem a mesma qtd de dados
X_credit_treinamento.shape, y_credit_treinamento.shape

X_credit_teste.shape, Y_credit_teste.shape

naive_credit = GaussianNB()

    #ML
naive_credit.fit(X_credit_treinamento, y_credit_treinamento)

previsoes_credit = naive_credit.predict(X_credit_teste)

# Teste do algoritmo treinado
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy_score(y_credit_teste, previsoes_credit)

confusion_matrix(y_credit_teste, previsoes_credit)

# NO terminal instalar (pip install yellowbrick)

from yellowbrick.classifier import ConfusionMatrix

cm = ConfusionMatrix(naive_credit)
cm.fit(X_credit_treinamento, y_credit_treinamento)
cm.score(X_credit_teste, y_credit_teste)

print(classification_report(y_credit_teste, previsoes_credit))


##################### base census
with open('census.pkl', 'rb') as f:
    X_census_treinamento, X_census_teste, y_census_treinamento, y_census_teste = pickle.load(f)

X_census_treinamento.shape, y_census_treinamento.shape
X_census_teste.shape, y_census_teste.shape

naive_census = GaussianNB()
naive_census.fit(X_census_treinamento, y_census_treinamento)

previsoes_census = naive_census.predict(X_census_teste)
   
accuracy_score(y_census_teste, previsoes_census)

cm = ConfusionMatrix(naive_census) # Fazer ainda em casa...


##########################################################################################################
#Aula 05 - Arvore de decisao

from sklearn.tree import DecisionTreeClassifier

# Base de dados crédito
import pickle

with open('risco_credito.pkl', 'rb') as f:
    X_risco_credito, y_risco_credito = pickle.load(f)

arvore_rico_credito = DecisionTreeClassifier(criterion='entropy')

arvore_rico_credito.fit(X_risco_credito, y_risco_credito)

arvore_rico_credito.feature_importances_ # MOstra uais atributos tem mais importancia (qual elege como nó)

arvore_rico_credito.classes_

previsao = arvore_rico_credito.predict([[0, 0, 1, 2], [2, 0, 0, 0]])

from sklearn import tree
previsores = ['história', 'dívida', 'garantias', 'renda']
figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
tree.plot_tree(arvore_rico_credito, feature_names=previsores, class_names=arvore_rico_credito.classes_, filled=True)

figura.savefig('risco_credito_tree.pdf')

#################### Base credit data
with open('credit.pkl', 'rb') as f:
    X_credit_treinamento, X_credit_teste, y_credit_treinamento, Y_credit_teste = pickle.load(f)


arvore_credit = DecisionTreeClassifier(criterion='entropy')

arvore_credit.fit(X_credit_treinamento, y_credit_treinamento)

previsoes = arvore_credit.predict(X_credit_teste)

# Avaliacao
from sklearn.metrics import accuracy_score, classification_report

accuracy_score(Y_credit_teste, previsoes)

previsores = ['income', 'age', 'loan']

figura2, eixos = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
tree.plot_tree(arvore_credit, feature_names=previsores, class_names=['0','1'], filled=True)

figura2.savefig('teste.pdf')

########################## Base Census
with open('census.pkl', 'rb') as f:
    X_census_treinamento, X_census_teste, y_census_treinamento, y_census_teste = pickle.load(f)

arvore_census = DecisionTreeClassifier(criterion='entropy')

arvore_census.fit(X_census_treinamento, y_census_treinamento)

previsoes = arvore_census.predict(X_census_teste)

accuracy_score(y_census_teste, previsoes)



from yellowbrick.classifier import ConfusionMatrix

cm = ConfusionMatrix(arvore_census)
cm.fit(X_census_treinamento, y_census_treinamento)
cm.score(X_census_teste, y_census_teste)

print(classification_report(y_census_teste, previsoes))

######################### RANDOM FOREST ###############################
from sklearn.ensemble import RandomForestClassifier

# Base credit data

with open('credit.pkl', 'rb') as f:
    X_credit_treinamento, X_credit_teste, y_credit_treinamento, Y_credit_teste = pickle.load(f)


random_forest_credit = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)

random_forest_credit.fit(X_credit_treinamento, y_credit_treinamento)

previsoes = random_forest_credit.predict(X_credit_teste)

accuracy_score(Y_credit_teste, previsoes)

print(classification_report(Y_credit_teste, previsoes))

# Base census
with open('census.pkl', 'rb') as f:
    X_census_treinamento, X_census_teste, y_census_treinamento, y_census_teste = pickle.load(f)


random_forest_census = RandomForestClassifier(n_estimators=100000, criterion='entropy', random_state=0)
random_forest_census.fit(X_census_treinamento, y_census_treinamento)
previsoes = random_forest_census.predict(X_census_teste)
accuracy_score(y_census_teste, previsoes)
print(classification_report(y_census_teste, previsoes))


################################# AULA 06 (KNN) #################################
from sklearn.neighbors import KNeighborsClassifier

import pickle

# Base credit data
with open('credit.pkl', 'rb') as f:
    X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = pickle.load(f)
    
knn_credit = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_credit.fit(X_credit_treinamento, y_credit_treinamento)

previsoes = knn_credit.predict(X_credit_teste)


from sklearn.metrics import accuracy_score, classification_report
accuracy_score(Y_credit_teste, previsoes)
print(classification_report(Y_credit_teste, previsoes))



# Base census
with open('census.pkl', 'rb') as f:
    X_census_treinamento, X_census_teste, y_census_treinamento, y_census_teste = pickle.load(f)
    
knn_census = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_census.fit(X_census_treinamento, y_census_treinamento)

previsoes = knn_census.predict(X_census_teste)

accuracy_score(y_census_teste, previsoes)
print(classification_report(y_census_teste, previsoes))

######################## SVM ########################
from sklearn.svm import SVC

with open('credit.pkl', 'rb') as f:
    X_credit_treinamento, X_credit_teste, y_credit_treinamento, Y_credit_teste = pickle.load(f)
    
svm_credit = SVC(C=5, kernel='rbf') # Se atentar ao parametro C
svm_credit.fit(X_credit_treinamento, y_credit_treinamento)

previsoes = svm_credit.predict(X_credit_teste)


from sklearn.metrics import accuracy_score, classification_report
accuracy_score(Y_credit_teste, previsoes)
print(classification_report(Y_credit_teste, previsoes))

#
with open('census.pkl', 'rb') as f:
    X_census_treinamento, X_census_teste, y_census_treinamento, y_census_teste = pickle.load(f)
    
svm_census = SVC(C=5, kernel='rbf') # Se atentar ao parametro C
svm_census.fit(X_census_treinamento, y_census_treinamento)

previsoes = svm_census.predict(X_census_teste)


from sklearn.metrics import accuracy_score, classification_report
accuracy_score(y_census_teste, previsoes)
print(classification_report(Y_census_teste, previsoes))


################################### REDE NEURAL ###################################
from sklearn.neural_network import MLPClassifier

with open('credit.pkl', 'rb') as f:
    X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = pickle.load(f)
    
rede_neural_credit = MLPClassifier(max_iter=1000, verbose=True)
rede_neural_credit.fit(X_credit_treinamento, y_credit_treinamento)

previsoes = rede_neural_credit.predict(X_credit_teste)


from sklearn.metrics import accuracy_score, classification_report
accuracy_score(Y_credit_teste, previsoes)
print(classification_report(Y_credit_teste, previsoes))

#
with open('census.pkl', 'rb') as f:
    X_census_treinamento, X_census_teste, y_census_treinamento, y_census_teste = pickle.load(f)
    
rede_neural_census = MLPClassifier(max_iter=1000, verbose=True)
rede_neural_census.fit(X_census_treinamento, y_census_treinamento)

previsoes = rede_neural_census.predict(X_census_teste)


from sklearn.metrics import accuracy_score, classification_report
accuracy_score(y_census_teste, previsoes)
print(classification_report(Y_census_teste, previsoes))


## Trabalho: Mecher na base, pre processamento, usar todos os algoritmos e fazer um gráfico com os resultados -> Para proxima aula



################################################ AULA 07 #####################################
# GRIDSEARCH (Testar os algoritmos com vários parametor) (Pode melhorar um pouco na precisão)
# TUNING de algorítmos

# Validação cruzada, cobre toda a base de dados na questão de treinamento e teste

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


import pickle


with open('credit.pkl', 'rb') as f:
    X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = pickle.load(f)

X_credit = np.concatenate((X_credit_teste, X_credit_teste), axis=0)

y_credit = np.concatenate((y_credit_teste, y_credit_teste), axis=0)


## Arvore de decisao
DecisionTreeClassifier()
parametros = {'criterion': ['gini', 'entropy'],
              'splitter': ['best', 'random'],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 5, 10]}

grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=parametros)
grid_search.fit(X_credit, y_credit)

melhores_resultados = grid_search.best_score_
melhores_parametros = grid_search.best_params_


## Random Forest
parametros = {'criterion': ['gini', 'entropy'],
              'n_estimators': [10, 40, 100, 150, 1000],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 5, 10]}

grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parametros)
grid_search.fit(X_credit, y_credit)

melhores_resultados = grid_search.best_score_
melhores_parametros = grid_search.best_params_


## KNN
parametros = {'n_neighbors': [3, 5, 10 20],
              'p': [1, 2]}
 
grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=parametros)
grid_search.fit(X_credit, y_credit)

melhores_resultados = grid_search.best_score_
melhores_parametros = grid_search.best_params_

## SVM
parametros = {'tol': [0.001, 0.0001, 0.00001],
              'C': [1, 1.5, 2],
              'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}
 
grid_search = GridSearchCV(estimator=SVC(), param_grid=parametros)
grid_search.fit(X_credit, y_credit)

melhores_resultados = grid_search.best_score_
melhores_parametros = grid_search.best_params_

## REDES Neurais
parametros = {'activation': ['relu', 'logistic', 'tahn'],
              'solver': ['adam', 'sgd'],
              'batch_size': [10, 60]}
 
grid_search = GridSearchCV(estimator=MLPClassifier(), param_grid=parametros)
grid_search.fit(X_credit, y_credit)

melhores_resultados = grid_search.best_score_
melhores_parametros = grid_search.best_params_



