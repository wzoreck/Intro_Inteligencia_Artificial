#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade 01

Created on Thu Sep  8 13:22:17 2022

@author: daniel
"""

import pandas as pandas
import numpy as numpy
import seaborn as seaborn
import matplotlib.pyplot as pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

base_car_evaluation = pandas.read_csv('car_data.csv')

base_car_evaluation

base_car_evaluation.describe()

seaborn.countplot(x=base_car_evaluation['buying'])
seaborn.countplot(x=base_car_evaluation['maint'])

pyplot.hist(x=base_car_evaluation['buying'])
pyplot.hist(x=base_car_evaluation['maint'])
pyplot.hist(x=base_car_evaluation['doors'])
pyplot.hist(x=base_car_evaluation['persons'])
pyplot.hist(x=base_car_evaluation['lug_boot'])
pyplot.hist(x=base_car_evaluation['safety'])
pyplot.hist(x=base_car_evaluation['class'])

seaborn.pairplot(base_car_evaluation, vars=['persons', 'doors'])

base_car_evaluation.isnull().sum()

# Divisão da base (Previsores e Classes)
X_car_attributes = base_car_evaluation.iloc[:, 0:6].values
X_car_attributes

Y_car_class = base_car_evaluation.iloc[:, 6].values
Y_car_class

#Normalização
label_encoder = LabelEncoder()

indices = [0, 1, 2, 3, 4, 5]

for i in indices:
    X_car_attributes[:, i] = label_encoder.fit_transform(X_car_attributes[:, i])

#scaler_evaluation = MinMaxScaler()

#X_car_attributes = scaler_evaluation.fit_transform(X_car_attributes)

# TIRAR DÚVIDA COM O PROFESSOR!
from sklearn.preprocessing import StandardScaler

scaler_car_valuation = StandardScaler()

X_car_attributes = scaler_car_valuation.fit_transform(X_car_attributes)

# Divisão bases Treinamento e Teste
from sklearn.model_selection import train_test_split

X_car_attributes_TREINAMENTO, X_car_attributes_TESTE, Y_car_class_TREINAMENTO, Y_car_class_TESTE = train_test_split(X_car_attributes, Y_car_class, test_size=0.25, random_state=0)

import pickle

with open('car_valuation.pkl', mode='wb') as f:
    pickle.dump([X_car_attributes_TREINAMENTO, X_car_attributes_TESTE, Y_car_class_TREINAMENTO, Y_car_class_TESTE], f)

