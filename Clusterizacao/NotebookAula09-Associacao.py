# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 15:41:34 2022

@author: elgom
"""
#pip install apyori

import pandas as pd
from apyori import apriori

## Base de dados mercado 1

base_mercado1 = pd.read_csv('mercado.csv', header = None)


transacoes = []
for i in range(len(base_mercado1)):
  transacoes.append([str(base_mercado1.values[i, j]) for j in range(base_mercado1.shape[1])])


regras = apriori(transacoes, min_support = 0.3, min_confidence = 0.8, min_lift = 2)
resultados = list(regras)

resultados

len(resultados)

resultados[2]


A = []
B = []
suporte = []
confianca = []
lift = []

for resultado in resultados:
  s = resultado[1]
  result_rules = resultado[2]
  for result_rule in result_rules:
    a = list(result_rule[0])
    b = list(result_rule[1])
    c = result_rule[2]
    l = result_rule[3]
    A.append(a)
    B.append(b)
    suporte.append(s)
    confianca.append(c)
    lift.append(l)


rules_df = pd.DataFrame({'A': A, 'B': B, 'suporte': suporte, 'confianca': confianca, 'lift': lift})

rules_df = rules_df.sort_values(by = 'lift', ascending = False)