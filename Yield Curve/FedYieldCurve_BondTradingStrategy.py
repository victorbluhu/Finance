# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 23:44:15 2023

@author: Victor
"""
# %% Preâmbulo de Importação

import os

# Para simplificar, estou rodando o pacote de yc sem ser um pacote de fato e
### importado como um script normal
os.chdir(r'C:\Users\Victor\Dropbox\Python Scripts\Repositories\Finance\Yield Curve')

import pandas as pd
import datetime as dt
import YieldCurveLib as yc
import matplotlib.pyplot as mp

os.chdir(r'C:\Users\Victor\Dropbox\Python Scripts\Repositories\Finance')

# %% Monta a curva
params = yc.montaParametrosGSKdoFED(tipo = 'nominal')

min_date = dt.datetime(1987,1,1)

fullCurve = yc.fetchCurvaGSKParametrizada(
        tipo = 'nominal', max_maturity = 360,
        mensal = True,
        min_date = min_date,
        # max_date = dt.datetime(2012,1,31)
        )

fullModel = yc.ACMcomClasse(fullCurve)

# %% 
"""Sabemos que os componentes principais estimados no modelo ACM precificam
retornos da curva, mas eles foram estimados utilizando a sample completa.
Portanto, faremos estimativas mensais do modelo, calcularemos o retorno esperado
para o mês seguinte nas maturidades selecionados e operaremos portfolios long-short
pelo rankeamento de retornos esperados.
"""

min_estimation_length = 240
first_date = fullCurve.dates[min_estimation_length-1]

last_date = fullCurve.dates[-1]

expectedReturns = pd.DataFrame()

for positionDate in fullCurve.dates[
        min_estimation_length-1:
            ]:
    
    positionCurve = yc.fetchCurvaGSKParametrizada(
            tipo = 'nominal', max_maturity = 360,
            mensal = True,
            min_date = min_date,
            max_date = positionDate
            )

    positionModel = yc.ACMcomClasse(positionCurve)
    positionModel.estimateNextExpectedReturn()
    
    expectedReturns = pd.concat([
        expectedReturns,
        positionModel.NextExpectedReturn
        ])
    
expectedReturns / expectedReturns.columns * 12

expectedReturns.max(axis = 0)

YieldCurveObject = yc.fetchCurvaGSKParametrizada(
        tipo = 'nominal', max_maturity = 360,
        mensal = True,
        min_date = dt.datetime(1987,1,1),
        # max_date = dt.datetime(2012,1,31)
        )

full = yc.ACMcomClasse(YieldCurveObject)

modeloACM.state_df
