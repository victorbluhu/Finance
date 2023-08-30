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

Embutido no modelo, temos o forecast linear, que é uma porcaria para um mês de horizonte.
Além disos, um random forest com baggin e janela expansiva permite aumentar a variância
do retorno esperado, mas não ajuda.

O próximo passo é configurar o modelo para estimar retornos esperados para horizontes maiores
do que 1 mês (2, 3, 6 e 12 meses).
A experiência em estimações anteriores (e a teoria) implicam maior estabilidade nos prêmios
de risco estimados para horizontes mais longos. Além disso, com horizontes mais longos,
é razoável esperar que a razão sinal-ruído seja mais favorável a forecasts.
"""

min_estimation_length = 240
first_date = fullCurve.dates[min_estimation_length-1]

last_date = fullCurve.dates[-1]

# expectedReturnsProjection = pd.DataFrame()
# expectedReturnsForecast = pd.DataFrame()
# expectedReturnsForecast_RandomForest = pd.DataFrame()

expectedERPAnual_RandomForest = pd.DataFrame()
expectedERPAnual_RandomForestIteration = pd.DataFrame()

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
    # positionModel.projectNextExpectedReturn()
    # positionModel.forecastNextExpectedReturn()
    positionModel.forecastNextExpectedReturn_RandomForest(anual = True)
    expectedERPAnual_RandomForest = pd.concat([
        expectedERPAnual_RandomForest,
        positionModel.anual_nextExpectedReturnForecast_RandomForest.copy()
        ])
    
    positionModel.forecastNextExpectedReturn_RandomForest(
        anual = True, iteratedCovariates=True
        )
    expectedERPAnual_RandomForestIteration = pd.concat([
        expectedERPAnual_RandomForestIteration,
        positionModel.anual_nextExpectedReturnForecast_RandomForest.copy()
        ])
    # expectedReturnsProjection = pd.concat([
    #     expectedReturnsProjection,
    #     positionModel.nextExpectedReturnProjection
    #     ])
    # expectedReturnsForecast = pd.concat([
    #     expectedReturnsForecast,
    #     positionModel.nextExpectedReturnForecast
    #     ])
    # expectedReturnsForecast_RandomForest = pd.concat([
    #     expectedReturnsForecast_RandomForest,
    #     positionModel.nextExpectedReturnForecast_RandomForest
    #     ])
    
    

# %% Métricas de Performance

# expectedReturnsForecast_RandomForest[[12,24,60,120]].plot(figsize = (14,7))
selected_maturity = 60
plot = pd.concat([
 fullCurve.LogRXsAnuais[[selected_maturity]],
 expectedERPAnual_RandomForest[[selected_maturity]],
 expectedERPAnual_RandomForestIteration[[selected_maturity]],
 ], axis = 1)
plot.columns = ['Realizado', 'Predito RF', 'Predito RF Iterado']
ax = plot.loc[plot.index.year >= 2007].plot(
    figsize = (14,7), grid = True, 
    fontsize = 12, lw = 2
    )

ax.set_title(
    f'Excesso de Retorno - Vértice de {selected_maturity} vs 12 meses',
    fontsize = 14)

mp.show()

expectedERPAnual_RandomForestIteration[[24,36,60,120]].dropna().plot(figsize = (14,7))

# expectedReturns / expectedReturns.columns * 12

# expectedReturns.max(axis = 0)

# YieldCurveObject = yc.fetchCurvaGSKParametrizada(
#         tipo = 'nominal', max_maturity = 360,
#         mensal = True,
#         min_date = dt.datetime(1987,1,1),
#         # max_date = dt.datetime(2012,1,31)
#         )

# full = yc.ACMcomClasse(YieldCurveObject)

# modeloACM.state_df
