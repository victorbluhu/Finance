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

YieldCurveObject = yc.fetchCurvaGSKParametrizada(
        tipo = 'nominal', max_maturity = 360,
        mensal = True,
        min_date = dt.datetime(1987,1,1),
        max_date = dt.datetime(2012,1,31)
        )

ax = YieldCurveObject.plotCurva('Yields', YieldCurveObject.dates[-5:])
# Optional parameters for ax
# ax.set_title('...', fontsize = 14)
mp.show()


modeloACM = yc.ACMcomClasse(YieldCurveObject)

print(modeloACM.diagnosticsTable)

# %%
params = yc.montaParametrosGSKdoFED(tipo = 'nominal')

YieldCurveObject = yc.fetchCurvaGSKParametrizada(
        tipo = 'nominal', max_maturity = 360,
        mensal = True,
        min_date = dt.datetime(1987,1,1),
        # max_date = dt.datetime(2012,1,31)
        )

ax = YieldCurveObject.plotCurva('Yields', YieldCurveObject.dates[-5:])
mp.show()


modeloACM = yc.ACMcomClasse(YieldCurveObject)

# print(modeloACM.diagnosticsTable)

# %% plota a decomposição da última curva
columns = [12, 24, 60, 120]
ax = modeloACM.BRP[columns].rename({i: i//12 for i in columns}, axis = 1).plot(
    figsize = (14,7),
    grid = True, lw = 2,
    title = r'Bond Risk Premia - Anos Selecionados'
    )
ax.get_figure().savefig(
    r'Yield Curve\Figures\BRP - Anos Selecionados.jpg',
    bbox_inches = 'tight'
    )

# %% plota a decomposição da última curva
modeloACM.plotDecomposicao(path = r'Yield Curve\Figures\Decomposicao da Curva.jpg')