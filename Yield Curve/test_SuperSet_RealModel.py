# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 22:32:45 2023

@author: Victor
"""
# %% Preâmbulo de Importação

import os

# Para simplificar, estou rodando o pacote de yc sem ser um pacote de fato e
### importado como um script normal
os.chdir(r'C:\Users\Victor\Dropbox\Python Scripts\Repositories\Finance\Yield Curve')

import datetime as dt
import YieldCurveLib as yc
import matplotlib.pyplot as mp

# %% Monta as curvas
os.chdir(r'C:\Users\Victor\Dropbox\Python Scripts\Repositories\Finance')
# params = yc.montaParametrosGSKdoFED(tipo = 'nominal')
# paramsReal = yc.montaParametrosGSKdoFED(tipo = 'tips')

nominal = yc.fetchCurvaGSKParametrizada(
        tipo = 'nominal', max_maturity = 360,
        mensal = True,
        min_date = dt.datetime(1989,12,1),
        max_date = dt.datetime(2023,12,31)
        )

tips = yc.fetchCurvaGSKParametrizada(
        tipo = 'tips', max_maturity = 360,
        mensal = True,
        min_date = dt.datetime(1989,12,1),
        max_date = dt.datetime(2023,12,31)
        )

# ax = YieldCurveObject.plotCurva('Yields', YieldCurveObject.dates[-5:])
# # Optional parameters for ax
# # ax.set_title('...', fontsize = 14)
# mp.show()


# modeloACM = yc.ACMcomClasse(YieldCurveObject)

# %%
CurveSet = yc.YieldCurveSet()
CurveSet.addYieldCurve(nominal, 'nominal')
CurveSet.addYieldCurve(tips, 'tips')

CurveSet.addBreakevenCurve('nominal', 'tips')

CurveSet.breakeven.Yields.loc[
    CurveSet.breakeven.Yields.index[-2:]
    ].dropna().T.plot(
    figsize = (14,7), grid = True
    )

axes_dict = CurveSet.plotCurvasDate(
    'Yields',
    [dt.datetime(2023,1,31),dt.datetime(2023,2,28)],
    title_string = 'Estrutura a Termo')

# %% Modelo ACM
# Decomposing Real and Nominal Yield Curves - Staff Reports
# Michael Abrahams, Tobias Adrian, Richard K. Crump, and Emanuel Moench


