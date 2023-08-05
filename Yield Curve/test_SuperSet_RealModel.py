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

import math
import pandas as pd
import numpy as np

# %% Monta as curvas
os.chdir(r'C:\Users\Victor\Dropbox\Python Scripts\Repositories\Finance')
# params = yc.montaParametrosGSKdoFED(tipo = 'nominal')
# paramsReal = yc.montaParametrosGSKdoFED(tipo = 'tips')

nominal = yc.fetchCurvaGSKParametrizada(
        tipo = 'nominal', max_maturity = 360,
        mensal = True,
        min_date = dt.datetime(1989,12,1),
        max_date = dt.datetime(2023,6,30)
        )

tips = yc.fetchCurvaGSKParametrizada(
        tipo = 'tips', max_maturity = 360,
        mensal = True,
        min_date = dt.datetime(1989,12,1),
        max_date = dt.datetime(2023,6,30)
        )

# Inflation returns
CPIU = pd.read_csv(
    'Data\CPIAUCNS.csv', index_col = 0, parse_dates=True
    )
CPIU_returns = ((1 + CPIU.pct_change())**12*100-100).shift(-1)
CPIULogReturns = (CPIU.applymap(math.log).diff()*12*100).shift(-1)

tips.getTipsNominalLogReturns(CPIULogReturns)

tips.tipsNominalLogReturns

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


# %% Modelo AACM
# Decomposing Real and Nominal Yield Curves - Staff Reports
# Michael Abrahams, Tobias Adrian, Richard K. Crump, and Emanuel Moench

AACM = yc.AACMcomClasse(CurveSet)
state_matrix = AACM.state_matrix

Y = AACM.nominalCurve.FixedRate.loc[AACM.estimateStep1Dates]
X = np.hstack([
    AACM.getAuxColOnes(len(AACM.estimateStep1Dates)),
    state_matrix.loc[AACM.estimateStep1Dates].values
    ])

deltas = Y.values.T @ np.linalg.pinv(X.T)


AACM.estimate()
AACM.estimatePi()
AACM.fitTipsYieldsFromPi(np.hstack([AACM.pi_0, AACM.pi_1]).flatten())
AACM.tipsCurve.LogYields
# AACM.getPhiArray()

# import numpy as np
# for x in AACM.B:
#     print(x[np.newaxis,:].shape)



# %%
