# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 23:44:15 2023

@author: Victor
"""
# %% Preâmbulo de Importação

import os

# Para simplificar, estou rodando o pacote de yc sem ser um pacote de fato e
### importado como um script normal
os.chdir(r'C:\Users\Victor\Dropbox\Python Scripts\Repositories\Yield Curve')

import datetime as dt
import YieldCurveLib as yc
import matplotlib.pyplot as mp


# %% Monta a curva
params = yc.montaParametrosGSKdoFED(tipo = 'nominal')

YieldCurveObject = yc.fetchCurvaGSKParametrizada(
        tipo = 'nominal', max_maturity = 360,
        mensal = True,
        min_date = dt.datetime(1989,12,1),
        max_date = dt.datetime(2023,12,31)
        )

ax = YieldCurveObject.plotCurva('Yields', YieldCurveObject.dates[-5:])
# Optional parameters for ax
# ax.set_title('...', fontsize = 14)
mp.show()


modeloACM = yc.ACMcomClasse(YieldCurveObject)
