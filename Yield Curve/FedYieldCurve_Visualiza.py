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
        # max_date = dt.datetime(2012,1,31)
        )

dates = [dt.datetime(2022,12,31), YieldCurveObject.dates[-1]]
ax = YieldCurveObject.plotCurva('Yields', dates)
# Optional parameters for ax
ax.set_title('Estrutura a Termo de Yields', fontsize = 14)
ax.get_figure().savefig(
    r'Yield Curve\Figures\Yields.jpg', bbox_inches = 'tight'
    )
mp.show()

ax = YieldCurveObject.plotCurva('FRAs', dates)
# Optional parameters for ax
ax.set_title('Estrutura a Termo de FRAs de 1 ano', fontsize = 14)
ax.get_figure().savefig(
    r'Yield Curve\Figures\FRAs.jpg', bbox_inches = 'tight'
    )
mp.show()


# %% plota a decomposição da última curva
# modeloACM.plotDecomposicao(path = r'Yield Curve\Figures\Decomposicao da Curva.jpg')